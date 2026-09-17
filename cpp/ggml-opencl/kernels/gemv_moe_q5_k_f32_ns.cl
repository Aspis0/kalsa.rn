#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define K_SCALE_SIZE 12
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

// Pass the bitmasks as kernel arguments instead of literal constants: the
// Qualcomm Adreno E031 compiler miscompiles the literal-mask form of this
// helper (device-verified on Adreno 740: all MUL_MAT_ID q4_K/q5_K cases fail
// with literals, pass with arguments; same workaround as the dense noshuffle
// kernels).
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

static inline float8 q5_k_to_fp32_packed8(ushort2 qs5x8, uchar qh5x8, half s, half m) {
    float8 fp32x8;
    fp32x8.s0 = (float)((( qs5x8.s0 & 0x000F)        | (( qh5x8       & 0x01) << 4)) * s + m);
    fp32x8.s1 = (float)((((qs5x8.s0 & 0x00F0) >> 4 ) | (((qh5x8 >> 1) & 0x01) << 4)) * s + m);
    fp32x8.s2 = (float)((((qs5x8.s0 & 0x0F00) >> 8 ) | (((qh5x8 >> 2) & 0x01) << 4)) * s + m);
    fp32x8.s3 = (float)((((qs5x8.s0 & 0xF000) >> 12) | (((qh5x8 >> 3) & 0x01) << 4)) * s + m);
    fp32x8.s4 = (float)((( qs5x8.s1 & 0x000F)        | (((qh5x8 >> 4) & 0x01) << 4)) * s + m);
    fp32x8.s5 = (float)((((qs5x8.s1 & 0x00F0) >> 4 ) | (((qh5x8 >> 5) & 0x01) << 4)) * s + m);
    fp32x8.s6 = (float)((((qs5x8.s1 & 0x0F00) >> 8 ) | (((qh5x8 >> 6) & 0x01) << 4)) * s + m);
    fp32x8.s7 = (float)((((qs5x8.s1 & 0xF000) >> 12) | (((qh5x8 >> 7) & 0x01) << 4)) * s + m);
    return fp32x8;
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q5_k_f32_ns(
    __global uint *         src0_q,
    __global uint *         src0_qh,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11,
    uchar                   mask_d6,
    uchar                   mask_d4,
    uchar                   mask_hi2
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    // All lanes must reach the reduction barrier; clamp padded lanes to a
    // valid row and predicate only the final store.
    const bool valid = (i01 < (uint)ne01);
    const uint i01_load = valid ? i01 : ((uint)ne01 - 1u);

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    // Expert offsets in the transposed noshuffle layout
    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float sum = 0.0f;

    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;
        uint j  = ib % 8;

        // Load d and dmin for this super-block
        half d_val   = src0_d[expert_d_offset + sb * ne01 + i01_load];
        half dm_val  = src0_dm[expert_d_offset + sb * ne01 + i01_load];

        // sub_block index = sb * 8 + j
        uint expert_qh_offset = expert_id * num_superblocks * 8 * ne01;
        uchar4 regQh = as_uchar4(src0_qh[expert_qh_offset + (sb * 8 + j) * ne01 + i01_load]);

        // Load sub-block scale and min
        global const uchar * sc = src0_s + (expert_id * ne01 + i01_load) * scales_per_row + sb * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(j, sc, &sv, &mn, mask_d6, mask_d4, mask_hi2);

        float scale = (float)d_val * (float)sv;
        float minv  = -(float)dm_val * (float)mn;

        // Load 4 uints of quants (32 nibbles = 32 elements)
        uint q_base = expert_q_offset + ib * ne01 * 4 + i01_load;

        uint4 regQ;
        regQ.s0 = src0_q[q_base];
        regQ.s1 = src0_q[q_base + ne01];
        regQ.s2 = src0_q[q_base + ne01 * 2];
        regQ.s3 = src0_q[q_base + ne01 * 3];

        // Load activations: 32 floats = 8 float4s
        uint y_offset = i11 * ne00 / 4 + ib * 8;

        float8 fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s0), regQh.s0, scale, minv);

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (y_offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s1), regQh.s1, scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 3));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s2), regQh.s2, scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 5));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s3), regQh.s3, scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 6));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 7));
        acc += shared_y4 * fp32x8.hi;

        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
    }

    // reduction in local memory, assumes #subgroups=4
    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    // 1 output per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        if (valid) {
            dst[i01 + i20 * ne01] = sum;
        }
    }
}
