#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

// ORACLE=GEMV unpack — identical extracts to kernel_dequant_row_q3_k_trans4_ns
static inline int qhat_q3(uint bits, uint k, uint qh_w, uint lane, uint m03, uint m01) {
    uint q_u = (bits >> (2 * k)) & m03;
    uint hm  = (qh_w >> (lane + k)) & m01;
    int qhat = (int)q_u;
    if (hm == 0) {
        qhat = (int)q_u - 4;
    }
    return qhat;
}
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q3_k_f32_ns(
    __global uint *         src0_q,
    __global half *         src0_d,
    __global uchar *        src0_s,
    __global uint *         src0_qh,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11,
    uchar                   mask_03,
    uchar                   mask_01,
    float                   scale_zero
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    // unlike the q4/q5/q6 clones, no early return: all lanes must reach the
    // barriers (gate admits ne01%32, grid pads to 64)
    const bool valid = (i01 < (uint)ne01);
    const uint i01_load = valid ? i01 : ((uint)ne01 - 1u);

    uint m03 = (uint)mask_03;
    uint m01 = (uint)mask_01;
    uint i11 = i20 % (uint)ne11;
    uint expert_id = src2[i20];
    uint n_sb = (uint)ne00 / QK_K;
    uint expert_d_offset  = expert_id * n_sb * (uint)ne01;
    uint expert_q_offset  = expert_id * ((uint)ne00 / 16) * (uint)ne01;
    uint expert_qh_offset = expert_id * ((uint)ne00 / 32) * (uint)ne01;
    uint s_row = (expert_id * (uint)ne01 + i01_load) * (n_sb * 16);

    __private float sum = 0.0f;

    for (uint sb = sgid; sb < n_sb; sb += N_SIMDGROUP) {
        __private float d = convert_float(src0_d[expert_d_offset + sb * (uint)ne01 + i01_load]);

        for (uint g = 0; g < 16; ++g) {
            uint sc = (uint)src0_s[s_row + sb * 16 + g];
            float dl = d * (convert_float(sc) - scale_zero);
            uint bits = src0_q[expert_q_offset + (sb * 16 + g) * (uint)ne01 + i01_load];
            uint p    = g >> 1;
            uint hi   = g & 1;
            uint lane = hi << 4;
            uint qh_w = src0_qh[expert_qh_offset + (sb * 8 + p) * (uint)ne01 + i01_load];

            uint y_off = i11 * ((uint)ne00 / 4) + sb * 64 + g * 4;
            float4 a0 = read_imagef(src1, (int)(y_off + 0));
            float4 a1 = read_imagef(src1, (int)(y_off + 1));
            float4 a2 = read_imagef(src1, (int)(y_off + 2));
            float4 a3 = read_imagef(src1, (int)(y_off + 3));
            float4 w0, w1, w2, w3;
            w0.s0 = dl * convert_float(qhat_q3(bits,  0, qh_w, lane, m03, m01));
            w0.s1 = dl * convert_float(qhat_q3(bits,  1, qh_w, lane, m03, m01));
            w0.s2 = dl * convert_float(qhat_q3(bits,  2, qh_w, lane, m03, m01));
            w0.s3 = dl * convert_float(qhat_q3(bits,  3, qh_w, lane, m03, m01));
            w1.s0 = dl * convert_float(qhat_q3(bits,  4, qh_w, lane, m03, m01));
            w1.s1 = dl * convert_float(qhat_q3(bits,  5, qh_w, lane, m03, m01));
            w1.s2 = dl * convert_float(qhat_q3(bits,  6, qh_w, lane, m03, m01));
            w1.s3 = dl * convert_float(qhat_q3(bits,  7, qh_w, lane, m03, m01));
            w2.s0 = dl * convert_float(qhat_q3(bits,  8, qh_w, lane, m03, m01));
            w2.s1 = dl * convert_float(qhat_q3(bits,  9, qh_w, lane, m03, m01));
            w2.s2 = dl * convert_float(qhat_q3(bits, 10, qh_w, lane, m03, m01));
            w2.s3 = dl * convert_float(qhat_q3(bits, 11, qh_w, lane, m03, m01));
            w3.s0 = dl * convert_float(qhat_q3(bits, 12, qh_w, lane, m03, m01));
            w3.s1 = dl * convert_float(qhat_q3(bits, 13, qh_w, lane, m03, m01));
            w3.s2 = dl * convert_float(qhat_q3(bits, 14, qh_w, lane, m03, m01));
            w3.s3 = dl * convert_float(qhat_q3(bits, 15, qh_w, lane, m03, m01));
            sum += dot(w0, a0) + dot(w1, a1) + dot(w2, a2) + dot(w3, a3);
        }
    }

    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    if (valid && sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * (uint)ne01] = sum;
    }
}
