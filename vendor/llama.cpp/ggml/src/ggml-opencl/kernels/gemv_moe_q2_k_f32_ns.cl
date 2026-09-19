#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

// ORACLE=GEMV unpack — identical extracts to kernel_dequant_row_q2_k_trans4_ns
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q2_k_f32_ns(
    __global uint *         src0_q,
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
    uchar                   mask_0F,
    uchar                   mask_03
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    // unlike the q4/q5/q6 clones, no early return: all lanes must reach the
    // barriers (gate admits ne01%32, grid pads to 64)
    const bool valid = (i01 < (uint)ne01);
    const uint i01_load = valid ? i01 : ((uint)ne01 - 1u);

    uint m0F = (uint)mask_0F;
    uint m03 = (uint)mask_03;
    uint i11 = i20 % (uint)ne11;
    uint expert_id = src2[i20];
    uint n_sb = (uint)ne00 / QK_K;
    uint expert_d_offset = expert_id * n_sb * (uint)ne01;
    uint expert_q_offset = expert_id * ((uint)ne00 / 16) * (uint)ne01;
    uint s_row = (expert_id * (uint)ne01 + i01_load) * (n_sb * 16);

    __private float sum = 0.0f;

    for (uint sb = sgid; sb < n_sb; sb += N_SIMDGROUP) {
        // HOIST — same loads as oracle, once per super-block
        __private float d    = convert_float(src0_d [expert_d_offset + sb * (uint)ne01 + i01_load]);
        __private float dmin = convert_float(src0_dm[expert_d_offset + sb * (uint)ne01 + i01_load]);

        for (uint g = 0; g < 16; ++g) {
            uint sc = (uint)src0_s[s_row + sb * 16 + g];
            float dl = d    * convert_float(sc & m0F);
            float ml = dmin * convert_float((sc >> 4) & m0F);
            uint bits = src0_q[expert_q_offset + (sb * 16 + g) * (uint)ne01 + i01_load];

            uint y_off = i11 * ((uint)ne00 / 4) + sb * 64 + g * 4;
            float4 a0 = read_imagef(src1, (int)(y_off + 0));
            float4 a1 = read_imagef(src1, (int)(y_off + 1));
            float4 a2 = read_imagef(src1, (int)(y_off + 2));
            float4 a3 = read_imagef(src1, (int)(y_off + 3));
            // ORACLE=GEMV unpack — same as kernel_dequant_row_q2_k_trans4_ns
            // y = dl * convert_float((bits >> (2*k)) & m03) - ml
            float4 w0, w1, w2, w3;
            w0.s0 = dl * convert_float((bits >>  0) & m03) - ml;
            w0.s1 = dl * convert_float((bits >>  2) & m03) - ml;
            w0.s2 = dl * convert_float((bits >>  4) & m03) - ml;
            w0.s3 = dl * convert_float((bits >>  6) & m03) - ml;
            w1.s0 = dl * convert_float((bits >>  8) & m03) - ml;
            w1.s1 = dl * convert_float((bits >> 10) & m03) - ml;
            w1.s2 = dl * convert_float((bits >> 12) & m03) - ml;
            w1.s3 = dl * convert_float((bits >> 14) & m03) - ml;
            w2.s0 = dl * convert_float((bits >> 16) & m03) - ml;
            w2.s1 = dl * convert_float((bits >> 18) & m03) - ml;
            w2.s2 = dl * convert_float((bits >> 20) & m03) - ml;
            w2.s3 = dl * convert_float((bits >> 22) & m03) - ml;
            w3.s0 = dl * convert_float((bits >> 24) & m03) - ml;
            w3.s1 = dl * convert_float((bits >> 26) & m03) - ml;
            w3.s2 = dl * convert_float((bits >> 28) & m03) - ml;
            w3.s3 = dl * convert_float((bits >> 30) & m03) - ml;
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
