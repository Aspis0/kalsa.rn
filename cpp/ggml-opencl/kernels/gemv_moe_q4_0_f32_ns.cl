#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_Q4_0 32
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

// KALSA (2026-08-19): mask-then-shift is MISCOMPILED on Adreno 740 (driver 0676.73,
// compiler E031.41.03.62). `(v & 0xF000) >> 12` treats the masked value as signed and
// performs an ARITHMETIC shift, so whenever the top bit is set the sign propagates and
// the result carries spurious high bits; a following `& 0x0F` does not save it, because
// the compiler removes that mask as dead by known-bits (it assumes a logical shift).
// Measured on device with an in-run buffer readback: corruption occurred in exactly the
// cases where the source byte was >= 0x80, 16/16.
// SHIFT FIRST, THEN MASK -- which is the form kernel_gemv_moe_q6_k_f32_ns.cl already
// uses, and q6_K is the only nibble-packed MoE type that passes on this device (3/3).
static inline float8 q4_0_to_fp32_packed8(ushort2 q4x8) {
    float8 fp32x8;
    fp32x8.s0 = (float)((int)( q4x8.s0        & 0x000F) - 8);
    fp32x8.s1 = (float)((int)((q4x8.s0 >>  4) & 0x000F) - 8);
    fp32x8.s2 = (float)((int)((q4x8.s0 >>  8) & 0x000F) - 8);
    fp32x8.s3 = (float)((int)((q4x8.s0 >> 12) & 0x000F) - 8);
    fp32x8.s4 = (float)((int)( q4x8.s1        & 0x000F) - 8);
    fp32x8.s5 = (float)((int)((q4x8.s1 >>  4) & 0x000F) - 8);
    fp32x8.s6 = (float)((int)((q4x8.s1 >>  8) & 0x000F) - 8);
    fp32x8.s7 = (float)((int)((q4x8.s1 >> 12) & 0x000F) - 8);
    return fp32x8;
}


__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_0_f32_ns(
    __global uint * src0_q,
    __global half * src0_d,
    __read_only image1d_buffer_t src1,
    __global uint * src2,
    __global float * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    int           ne11
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float sum = 0.0f; // each thread calculate partial sum of one output

    // loop along ne00 in block granularity, skip 4 blocks every iter
    for (uint ib00 = sgid; ib00 < (ne00 / QK_Q4_0); ib00 += N_SIMDGROUP) {

        // load one block of q
        uint4 regQ;
        uint block_offset = expert_offset * 4 + ib00 * ne01 * 4 + i01;

        regQ.s0 = src0_q[block_offset];
        regQ.s1 = src0_q[block_offset + ne01];
        regQ.s2 = src0_q[block_offset + ne01 * 2];
        regQ.s3 = src0_q[block_offset + ne01 * 3];

        uint offset = i11 * ne00 / 4 + ib00 * 8;

        float8 fp32x8 = q4_0_to_fp32_packed8(as_ushort2(regQ.s0));

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_0_to_fp32_packed8(as_ushort2(regQ.s1));

        shared_y4 = read_imagef(src1, (offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (offset + 3));
        acc += shared_y4 * fp32x8.hi;


        fp32x8 = q4_0_to_fp32_packed8(as_ushort2(regQ.s2));

        shared_y4 = read_imagef(src1, (offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (offset + 5));
        acc += shared_y4 * fp32x8.hi;


        fp32x8 = q4_0_to_fp32_packed8(as_ushort2(regQ.s3));

        shared_y4 = read_imagef(src1, (offset + 6));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (offset + 7));
        acc += shared_y4 * fp32x8.hi;

        half regS = src0_d[ib00 * ne01 + i01 + expert_offset];
        sum += (float)(regS) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
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

    // 1 outputs per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }

}
