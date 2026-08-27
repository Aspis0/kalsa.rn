// KALSA consumer-side oracle: GEMM idiom for q4_0.
// Concatenated AFTER gemm_moe_q4_0_f32_ns.cl so dequantize_q4_0 is the exact
// macro text the real MUL_MAT_ID GEMM consumer compiles.
// No printf. gid is local to a host chunk; sblk = block_begin + gid (AoS).

static inline void oracle_q4_0_gemm_store_raw(uint4 regQ, half s, uint n_loc, __global uint * dst_raw, __global ushort * dst_d)
{
    dst_raw[n_loc * 4 + 0] = regQ.s0;
    dst_raw[n_loc * 4 + 1] = regQ.s1;
    dst_raw[n_loc * 4 + 2] = regQ.s2;
    dst_raw[n_loc * 4 + 3] = regQ.s3;
    dst_d[n_loc] = as_ushort(s);
}

static inline void oracle_q4_0_gemm_dequant_block(uint4 regQ, half s, uint n_loc, __global float * dst_f32)
{
    const uint fbase = n_loc * 32;
    half16 reg_a;
    uint2 q4x16;

    q4x16.x = regQ.s0;
    q4x16.y = regQ.s1;
    dequantize_q4_0(as_ushort4(q4x16), reg_a, s);
    dst_f32[fbase +  0] = (float)reg_a.s0;
    dst_f32[fbase +  1] = (float)reg_a.s1;
    dst_f32[fbase +  2] = (float)reg_a.s2;
    dst_f32[fbase +  3] = (float)reg_a.s3;
    dst_f32[fbase +  4] = (float)reg_a.s4;
    dst_f32[fbase +  5] = (float)reg_a.s5;
    dst_f32[fbase +  6] = (float)reg_a.s6;
    dst_f32[fbase +  7] = (float)reg_a.s7;
    dst_f32[fbase +  8] = (float)reg_a.s8;
    dst_f32[fbase +  9] = (float)reg_a.s9;
    dst_f32[fbase + 10] = (float)reg_a.sa;
    dst_f32[fbase + 11] = (float)reg_a.sb;
    dst_f32[fbase + 12] = (float)reg_a.sc;
    dst_f32[fbase + 13] = (float)reg_a.sd;
    dst_f32[fbase + 14] = (float)reg_a.se;
    dst_f32[fbase + 15] = (float)reg_a.sf;

    q4x16.x = regQ.s2;
    q4x16.y = regQ.s3;
    dequantize_q4_0(as_ushort4(q4x16), reg_a, s);
    dst_f32[fbase + 16] = (float)reg_a.s0;
    dst_f32[fbase + 17] = (float)reg_a.s1;
    dst_f32[fbase + 18] = (float)reg_a.s2;
    dst_f32[fbase + 19] = (float)reg_a.s3;
    dst_f32[fbase + 20] = (float)reg_a.s4;
    dst_f32[fbase + 21] = (float)reg_a.s5;
    dst_f32[fbase + 22] = (float)reg_a.s6;
    dst_f32[fbase + 23] = (float)reg_a.s7;
    dst_f32[fbase + 24] = (float)reg_a.s8;
    dst_f32[fbase + 25] = (float)reg_a.s9;
    dst_f32[fbase + 26] = (float)reg_a.sa;
    dst_f32[fbase + 27] = (float)reg_a.sb;
    dst_f32[fbase + 28] = (float)reg_a.sc;
    dst_f32[fbase + 29] = (float)reg_a.sd;
    dst_f32[fbase + 30] = (float)reg_a.se;
    dst_f32[fbase + 31] = (float)reg_a.sf;
}

__kernel void oracle_q4_0_gemm_buffer(
    __global uint * src0_q,
    __global half * src0_d,
    __global float * dst_f32,
    __global uint * dst_raw,
    int ne00,
    int ne01,
    int ne02,
    uint block_begin,
    __global ushort * dst_d)
{
    const int nb = ne00 / 32;
    const uint nblocks = (uint)nb * (uint)ne01 * (uint)ne02;
    const uint n_loc = (uint)get_global_id(0);
    const uint sblk = block_begin + n_loc;
    if (sblk >= nblocks) {
        return;
    }

    const uint i00 = sblk % (uint)nb;
    const uint tmp = sblk / (uint)nb;
    const uint i01 = tmp % (uint)ne01;
    const uint i02 = tmp / (uint)ne01;

    const uint expert_offset = i02 * (uint)nb * (uint)ne01;
    const uint block_offset = expert_offset * 4u + i00 * (uint)ne01 * 4u + i01;

    uint4 regQ;
    regQ.s0 = src0_q[block_offset];
    regQ.s1 = src0_q[block_offset + (uint)ne01];
    regQ.s2 = src0_q[block_offset + (uint)ne01 * 2u];
    regQ.s3 = src0_q[block_offset + (uint)ne01 * 3u];

    const half s = src0_d[i00 * (uint)ne01 + i01 + expert_offset];
    oracle_q4_0_gemm_store_raw(regQ, s, n_loc, dst_raw, dst_d);
    oracle_q4_0_gemm_dequant_block(regQ, s, n_loc, dst_f32);
}

__kernel void oracle_q4_0_gemm_image(
    __read_only image1d_buffer_t src0_q,
    __global half * src0_d,
    __global float * dst_f32,
    __global uint * dst_raw,
    int ne00,
    int ne01,
    int ne02,
    uint block_begin,
    __global ushort * dst_d)
{
    const int nb = ne00 / 32;
    const uint nblocks = (uint)nb * (uint)ne01 * (uint)ne02;
    const uint n_loc = (uint)get_global_id(0);
    const uint sblk = block_begin + n_loc;
    if (sblk >= nblocks) {
        return;
    }

    const uint i00 = sblk % (uint)nb;
    const uint tmp = sblk / (uint)nb;
    const uint i01 = tmp % (uint)ne01;
    const uint i02 = tmp / (uint)ne01;

    const uint expert_offset = i02 * (uint)nb * (uint)ne01;
    const uint block_offset = expert_offset * 4u + i00 * (uint)ne01 * 4u + i01;

    uint4 regQ;
    regQ.s0 = read_imageui(src0_q, (int)(block_offset)).x;
    regQ.s1 = read_imageui(src0_q, (int)(block_offset + (uint)ne01)).x;
    regQ.s2 = read_imageui(src0_q, (int)(block_offset + (uint)ne01 * 2u)).x;
    regQ.s3 = read_imageui(src0_q, (int)(block_offset + (uint)ne01 * 3u)).x;

    const half s = src0_d[i00 * (uint)ne01 + i01 + expert_offset];
    oracle_q4_0_gemm_store_raw(regQ, s, n_loc, dst_raw, dst_d);
    oracle_q4_0_gemm_dequant_block(regQ, s, n_loc, dst_f32);
}
