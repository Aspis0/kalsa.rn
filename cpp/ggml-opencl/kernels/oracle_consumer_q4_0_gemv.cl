// KALSA consumer-side oracle: GEMV idiom for q4_0.
// Concatenated AFTER gemv_moe_q4_0_f32_ns.cl so q4_0_to_fp32_packed8 is the
// exact function text the real MUL_MAT_ID GEMV consumer compiles.
// No printf. gid is local to a host chunk; sblk = block_begin + gid (AoS).

static inline void oracle_q4_0_gemv_store(
    uint4 regQ,
    half hs,
    uint n_loc,
    __global float * dst_f32,
    __global uint * dst_raw,
    __global ushort * dst_d)
{
    dst_raw[n_loc * 4 + 0] = regQ.s0;
    dst_raw[n_loc * 4 + 1] = regQ.s1;
    dst_raw[n_loc * 4 + 2] = regQ.s2;
    dst_raw[n_loc * 4 + 3] = regQ.s3;
    dst_d[n_loc] = as_ushort(hs);

    const float scale = (float)hs;
    const uint fbase = n_loc * 32;

    float8 a = q4_0_to_fp32_packed8(as_ushort2(regQ.s0));
    dst_f32[fbase +  0] = a.s0 * scale;
    dst_f32[fbase +  1] = a.s1 * scale;
    dst_f32[fbase +  2] = a.s2 * scale;
    dst_f32[fbase +  3] = a.s3 * scale;
    dst_f32[fbase +  4] = a.s4 * scale;
    dst_f32[fbase +  5] = a.s5 * scale;
    dst_f32[fbase +  6] = a.s6 * scale;
    dst_f32[fbase +  7] = a.s7 * scale;

    a = q4_0_to_fp32_packed8(as_ushort2(regQ.s1));
    dst_f32[fbase +  8] = a.s0 * scale;
    dst_f32[fbase +  9] = a.s1 * scale;
    dst_f32[fbase + 10] = a.s2 * scale;
    dst_f32[fbase + 11] = a.s3 * scale;
    dst_f32[fbase + 12] = a.s4 * scale;
    dst_f32[fbase + 13] = a.s5 * scale;
    dst_f32[fbase + 14] = a.s6 * scale;
    dst_f32[fbase + 15] = a.s7 * scale;

    a = q4_0_to_fp32_packed8(as_ushort2(regQ.s2));
    dst_f32[fbase + 16] = a.s0 * scale;
    dst_f32[fbase + 17] = a.s1 * scale;
    dst_f32[fbase + 18] = a.s2 * scale;
    dst_f32[fbase + 19] = a.s3 * scale;
    dst_f32[fbase + 20] = a.s4 * scale;
    dst_f32[fbase + 21] = a.s5 * scale;
    dst_f32[fbase + 22] = a.s6 * scale;
    dst_f32[fbase + 23] = a.s7 * scale;

    a = q4_0_to_fp32_packed8(as_ushort2(regQ.s3));
    dst_f32[fbase + 24] = a.s0 * scale;
    dst_f32[fbase + 25] = a.s1 * scale;
    dst_f32[fbase + 26] = a.s2 * scale;
    dst_f32[fbase + 27] = a.s3 * scale;
    dst_f32[fbase + 28] = a.s4 * scale;
    dst_f32[fbase + 29] = a.s5 * scale;
    dst_f32[fbase + 30] = a.s6 * scale;
    dst_f32[fbase + 31] = a.s7 * scale;
}

static inline void oracle_q4_0_gemv_body(
    uint4 regQ,
    half hs,
    uint n_loc,
    __global float * dst_f32,
    __global uint * dst_raw,
    __global ushort * dst_d)
{
    oracle_q4_0_gemv_store(regQ, hs, n_loc, dst_f32, dst_raw, dst_d);
}

__kernel void oracle_q4_0_gemv_buffer(
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

    // AoS block index sblk = i00 + i01*nb + i02*nb*ne01 (ggml / dequantize_row_q4_0).
    const uint i00 = sblk % (uint)nb;
    const uint tmp = sblk / (uint)nb;
    const uint i01 = tmp % (uint)ne01;
    const uint i02 = tmp / (uint)ne01;

    // GEMV consumer addressing (kernel_gemv_moe_q4_0_f32_ns):
    //   expert_offset = expert_id * ne00 * ne01 / 32
    //   block_offset  = expert_offset * 4 + ib00 * ne01 * 4 + i01
    const uint expert_offset = i02 * (uint)nb * (uint)ne01;
    const uint block_offset = expert_offset * 4u + i00 * (uint)ne01 * 4u + i01;

    uint4 regQ;
    regQ.s0 = src0_q[block_offset];
    regQ.s1 = src0_q[block_offset + (uint)ne01];
    regQ.s2 = src0_q[block_offset + (uint)ne01 * 2u];
    regQ.s3 = src0_q[block_offset + (uint)ne01 * 3u];

    const half hs = src0_d[i00 * (uint)ne01 + i01 + expert_offset];
    oracle_q4_0_gemv_body(regQ, hs, n_loc, dst_f32, dst_raw, dst_d);
}

__kernel void oracle_q4_0_gemv_image(
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

    const half hs = src0_d[i00 * (uint)ne01 + i01 + expert_offset];
    oracle_q4_0_gemv_body(regQ, hs, n_loc, dst_f32, dst_raw, dst_d);
}
