#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((intel_reqd_sub_group_size(64)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

// Q6_K x f32 GEMV over the GGUF AoS block layout, in place (the OpenCL HOST weight leg).
//
// The plain kernel_mul_mv_q6_K_f32 runs the tied lm_head at ~9.5 GB/s on the S23 (215 MB in
// 22.5 ms, measured with an OpenCL-profiling build), while kernel_mul_mv_q6_K_f32_flat reads
// the same weights at ~51 GB/s once they sit repacked in a GPU buffer. The bytes cannot be
// repacked here (one dma-buf is read by CPU, HTP and GPU), so this kernel keeps them where
// they are and removes what the plain kernel spends on them:
//   - its lane mapping is the transpose of the flat kernels': four CONSECUTIVE lanes take the
//     four block slots (tid = sglid/4, ix = sglid%4), so a subgroup's weight loads land in
//     four blocks 210 bytes apart instead of tiling one. Here tid = sglid%16 indexes the
//     block and ix = sglid/16 the block slot, as in the flat kernel.
//   - a subgroup covers 2 rows and re-reads the activation once per row, one scalar load per
//     weight. Here a subgroup covers Q6K_AOS_N_DST rows and the four rows of a row group
//     share one 16-value activation slice, read once per block.
//   - the 6-bit weights are rebuilt four at a time from one uint (two 2-byte loads) instead
//     of one byte at a time, and the -32 offset is paid once per quadrant as
//     -32*sum(y) instead of once per weight.
//
// Alignment: sizeof(block_q6_K) is 210 (ql[128], qh[64], scales[16], half d, 2 pad), so every
// ql/qh/scales/d address inside a row is 2-byte aligned at best: the leg's view clause pins
// the tensor offset to 16 bytes and both the row stride (nb*210) and the block stride keep it
// even, so the ushort loads below are defined and uint loads would not be. The dequant and
// the dot are scalar for the same reason the flat kernel's ADRENO_OLD_COMPILER path is: the
// E031.41.03.62 compiler of the target miscompiles the vector carries (convert_*4, dot(),
// vload4) of the flat kernels, and a kernel that lies about its numbers would be caught by
// the leg's self-check, not by a KLD run.

#define QK_K 256
#define Q6K_AOS_BYTES   210          // the AoS block, byte for byte as the GGUF stores it
#define Q6K_AOS_QH      128
#define Q6K_AOS_SCALES  192
#define Q6K_AOS_D       208
#define Q6K_AOS_N_DST   16           // rows a subgroup holds, four per row group
#define Q6K_AOS_N_SIMDGROUP 2
#define Q6K_AOS_N_SIMDWIDTH 64
#define Q6K_AOS_BLOCK_STRIDE (Q6K_AOS_N_SIMDWIDTH/16)   // block slots per subgroup

// Four consecutive bytes of a ql/qh array as one uint.
inline uint q6k_aos_u32(const global uchar * p) {
    return (uint) (*(const global ushort *) (p + 0)) | ((uint) (*(const global ushort *) (p + 2)) << 16);
}

// One AoS block (256 weights) against the lane's activation slice: four 4-value quadrants.
// qlc/qhc/isc are the lane's byte offsets in the block: ql[64*ip + l0] carries the low nibbles
// of quadrants 0 and 2, ql[64*ip + l0 + QK_K/8] those of quadrants 1 and 3, qh[32*ip + l0]
// carries the 2 high bits of all four, and the scales are signed int8 at scales[8*ip + l0/16].
// ys* is 32*sum(y) of the matching quadrant, the folded -32 of the 6-bit weights.
inline float q6k_aos_block_dot(const global uchar * blk, const int qlc, const int qhc, const int isc,
                               const float4 y0, const float4 y1, const float4 y2, const float4 y3,
                               const float ys0, const float ys1, const float ys2, const float ys3) {
    const uint m4 = 0x0F0F0F0Fu;   // four packed 4-bit fields
    const uint m2 = 0x03030303u;   // four packed qh planes
    const uint q1 = q6k_aos_u32(blk + qlc);
    const uint q2 = q6k_aos_u32(blk + qlc + QK_K/8);
    const uint h  = q6k_aos_u32(blk + Q6K_AOS_QH + qhc);

    const uint w0 = (q1 & m4) | ((h & m2) << 4);
    const uint w1 = (q2 & m4) | (((h >> 2) & m2) << 4);
    const uint w2 = ((q1 >> 4) & m4) | (((h >> 4) & m2) << 4);
    const uint w3 = ((q2 >> 4) & m4) | (((h >> 6) & m2) << 4);

    const float d0 = (float) (w0 & 0xFFu) * y0.s0 + (float) ((w0 >>  8) & 0xFFu) * y0.s1
                   + (float) ((w0 >> 16) & 0xFFu) * y0.s2 + (float)  (w0 >> 24)         * y0.s3 - ys0;
    const float d1 = (float) (w1 & 0xFFu) * y1.s0 + (float) ((w1 >>  8) & 0xFFu) * y1.s1
                   + (float) ((w1 >> 16) & 0xFFu) * y1.s2 + (float)  (w1 >> 24)         * y1.s3 - ys1;
    const float d2 = (float) (w2 & 0xFFu) * y2.s0 + (float) ((w2 >>  8) & 0xFFu) * y2.s1
                   + (float) ((w2 >> 16) & 0xFFu) * y2.s2 + (float)  (w2 >> 24)         * y2.s3 - ys2;
    const float d3 = (float) (w3 & 0xFFu) * y3.s0 + (float) ((w3 >>  8) & 0xFFu) * y3.s1
                   + (float) ((w3 >> 16) & 0xFFu) * y3.s2 + (float)  (w3 >> 24)         * y3.s3 - ys3;

    const global char * sc = (const global char *) (blk + Q6K_AOS_SCALES + isc);
    return vload_half(0, (const global half *) (blk + Q6K_AOS_D)) *
           (d0 * sc[0] + d1 * sc[2] + d2 * sc[4] + d3 * sc[6]);
}

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mv_q6_K_f32_aos(
        global uchar * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global uchar *) ((global char *) src0 + offset0);
    src1 = (global float *) ((global char *) src1 + offset1);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int nb  = ne00/QK_K;
    const int im  = get_group_id(2);            // the admitted shape has ne02 == ne12 == ne13 == 1
    const int i12 = im%ne12;
    const int i13 = im/ne12;
    const ulong plane = (ulong) ((i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * Q6K_AOS_BYTES;

    const int first_row = (Q6K_AOS_N_SIMDGROUP*get_group_id(0) + get_sub_group_id())*Q6K_AOS_N_DST;
    const int sglid = get_sub_group_local_id();
    const int tid   = sglid%16;                 // the lane's 16-value slice of a block
    const int ix    = sglid/16;                 // its block slot (stride Q6K_AOS_BLOCK_STRIDE)
    const int ip    = tid/8;
    const int l0    = 4*(tid%8);
    const int qlc   = 64*ip + l0;
    const int qhc   = 32*ip + l0;
    const int isc   = 8*ip + l0/16;

    // Rows this subgroup still owns: a weight whose ne01 is not a multiple of the 32 rows a
    // work group covers has fewer in its last group.
    int nrow = ne01 - first_row;
    if (nrow > Q6K_AOS_N_DST) {
        nrow = Q6K_AOS_N_DST;
    }
    if (nrow <= 0) {
        return;
    }

    const global float * yy = src1 + (ulong) get_group_id(1)*ne10 + (ulong) im*ne00*ne1;
    const int stride = nb*Q6K_AOS_BYTES;

    for (int rg = 0; rg < nrow; rg += 4) {
        // A row past ne01 reads the group's first row instead - in bounds, and its accumulator
        // is never stored - so the block loop below needs no bounds test.
        const int ra = first_row + rg + 0 < ne01 ? first_row + rg + 0 : first_row;
        const int rb = first_row + rg + 1 < ne01 ? first_row + rg + 1 : first_row;
        const int rc = first_row + rg + 2 < ne01 ? first_row + rg + 2 : first_row;
        const int rd = first_row + rg + 3 < ne01 ? first_row + rg + 3 : first_row;
        const global uchar * wa = src0 + (ulong) ra*stride + plane;
        const global uchar * wb = src0 + (ulong) rb*stride + plane;
        const global uchar * wc = src0 + (ulong) rc*stride + plane;
        const global uchar * wd = src0 + (ulong) rd*stride + plane;

        float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
        for (int ib = ix; ib < nb; ib += Q6K_AOS_BLOCK_STRIDE) {
            // The four rows share this slice: 16 scalar loads, no vector load (vload4 is
            // miscompiled by the target compiler).
            const global float * yv = yy + ib*QK_K + 128*ip + l0;
            const float4 y0 = (float4) (yv[ 0], yv[ 1], yv[ 2], yv[ 3]);
            const float4 y1 = (float4) (yv[32], yv[33], yv[34], yv[35]);
            const float4 y2 = (float4) (yv[64], yv[65], yv[66], yv[67]);
            const float4 y3 = (float4) (yv[96], yv[97], yv[98], yv[99]);
            const float ys0 = 32.0f*(y0.s0 + y0.s1 + y0.s2 + y0.s3);
            const float ys1 = 32.0f*(y1.s0 + y1.s1 + y1.s2 + y1.s3);
            const float ys2 = 32.0f*(y2.s0 + y2.s1 + y2.s2 + y2.s3);
            const float ys3 = 32.0f*(y3.s0 + y3.s1 + y3.s2 + y3.s3);

            const ulong boff = (ulong) ib * Q6K_AOS_BYTES;
            t0 += q6k_aos_block_dot(wa + boff, qlc, qhc, isc, y0, y1, y2, y3, ys0, ys1, ys2, ys3);
            t1 += q6k_aos_block_dot(wb + boff, qlc, qhc, isc, y0, y1, y2, y3, ys0, ys1, ys2, ys3);
            t2 += q6k_aos_block_dot(wc + boff, qlc, qhc, isc, y0, y1, y2, y3, ys0, ys1, ys2, ys3);
            t3 += q6k_aos_block_dot(wd + boff, qlc, qhc, isc, y0, y1, y2, y3, ys0, ys1, ys2, ys3);
        }

        // One reduction per row; the store is guarded, the collectives are not (every lane of
        // the subgroup has to reach them).
        const float s0 = sub_group_reduce_add(t0);
        const float s1 = sub_group_reduce_add(t1);
        const float s2 = sub_group_reduce_add(t2);
        const float s3 = sub_group_reduce_add(t3);
        if (sglid == 0) {
            const int base = get_group_id(1)*ne0 + im*ne0*ne1 + first_row + rg;
            if (first_row + rg + 0 < ne01) { dst[base + 0] = s0; }
            if (first_row + rg + 1 < ne01) { dst[base + 1] = s1; }
            if (first_row + rg + 2 < ne01) { dst[base + 2] = s2; }
            if (first_row + rg + 3 < ne01) { dst[base + 3] = s3; }
        }
    }
}
