#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_64
#endif

// y = W x for one column of x (batch 1 decode GEMV), W in the NPU's native
// tiled Q4_0 layout — tiledv4 = tiledv3i (same image views, lane mapping and
// reduce) with the per-weight ALU cut: the fp16 scales are converted natively
// by the texture path instead of by bit math, the nibbles stay raw with the -8
// offset folded once per block, and each row scale is applied by one mad.

// Op count per lane per K-tile (32 useful MACs = 16 rows x 1 K pair), counted
// as scalar ops — Adreno 740 executes ~0.7 Tops/s of these (implied by the
// tiledv3i column), so the next measurement can be checked against the model:
//   tiledv3i: 4 image reads (quants uint4, x float4, 2x scale uint4);
//     unpack+dequant 112 (per uint: lo 4 and + 4 cvt + 4 sub, hi + 4 shr);
//     32 mads; scales 16 x ~14 bit-ops = 224 plus 2 branches each; scale apply
//     16 mul + 16 add; 16 d resets  ->  ~416 ops
//   tiledv4:  6 image reads (quants uint4, x float4, 4x scale float4 — the 32
//     scale bytes are L1-served, shared by the 16 lanes of one row half);
//     unpack 80 (raw nibbles: lo 4 and + 4 cvt, hi + 4 shr); 32 mads; corr 2;
//     16 subs; 16 mads; 16 d resets  ->  ~162 ops, the f16->f32 converts
//     happen in the texture hardware, off the ALU
//   Model for ffn_gate_up (M*K/32 = 688k lane-tiles): v3i 286M ops ~ 0.41 ms
//   (measured 0.414), v4 111M ops ~ 0.16 ms; the two extra scale reads could
//   push toward 0.2 ms if the image path saturates first. attn_k (8
//   work-groups) stays launch-bound and should not move.

// Scales: read through a CL_RGBA/CL_HALF_FLOAT view of the same tiled region
// (1 px = 4 fp16), so read_imagef returns them already converted — the
// texture-side f16->f32 conversion, no fp16 type in the kernel. A half pointer
// argument was tried first and cannot ship: Apple's OpenCL pipeline rejects
// kernels with fp16 content outright (createKernel: newComputePipelineState
// failed; same class as the as_half2 reject documented in tiled_v2, and
// AppleCL does not even list cl_khr_fp16). On Adreno the native conversion is
// expected to flush denormal scales to zero like the engine's ALU-side half
// conversion — the engine's own soav arm FAILs the probe's scales_denorm shape
// for exactly that reason — but the texture unit may behave differently, so
// v4's scales_denorm row is checked against whichever reference the device's
// conversion matches (flush-to-zero or exact), see shapes.c.

inline float4 q4t_lo(uint w) {
    return convert_float4(as_uchar4(w) & (uchar4)(0x0F));
}

inline float4 q4t_hi(uint w) {
    // OpenCL C never splats a scalar shift amount, and the Adreno front end
    // rejects vector-shift-by-int outright; the shift must be vector-by-vector.
    return convert_float4((as_uchar4(w) >> (uchar4)(4)) & (uchar4)(0x0F));
}

REQD_SUBGROUP_64
kernel void kernel_mul_mat_q4_0_f32_tiled_v4(
        read_only image1d_buffer_t wq,    // quants, RGBA/UINT32: 1 px = 16 B
        uint poff,                        // pixel offset of the tiled region in wq
        read_only image1d_buffer_t ximg,  // activations, RGBA/FLOAT: 1 px = 4 floats
        read_only image1d_buffer_t ws,    // scales, RGBA/HALF_FLOAT: 1 px = 4 fp16
        global float * dst,
        int K,   // multiple of 32 (the packed layout's precondition)
        int M)
{
    const int nkt = K / 32;             // K-tiles per row tile
    const int j   = get_local_id(0);
    const int p   = j / 32;             // even/odd K-tile slot
    const int l   = j % 32;             // lane within the K-tile
    const int cp  = l / 2;              // my K pair within a K-tile
    const int h16 = (l & 1) * 16;       // first of my 16 rows within the tile
    const int rt  = get_group_id(0);    // one row tile per work-group

    __local float lred[64 * 16];   // one 16-row partial per lane

    float4 t0 = (float4)(0), t1 = (float4)(0), t2 = (float4)(0), t3 = (float4)(0);
    float4 d0 = (float4)(0), d1 = (float4)(0), d2 = (float4)(0), d3 = (float4)(0);
    // both views cover the same tiled region: wq counts 16 B pixels, ws 8 B,
    // so the region's offset in ws is 2*poff
    const int spoff = (int) poff * 2;
    for (int kt = p; kt < nkt; kt += 2) {
        const int tp = (int) poff + (rt * nkt + kt) * 36;   // 576 B tile = 36 px
        const uint4 wv = read_imageui(wq, tp + l);
        const float4 xv = read_imagef(ximg, kt * 8 + (cp >> 1));
        const float2 xy = (cp & 1) ? xv.zw : xv.xy;
        // raw nibbles 0..15: sum(q x) over the lane's K pair 2*cp; the -8
        // offset is folded once per block below (sum(q x) - 8 sum(x))
        d0 = mad(q4t_lo(wv.s0), (float4)(xy.x), d0);
        d0 = mad(q4t_hi(wv.s0), (float4)(xy.y), d0);
        d1 = mad(q4t_lo(wv.s1), (float4)(xy.x), d1);
        d1 = mad(q4t_hi(wv.s1), (float4)(xy.y), d1);
        d2 = mad(q4t_lo(wv.s2), (float4)(xy.x), d2);
        d2 = mad(q4t_hi(wv.s2), (float4)(xy.y), d2);
        d3 = mad(q4t_lo(wv.s3), (float4)(xy.x), d3);
        d3 = mad(q4t_hi(wv.s3), (float4)(xy.y), d3);
        // the 16 fp16 scales of rows h16..h16+15 sit at tile bytes 512 + 2*h16
        // = 4 px from px 64 + h16/4 of the tile (a 576 B tile = 72 ws px)
        const int sp = spoff + (rt * nkt + kt) * 72 + 64 + (h16 >> 2);
        const float corr = 8.0f * (xy.x + xy.y);
        t0 = mad(d0 - corr, read_imagef(ws, sp + 0), t0);
        t1 = mad(d1 - corr, read_imagef(ws, sp + 1), t1);
        t2 = mad(d2 - corr, read_imagef(ws, sp + 2), t2);
        t3 = mad(d3 - corr, read_imagef(ws, sp + 3), t3);
        d0 = (float4)(0); d1 = (float4)(0); d2 = (float4)(0); d3 = (float4)(0);
    }

    // Same reduce and guarded stores as v2/v3i: every lane stores its 16 row
    // partials, lanes 0 and 1 sum the 16 K pairs x 2 slots of their row half;
    // padding rows carry quant 8 (real value 0) and scale 0.
    vstore4(t0, j * 4 + 0, lred);
    vstore4(t1, j * 4 + 1, lred);
    vstore4(t2, j * 4 + 2, lred);
    vstore4(t3, j * 4 + 3, lred);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (j < 2) {
        t0 = (float4)(0); t1 = (float4)(0); t2 = (float4)(0); t3 = (float4)(0);
        for (int c = 0; c < 16; c++) {
            const int w0 = (2 * c + j) * 4;
            const int w1 = (2 * c + j + 32) * 4;
            t0 += vload4(w0 + 0, lred) + vload4(w1 + 0, lred);
            t1 += vload4(w0 + 1, lred) + vload4(w1 + 1, lred);
            t2 += vload4(w0 + 2, lred) + vload4(w1 + 2, lred);
            t3 += vload4(w0 + 3, lred) + vload4(w1 + 3, lred);
        }
    }
    if (j < 2) {
        global float * out = dst + rt * 32 + 16 * j;
        if (rt * 32 + 16 * j + 0 < M) { out[0] = t0.x; }
        if (rt * 32 + 16 * j + 1 < M) { out[1] = t0.y; }
        if (rt * 32 + 16 * j + 2 < M) { out[2] = t0.z; }
        if (rt * 32 + 16 * j + 3 < M) { out[3] = t0.w; }
        if (rt * 32 + 16 * j + 4 < M) { out[4] = t1.x; }
        if (rt * 32 + 16 * j + 5 < M) { out[5] = t1.y; }
        if (rt * 32 + 16 * j + 6 < M) { out[6] = t1.z; }
        if (rt * 32 + 16 * j + 7 < M) { out[7] = t1.w; }
        if (rt * 32 + 16 * j + 8 < M) { out[8] = t2.x; }
        if (rt * 32 + 16 * j + 9 < M) { out[9] = t2.y; }
        if (rt * 32 + 16 * j + 10 < M) { out[10] = t2.z; }
        if (rt * 32 + 16 * j + 11 < M) { out[11] = t2.w; }
        if (rt * 32 + 16 * j + 12 < M) { out[12] = t3.x; }
        if (rt * 32 + 16 * j + 13 < M) { out[13] = t3.y; }
        if (rt * 32 + 16 * j + 14 < M) { out[14] = t3.z; }
        if (rt * 32 + 16 * j + 15 < M) { out[15] = t3.w; }
    }
}
