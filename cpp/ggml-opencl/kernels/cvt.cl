//------------------------------------------------------------------------------
// This file is contains kernels for data conversion.
// These kernels are used when loading the model, so its performance is less
// important.
//------------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK1_0                   128
#define QR1_0                   1
#define QK_K                    256
#define K_SCALE_SIZE            (3 * QK_K / 64)
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

// GGML_OPENCL_CVT_VARIANT=bitselect defines GGML_OPENCL_CVT_BITSELECT.
// That arm is reconstructed from the trans4 layout contract; it is not a
// recovery of the historical experimental patch. Both arms retain byte-wise
// source reads and scalar-uint global stores.
// defect P, attempt 2: pure scalar uint pack helpers. The uchar helpers were
// removed because the Adreno compiler re-vectorizes byte-lane code (uchar
// loads/stores/arith) into short lanes with arithmetic shifts, sign-filling
// the even-partner byte's bit7 into the odd high nibble (the WRITE-side
// cross-byte smear localized by the defect-P probe — see
// scratchpad/agents/defectP-probe/REPORT.md). Doing all nibble extraction and
// word assembly in scalar uint defeats the re-vectorization; this is the same
// form that fixed the restore. Each return value is a uint holding a byte
// (0..255); the caller ORs four of them (shifted by 0/8/16/24) into one uint
// output word with a single uint store. GGML_OPENCL_CVT_BITSELECT is preserved
// as the env selector: the bitselect arm uses the bitselect builtin on uint
// (no byte lanes); the default arm uses plain uint OR. Both arms yield
// identical bytes (bitselect(a,b,c) == (a & ~c) | (b & c)).
inline uint trans4_pack_lo_u32(uint ux0, uint ux1) {
    // low nibbles: byte = lo(x0) | (lo(x1) << 4)
#ifdef GGML_OPENCL_CVT_BITSELECT
    return bitselect(ux0 & 0x0Fu, (ux1 & 0x0Fu) << 4, 0xF0u);
#else
    return (ux0 & 0x0Fu) | ((ux1 & 0x0Fu) << 4);
#endif
}

inline uint trans4_pack_hi_u32(uint ux0, uint ux1) {
    // high nibbles: byte = (hi(x0) >> 4) | hi(x1)
#ifdef GGML_OPENCL_CVT_BITSELECT
    return bitselect((ux0 >> 4) & 0x0Fu, ux1 & 0xF0u, 0xF0u);
#else
    return ((ux0 >> 4) & 0x0Fu) | (ux1 & 0xF0u);
#endif
}

//------------------------------------------------------------------------------
// block_q1_0
//------------------------------------------------------------------------------
typedef struct {
    half d;             // delta
    uchar qs[QK1_0/8];  // 1-bit signs (16 bytes)
} block_q1_0;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// block_q4_1
//------------------------------------------------------------------------------
struct block_q4_1 {
    half d; // delta
    half m; // min
    uchar qs[QK4_1 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_0
//------------------------------------------------------------------------------
struct block_q5_0 {
    half d; // delta
    uchar qh[4]; // 5-th bit of quants
    uchar qs[QK5_0 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_1
//------------------------------------------------------------------------------
struct block_q5_1 {
    half d; // delta
    half m; // min
    uchar qh[4]; // 5-th bit of quants
    uchar qs[QK5_1 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q4_k
//------------------------------------------------------------------------------
struct block_q4_K {
    half d; // delta
    half dm; // min
    uchar s[K_SCALE_SIZE];
    uchar q[QK_K / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_k
//------------------------------------------------------------------------------
struct block_q5_K {
    half d; // delta
    half dm; // min
    uchar s[K_SCALE_SIZE];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q6_K
//------------------------------------------------------------------------------
struct block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
};

//------------------------------------------------------------------------------
// block_q2_K — 84 B, 84%4==0. Host ggml-common.h:298-309.
//------------------------------------------------------------------------------
struct block_q2_K {
    uchar scales[QK_K/16]; // 16: lo nibble scale, hi nibble min
    uchar qs[QK_K/4];      // 64: 2-bit, interleaved (not 16-consecutive)
    half  d;
    half  dmin;
};

//------------------------------------------------------------------------------
// block_q3_K — 110 B, 110%4==2 (same class as block_q6_K 210 B).
// Host ggml-common.h:315-321. Pointer stride must stay 110.
//------------------------------------------------------------------------------
struct block_q3_K {
    uchar hmask[QK_K/8]; // 32
    uchar qs[QK_K/4];    // 64
    uchar scales[12];    // 16 x 6-bit packed
    half  d;
};

//------------------------------------------------------------------------------
// block_iq4_nl
//------------------------------------------------------------------------------
#define QK4_NL 32

struct block_iq4_nl
{
    half d;
    uint8_t qs[QK4_NL / 2];
};

//------------------------------------------------------------------------------
// bf16 to f16
//------------------------------------------------------------------------------
kernel void kernel_convert_bf16_to_f16(
    global const ushort * src,
    global half * dst,
    ulong off_dst,
    ulong n
) {
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    dst[i + off_dst] = (half) as_float((uint) src[i] << 16);
}

//------------------------------------------------------------------------------
// f16 to bf16
//------------------------------------------------------------------------------
kernel void kernel_convert_f16_to_bf16(
    global const half * src,
    ulong off_src,
    global ushort * dst,
    ulong n
) {
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    float f = (float) src[i + off_src];
    uint bits = as_uint(f);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        // nan to quiet nan
        dst[i] = (ushort)((bits >> 16) | 0x40u);
    } else {
        uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        dst[i] = (ushort)(rounded >> 16);
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q1_0
// Convert block_q1_0 (AOS) to 2 separate arrays (SOA): quant bytes + scales.
// q1_0 bits are stored in natural order (bit j of byte i -> weight 8*i + j)
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q1_0(
    global block_q1_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q1_0 * b = (global block_q1_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + (QK1_0/8)*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK1_0/8; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q1_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q1_0 * dst
) {
    global block_q1_0 * b = (global block_q1_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + (QK1_0/8)*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK1_0/8; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle
// Flatten q4_0 weights and unshuffle the bits
//------------------------------------------------------------------------------

kernel void kernel_convert_block_q4_0_noshuffle(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        // Workaround for adreno - must have the following printf statement for
        // the kernel to work properly. Otherwise it produces incorrect result.
        // convert_uchar above also seems necessary.
        // Compare against a large number so that it does not print anything.
        // get_sub_group_local_id() also works.
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q4_0_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = q[i + 0      ] ;
        uchar x1 = q[i + QK4_0/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q4_0_trans4_ns(
    __global struct block_q4_0 * src0,
    __global uint * dst_q,
    __global half * dst_d,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q4_0 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;

    // ------------------------------------------------------------------
    // Output layout contract (unchanged, byte for byte):
    //   word w (w = 0..3) of block (i00,i01,i02) is the uint at
    //       dst_q[i02*ne00_blk*ne01*4 + i00*ne01*4 + i01 + w*ne01]
    //   whose little-endian bytes are
    //       byte[i]   = (qs[2i]  & 0x0F) | ((qs[2i+1] & 0x0F) << 4),  i = 0..7
    //       byte[i+8] = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0),   i = 0..7
    //   i.e. low nibbles fill words 0-1, high nibbles fill words 2-3.
    // qs sits at offset 2 in block_q4_0: keep byte-wise reads (source may be
    // misaligned in the AoS block).
    // defect P, attempt 2 (scratchpad/agents/defectP-probe/REPORT.md): the
    // uchar byte-store loop was the WRITE-side defect (Adreno re-vectorizes
    // uchar-lane code into short lanes w/ arithmetic shifts; even-partner bit7
    // sign-fills the odd high nibble -> got=(want&0x0F)|0xF0). Rewritten in
    // scalar uint: per-byte qs loads widened to uint, one uint expression per
    // output word, ONE uint store (dst uint-aligned); no uchar/ushort in the
    // dataflow. Same form that fixed the restore; supersedes the old
    // "uint-in-registers miscompiled" note (the byte-store loop's fault).
    // ------------------------------------------------------------------
    uint dst_idx = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint stride  = ne01;

    for (int w = 0; w < 2; ++w) {
        uint lo_word = 0u;
        uint hi_word = 0u;
        for (int j = 0; j < 4; ++j) {
            uint ux0 = (uint) b->qs[2*(4*w + j) + 0];
            uint ux1 = (uint) b->qs[2*(4*w + j) + 1];
            lo_word |= (trans4_pack_lo_u32(ux0, ux1) << (j*8));
            hi_word |= (trans4_pack_hi_u32(ux0, ux1) << (j*8));
        }
        dst_q[dst_idx +         w * stride] = lo_word;  // word 4w+0 .. 4w+3
        dst_q[dst_idx + (2 + w) * stride] = hi_word;    // word 4w+8 .. 4w+11
    }
}

kernel void kernel_restore_block_q4_0_trans4_ns(
    __global uint * src_q,
    __global half * src_d,
    __global struct block_q4_0 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q4_0 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_d_offset];

    // Pure integer extraction avoids the scalar-private vector/byte alias
    // that Adreno A7xx miscompiles. This is the exact inverse of the producer
    // contract above: value k is word k/8, nibble k%8.
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];

    for (int i = 0; i < QK4_0 / 4; ++i) {
        uint byte_shift = 8u * (uint)(i & 3);
        uint lo_word = i < 4 ? q0 : q1;
        uint hi_word = i < 4 ? q2 : q3;
        uchar x0 = (uchar)((lo_word >> byte_shift) & 0xFFu);
        uchar x1 = (uchar)((hi_word >> byte_shift) & 0xFFu);

        b->qs[2 * i + 0] = convert_uchar((x0 & 0x0F) | ((x1 & 0x0F) << 4));
        b->qs[2 * i + 1] = convert_uchar(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
    }
    // If the device retrial fails, the documented fallback candidate is the
    // q6_K-style loop-indexed uint4[8] form, never a scalar or one-element array.
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_1
// Convert the block_q4_1 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_1(
    global struct block_q4_1 * src0,
    global uchar * dst_q,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);
    global half  * m = (global half *) dst_m + get_global_id(0);

    *d = b->d;
    *m = b->m;

    for (int i = 0; i < QK4_1/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_1(
    global uchar * src_q,
    global half  * src_d,
    global half  * src_m,
    global struct block_q4_1 * dst
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);
    global half  * m = (global half *) src_m + get_global_id(0);

    b->d = *d;
    b->m = *m;
    for (int i = 0; i < QK4_1/2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_convert_block_q4_1_noshuffle(
    global struct block_q4_1 * src0,
    global uchar * dst_q,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);
    global half  * m = (global half *) dst_m + get_global_id(0);

    *d = b->d;
    *m = b->m;
    for (int i = 0; i < QK4_1/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_1/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q4_1_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global half  * src_m,
    global struct block_q4_1 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);
    global half  * m = (global half *) src_m + get_global_id(0);

    b->d = *d;
    b->m = *m;
    for (int i = 0; i < QK4_1/4; ++i) {
        uchar x0 = q[i + 0      ] ;
        uchar x1 = q[i + QK4_1/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q4_1_trans4_ns(
    __global struct block_q4_1 * src0,
    __global uint * dst_q,
    __global half * dst_d,
    __global half * dst_m,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_1;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q4_1 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;
    dst_m[dst_blk_offset] = b->m;

    // ------------------------------------------------------------------
    // Output layout contract (unchanged, byte for byte):
    //   word w (w = 0..3) of block (i00,i01,i02) is the uint at
    //       dst_q[i02*ne00_blk*ne01*4 + i00*ne01*4 + i01 + w*ne01]
    //   whose little-endian bytes are
    //       byte[i]   = (qs[2i]  & 0x0F) | ((qs[2i+1] & 0x0F) << 4),  i = 0..7
    //       byte[i+8] = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0),   i = 0..7
    //   i.e. low nibbles fill words 0-1, high nibbles fill words 2-3.
    // qs sits at offset 4 in block_q4_1: keep byte-wise reads (source may be
    // misaligned in the AoS block).
    // defect P, attempt 2 (scratchpad/agents/defectP-probe/REPORT.md): the
    // uchar byte-store loop was the WRITE-side defect (Adreno re-vectorizes
    // uchar-lane code into short lanes w/ arithmetic shifts; even-partner bit7
    // sign-fills the odd high nibble -> got=(want&0x0F)|0xF0). Rewritten in
    // scalar uint: per-byte qs loads widened to uint, one uint expression per
    // output word, ONE uint store (dst uint-aligned); no uchar/ushort in the
    // dataflow. Same form that fixed the restore; supersedes the old
    // "uint-in-registers miscompiled" note (the byte-store loop's fault).
    // ------------------------------------------------------------------
    uint dst_idx = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint stride  = ne01;

    for (int w = 0; w < 2; ++w) {
        uint lo_word = 0u;
        uint hi_word = 0u;
        for (int j = 0; j < 4; ++j) {
            uint ux0 = (uint) b->qs[2*(4*w + j) + 0];
            uint ux1 = (uint) b->qs[2*(4*w + j) + 1];
            lo_word |= (trans4_pack_lo_u32(ux0, ux1) << (j*8));
            hi_word |= (trans4_pack_hi_u32(ux0, ux1) << (j*8));
        }
        dst_q[dst_idx +         w * stride] = lo_word;  // word 4w+0 .. 4w+3
        dst_q[dst_idx + (2 + w) * stride] = hi_word;    // word 4w+8 .. 4w+11
    }
}

kernel void kernel_restore_block_q4_1_trans4_ns(
    __global uint * src_q,
    __global half * src_d,
    __global half * src_m,
    __global struct block_q4_1 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_1;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_dm_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q4_1 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_dm_offset];
    b->m = src_m[src_dm_offset];

    // Restore qs with shifts and masks only; no private alias or array.
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];

    for (int i = 0; i < QK4_1 / 4; ++i) {
        uint byte_shift = 8u * (uint)(i & 3);
        uint lo_word = i < 4 ? q0 : q1;
        uint hi_word = i < 4 ? q2 : q3;
        uchar x0 = (uchar)((lo_word >> byte_shift) & 0xFFu);
        uchar x1 = (uchar)((hi_word >> byte_shift) & 0xFFu);

        b->qs[2 * i + 0] = convert_uchar((x0 & 0x0F) | ((x1 & 0x0F) << 4));
        b->qs[2 * i + 1] = convert_uchar(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_0
// Convert the block_q5_0 format to 3 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_0(
    global struct block_q5_0 * src0,
    global uchar * dst_qs,
    global uint  * dst_qh,
    global half  * dst_d,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }

    global struct block_q5_0 * b = (global struct block_q5_0 *) src0 + get_global_id(0);
    global uchar * qs = (global uchar *) dst_qs + (QK5_0/2)*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_0/2; ++i) {
        qs[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q5_0(
    global uchar * src_qs,
    global uint  * src_qh,
    global half  * src_d,
    global struct block_q5_0 * dst
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) dst + get_global_id(0);
    global uchar * qs = (global uchar *) src_qs + (QK5_0/2)*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;
    *((global uint *)(b->qh)) = *qh;
    for (int i = 0; i < QK5_0/2; ++i) {
        b->qs[i] = qs[i];
    }
}

kernel void kernel_convert_block_q5_0_noshuffle(
    global struct block_q5_0 * src0,
    global uchar * dst_q,
    global uint  * dst_qh,
    global half  * dst_d
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q + QK5_0/2*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK5_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q5_0_noshuffle(
    global uchar * src_q,
    global uint  * src_qh,
    global half  * src_d,
    global struct block_q5_0 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q + QK5_0/2*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;
    *((global uint *)(b->qh)) = *qh;

    for (int i = 0; i < QK5_0/4; ++i) {
        uchar x0 = q[i + 0      ];
        uchar x1 = q[i + QK5_0/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q5_0_trans4_ns(
    __global struct block_q5_0 * src0,
    __global uint * dst_qs,
    __global uint * dst_qh,
    __global half * dst_d,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_0;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q5_0 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;

    // ------------------------------------------------------------------
    // Output layout contract (unchanged, byte for byte):
    //   word w (w = 0..3) of block (i00,i01,i02) is the uint at
    //       dst_qs[i02*ne00_blk*ne01*4 + i00*ne01*4 + i01 + w*ne01]
    //   whose little-endian bytes are
    //       byte[i]   = (qs[2i]  & 0x0F) | ((qs[2i+1] & 0x0F) << 4),  i = 0..7
    //       byte[i+8] = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0),   i = 0..7
    //   i.e. low nibbles fill words 0-1, high nibbles fill words 2-3.
    //   qh: the little-endian uint at dst_qh[dst_blk_offset] holds
    //       b->qh[0..3] verbatim (byte copy, no shifts).
    // qs sits at offset 6, qh at offset 2 in block_q5_0: keep byte-wise reads
    // (source may be misaligned in the AoS block). qh is a plain byte copy
    // (no shifts; defect-P probe confirmed qh clean) — left as-is.
    // defect P, attempt 2 (scratchpad/agents/defectP-probe/REPORT.md): the
    // uchar byte-store loop was the WRITE-side defect (Adreno re-vectorizes
    // uchar-lane code into short lanes w/ arithmetic shifts; even-partner bit7
    // sign-fills the odd high nibble -> got=(want&0x0F)|0xF0). Rewritten in
    // scalar uint: per-byte qs loads widened to uint, one uint expression per
    // output word, ONE uint store (dst uint-aligned); no uchar/ushort in the
    // dataflow. Same form that fixed the restore; supersedes the old
    // "uint-in-registers miscompiled" note (the byte-store loop's fault).
    // ------------------------------------------------------------------
    {
        global uchar * qh = (global uchar *) dst_qh + dst_blk_offset * 4;
        qh[0] = b->qh[0];
        qh[1] = b->qh[1];
        qh[2] = b->qh[2];
        qh[3] = b->qh[3];
    }

    uint dst_idx = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint stride  = ne01;

    for (int w = 0; w < 2; ++w) {
        uint lo_word = 0u;
        uint hi_word = 0u;
        for (int j = 0; j < 4; ++j) {
            uint ux0 = (uint) b->qs[2*(4*w + j) + 0];
            uint ux1 = (uint) b->qs[2*(4*w + j) + 1];
            lo_word |= (trans4_pack_lo_u32(ux0, ux1) << (j*8));
            hi_word |= (trans4_pack_hi_u32(ux0, ux1) << (j*8));
        }
        dst_qs[dst_idx +         w * stride] = lo_word;  // word 4w+0 .. 4w+3
        dst_qs[dst_idx + (2 + w) * stride] = hi_word;    // word 4w+8 .. 4w+11
    }
}

kernel void kernel_restore_block_q5_0_trans4_ns(
    __global uint * src_qs,
    __global uint * src_qh,
    __global half * src_d,
    __global struct block_q5_0 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_0;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q5_0 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_blk_offset];

    {
        uint qh = src_qh[src_blk_offset];
        b->qh[0] = (uchar)(qh);
        b->qh[1] = (uchar)(qh >> 8);
        b->qh[2] = (uchar)(qh >> 16);
        b->qh[3] = (uchar)(qh >> 24);
    }

    // Restore qs with shifts and masks only; qh remains byte-wise above.
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_qs[src_q_offset];
    uint q1 = src_qs[src_q_offset + ne01];
    uint q2 = src_qs[src_q_offset + ne01 * 2];
    uint q3 = src_qs[src_q_offset + ne01 * 3];

    for (int i = 0; i < QK5_0 / 4; ++i) {
        uint byte_shift = 8u * (uint)(i & 3);
        uint lo_word = i < 4 ? q0 : q1;
        uint hi_word = i < 4 ? q2 : q3;
        uchar x0 = (uchar)((lo_word >> byte_shift) & 0xFFu);
        uchar x1 = (uchar)((hi_word >> byte_shift) & 0xFFu);

        b->qs[2 * i + 0] = convert_uchar((x0 & 0x0F) | ((x1 & 0x0F) << 4));
        b->qs[2 * i + 1] = convert_uchar(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_1
// Convert the block_q5_1 format to 4 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_1(
    global struct block_q5_1 * src0,
    global uchar * dst_qs,
    global uint  * dst_qh,
    global half  * dst_d,
    global half  * dst_m,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }

    global struct block_q5_1 * b = (global struct block_q5_1 *) src0 + get_global_id(0);
    global uchar * qs = (global uchar *) dst_qs + (QK5_1/2)*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * m  = (global half  *) dst_m  + get_global_id(0);

    *d = b->d;
    *m = b->m;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_1/2; ++i) {
        qs[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q5_1(
    global uchar * src_qs,
    global uint  * src_qh,
    global half  * src_d,
    global half  * src_m,
    global struct block_q5_1 * dst
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) dst + get_global_id(0);
    global uchar * qs = (global uchar *) src_qs + (QK5_1/2)*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * m  = (global half  *) src_m  + get_global_id(0);

    b->d = *d;
    b->m = *m;
    *((global uint *)(b->qh)) = *qh;
    for (int i = 0; i < QK5_1/2; ++i) {
        b->qs[i] = qs[i];
    }
}

kernel void kernel_convert_block_q5_1_noshuffle(
    global struct block_q5_1 * src0,
    global uchar * dst_q,
    global uint  * dst_qh,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q + QK5_1/2*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * m  = (global half  *) dst_m  + get_global_id(0);

    *d = b->d;
    *m = b->m;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_1/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK5_1/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q5_1_noshuffle(
    global uchar * src_q,
    global uint  * src_qh,
    global half  * src_d,
    global half  * src_m,
    global struct block_q5_1 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q + QK5_1/2*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * m  = (global half  *) src_m  + get_global_id(0);

    b->d = *d;
    b->m = *m;
    *((global uint *)(b->qh)) = *qh;

    for (int i = 0; i < QK5_1/4; ++i) {
        uchar x0 = q[i + 0      ];
        uchar x1 = q[i + QK5_1/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q5_1_trans4_ns(
    __global struct block_q5_1 * src0,
    __global uint * dst_qs,
    __global uint * dst_qh,
    __global half * dst_d,
    __global half * dst_m,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_1;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q5_1 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;
    dst_m[dst_blk_offset] = b->m;

    // ------------------------------------------------------------------
    // Output layout contract (unchanged, byte for byte):
    //   word w (w = 0..3) of block (i00,i01,i02) is the uint at
    //       dst_qs[i02*ne00_blk*ne01*4 + i00*ne01*4 + i01 + w*ne01]
    //   whose little-endian bytes are
    //       byte[i]   = (qs[2i]  & 0x0F) | ((qs[2i+1] & 0x0F) << 4),  i = 0..7
    //       byte[i+8] = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0),   i = 0..7
    //   i.e. low nibbles fill words 0-1, high nibbles fill words 2-3.
    //   qh: the little-endian uint at dst_qh[dst_blk_offset] holds
    //       b->qh[0..3] verbatim (byte copy, no shifts).
    // qs sits at offset 8, qh at offset 4 in block_q5_1: keep byte-wise reads
    // (source may be misaligned in the AoS block). qh is a plain byte copy
    // (no shifts; defect-P probe confirmed qh clean) — left as-is.
    // defect P, attempt 2 (scratchpad/agents/defectP-probe/REPORT.md): the
    // uchar byte-store loop was the WRITE-side defect (Adreno re-vectorizes
    // uchar-lane code into short lanes w/ arithmetic shifts; even-partner bit7
    // sign-fills the odd high nibble -> got=(want&0x0F)|0xF0). Rewritten in
    // scalar uint: per-byte qs loads widened to uint, one uint expression per
    // output word, ONE uint store (dst uint-aligned); no uchar/ushort in the
    // dataflow. Same form that fixed the restore; supersedes the old
    // "uint-in-registers miscompiled" note (the byte-store loop's fault).
    // ------------------------------------------------------------------
    {
        global uchar * qh = (global uchar *) dst_qh + dst_blk_offset * 4;
        qh[0] = b->qh[0];
        qh[1] = b->qh[1];
        qh[2] = b->qh[2];
        qh[3] = b->qh[3];
    }

    uint dst_idx = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint stride  = ne01;

    for (int w = 0; w < 2; ++w) {
        uint lo_word = 0u;
        uint hi_word = 0u;
        for (int j = 0; j < 4; ++j) {
            uint ux0 = (uint) b->qs[2*(4*w + j) + 0];
            uint ux1 = (uint) b->qs[2*(4*w + j) + 1];
            lo_word |= (trans4_pack_lo_u32(ux0, ux1) << (j*8));
            hi_word |= (trans4_pack_hi_u32(ux0, ux1) << (j*8));
        }
        dst_qs[dst_idx +         w * stride] = lo_word;  // word 4w+0 .. 4w+3
        dst_qs[dst_idx + (2 + w) * stride] = hi_word;    // word 4w+8 .. 4w+11
    }
}

kernel void kernel_restore_block_q5_1_trans4_ns(
    __global uint * src_qs,
    __global uint * src_qh,
    __global half * src_d,
    __global half * src_m,
    __global struct block_q5_1 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_1;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q5_1 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_blk_offset];
    b->m = src_m[src_blk_offset];

    {
        uint qh = src_qh[src_blk_offset];
        b->qh[0] = (uchar)(qh);
        b->qh[1] = (uchar)(qh >> 8);
        b->qh[2] = (uchar)(qh >> 16);
        b->qh[3] = (uchar)(qh >> 24);
    }

    // Restore qs with shifts and masks only; qh remains byte-wise above.
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_qs[src_q_offset];
    uint q1 = src_qs[src_q_offset + ne01];
    uint q2 = src_qs[src_q_offset + ne01 * 2];
    uint q3 = src_qs[src_q_offset + ne01 * 3];

    for (int i = 0; i < QK5_1 / 4; ++i) {
        uint byte_shift = 8u * (uint)(i & 3);
        uint lo_word = i < 4 ? q0 : q1;
        uint hi_word = i < 4 ? q2 : q3;
        uchar x0 = (uchar)((lo_word >> byte_shift) & 0xFFu);
        uchar x1 = (uchar)((hi_word >> byte_shift) & 0xFFu);

        b->qs[2 * i + 0] = convert_uchar((x0 & 0x0F) | ((x1 & 0x0F) << 4));
        b->qs[2 * i + 1] = convert_uchar(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
    }
}

kernel void kernel_convert_block_q4_k_trans4_ns(
    __global struct block_q4_K * src0,
    __global uint  * dst_q,
    __global half  * dst_d,
    __global half  * dst_dm,
    __global uchar * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q4_K * b = src0 + src_blk_offset;

    dst_d [dst_blk_offset] = b->d;
    dst_dm[dst_blk_offset] = b->dm;

    uint4 qv[8];
    uchar * qv_bytes = (uchar *)qv;
    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->q[i*32 + 2*j];
            uchar x1 = b->q[i*32 + 2*j + 1];

            qv_bytes[i*32 + j     ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            qv_bytes[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
        }
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint4 v = qv[p];
        dst_q[base + (p * 4 + 0) * ne01] = v.x;
        dst_q[base + (p * 4 + 1) * ne01] = v.y;
        dst_q[base + (p * 4 + 2) * ne01] = v.z;
        dst_q[base + (p * 4 + 3) * ne01] = v.w;
    }

    __global uchar * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    #pragma unroll
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s_dst[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q4_k_trans4_ns(
    __global uint  * src_q,
    __global half  * src_d,
    __global half  * src_dm,
    __global uchar * src_s,
    __global struct block_q4_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q4_K * b = dst0 + dst_blk_offset;

    b->d  = src_d[src_blk_offset];
    b->dm = src_dm[src_blk_offset];

    __global uchar * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s_src[i];
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;

    uint4 qv[8];
    for (int p = 0; p < 8; ++p) {
        qv[p].x = src_q[base + (p * 4 + 0) * ne01];
        qv[p].y = src_q[base + (p * 4 + 1) * ne01];
        qv[p].z = src_q[base + (p * 4 + 2) * ne01];
        qv[p].w = src_q[base + (p * 4 + 3) * ne01];
    }

    uchar * qv_bytes = (uchar *)qv;
    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = qv_bytes[i*32 + j];
            uchar hi = qv_bytes[i*32 + j + 16];
            b->q[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->q[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }
}

// ============================================================================
// q2_K / q3_K trans4-moe convert + restore.
// Masks are kernel args. Convert stores assembled uint words into SoA.
// Restore: q2 may uint-store (block 84 B, 4-aligned); q3 must convert_uchar
// per byte (block 110 B, 110%4==2, odd block index +2).
// ============================================================================

kernel void kernel_convert_block_q2_k_trans4_ns(
    __global struct block_q2_K * src0,
    __global uint  * dst_q,
    __global half  * dst_d,
    __global half  * dst_dm,
    __global uint  * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_03
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint m03 = (uint)mask_03;
    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q2_K * b = src0 + src_blk_offset;

    dst_d [dst_blk_offset] = b->d;
    dst_dm[dst_blk_offset] = b->dmin;

    uint s_base = ((i02 * ne01 + i01) * ne00_blk + i00) * 4;
    for (uint t = 0; t < 4; ++t) {
        uint w = 0;
        w |= ((uint)b->scales[4*t + 0]) <<  0;
        w |= ((uint)b->scales[4*t + 1]) <<  8;
        w |= ((uint)b->scales[4*t + 2]) << 16;
        w |= ((uint)b->scales[4*t + 3]) << 24;
        dst_s[s_base + t] = w;
    }

    uint word[16];
    for (uint g = 0; g < 16; ++g) {
        uint hpart = g >> 3;
        uint g8    = g & 7;
        uint j     = g8 >> 1;
        uint hi    = g8 & 1;
        uint shift = j << 1;
        uint byte_base = hpart * 32 + hi * 16;
        uint u = 0;
        for (uint k = 0; k < 16; ++k) {
            uint q = ((uint)b->qs[byte_base + k] >> shift) & m03;
            u |= q << (2 * k);
        }
        word[g] = u;
    }

    uint base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    for (uint p = 0; p < 16; ++p) {
        dst_q[base + p * ne01] = word[p];
    }
}

kernel void kernel_restore_block_q2_k_trans4_ns(
    __global uint  * src_q,
    __global half  * src_d,
    __global half  * src_dm,
    __global uint  * src_s,
    __global struct block_q2_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_03
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint m03 = (uint)mask_03;
    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q2_K * b = dst0 + dst_blk_offset;

    b->d    = src_d [src_blk_offset];
    b->dmin = src_dm[src_blk_offset];

    uint s_base = ((i02 * ne01 + i01) * ne00_blk + i00) * 4;
    __global uint * scales_u = (__global uint *)b->scales;
    for (uint t = 0; t < 4; ++t) {
        scales_u[t] = src_s[s_base + t];
    }

    uint base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    uint word[16];
    for (uint p = 0; p < 16; ++p) {
        word[p] = src_q[base + p * ne01];
    }

    __global uint * qs_u = (__global uint *)b->qs;
    for (uint t = 0; t < 16; ++t) {
        uint hpart = t >> 3;
        uint hi    = (t >> 2) & 1;
        uint l0    = (t & 3) << 2;
        uint w = 0;
        for (uint r = 0; r < 4; ++r) {
            uint l = l0 + r;
            uint byte = 0;
            for (uint j = 0; j < 4; ++j) {
                uint g = hpart * 8 + 2 * j + hi;
                uint q = (word[g] >> (2 * l)) & m03;
                byte |= q << (2 * j);
            }
            w |= byte << (8 * r);
        }
        qs_u[t] = w;
    }
}

kernel void kernel_convert_block_q3_k_trans4_ns(
    __global struct block_q3_K * src0,
    __global uint  * dst_q,
    __global uint  * dst_qh,
    __global half  * dst_d,
    __global uint  * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_03,
    uchar mask_0F,
    uchar mask_01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint m03 = (uint)mask_03;
    uint m0F = (uint)mask_0F;
    uint m01 = (uint)mask_01;
    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q3_K * b = src0 + src_blk_offset;

    dst_d[dst_blk_offset] = b->d;

    uint bsc[12];
    for (uint j = 0; j < 12; ++j) {
        bsc[j] = (uint)b->scales[j];
    }
    uint S[16];
    for (uint j = 0; j < 16; ++j) {
        uint lo4 = (j < 8) ? ((bsc[j] >> 0) & m0F)
                           : ((bsc[j - 8] >> 4) & m0F);
        uint hi2 = (bsc[8 + (j & 3)] >> (2 * (j >> 2))) & m03;
        S[j] = lo4 | (hi2 << 4);
    }
    uint s_base = ((i02 * ne01 + i01) * ne00_blk + i00) * 4;
    for (uint t = 0; t < 4; ++t) {
        uint w = 0;
        w |= S[4*t + 0] <<  0;
        w |= S[4*t + 1] <<  8;
        w |= S[4*t + 2] << 16;
        w |= S[4*t + 3] << 24;
        dst_s[s_base + t] = w;
    }

    uint word[16];
    for (uint g = 0; g < 16; ++g) {
        uint hpart = g >> 3;
        uint g8    = g & 7;
        uint j     = g8 >> 1;
        uint hi    = g8 & 1;
        uint shift = j << 1;
        uint byte_base = hpart * 32 + hi * 16;
        uint u = 0;
        for (uint k = 0; k < 16; ++k) {
            uint q = ((uint)b->qs[byte_base + k] >> shift) & m03;
            u |= q << (2 * k);
        }
        word[g] = u;
    }
    uint q_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    for (uint p = 0; p < 16; ++p) {
        dst_q[q_base + p * ne01] = word[p];
    }

    uint qh_word[8];
    for (uint p = 0; p < 8; ++p) {
        uint bit = p;
        uint u = 0;
        for (uint k = 0; k < 16; ++k) {
            uint lo = ((uint)b->hmask[k]      >> bit) & m01;
            uint hi = ((uint)b->hmask[16 + k] >> bit) & m01;
            u |= lo << k;
            u |= hi << (16 + k);
        }
        qh_word[p] = u;
    }
    uint h_base = i02 * ne00_blk * ne01 * 8 + i00 * ne01 * 8 + i01;
    for (uint p = 0; p < 8; ++p) {
        dst_qh[h_base + p * ne01] = qh_word[p];
    }
}

kernel void kernel_restore_block_q3_k_trans4_ns(
    __global uint  * src_q,
    __global uint  * src_qh,
    __global half  * src_d,
    __global uint  * src_s,
    __global struct block_q3_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_03,
    uchar mask_0F,
    uchar mask_01,
    uchar mask_FF
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint m03 = (uint)mask_03;
    uint m0F = (uint)mask_0F;
    uint m01 = (uint)mask_01;
    uint mFF = (uint)mask_FF;
    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q3_K * b = dst0 + dst_blk_offset;

    b->d = src_d[src_blk_offset];

    uint s_base = ((i02 * ne01 + i01) * ne00_blk + i00) * 4;
    uint S[16];
    for (uint t = 0; t < 4; ++t) {
        uint w = src_s[s_base + t];
        S[4*t + 0] = (w >>  0) & mFF;
        S[4*t + 1] = (w >>  8) & mFF;
        S[4*t + 2] = (w >> 16) & mFF;
        S[4*t + 3] = (w >> 24) & mFF;
    }
    uint s_w[3];
    s_w[0] = 0;
    s_w[1] = 0;
    s_w[2] = 0;
    for (uint j = 0; j < 16; ++j) {
        uint l = S[j];
        if (j < 8) {
            s_w[j >> 2] |= (l & m0F) << (8 * (j & 3));
        } else {
            s_w[(j - 8) >> 2] |= ((l & m0F) << 4) << (8 * ((j - 8) & 3));
        }
        l >>= 4;
        s_w[2] |= (l << (2 * (j >> 2))) << (8 * (j & 3));
    }
    for (uint t = 0; t < 3; ++t) {
        for (uint r = 0; r < 4; ++r) {
            b->scales[4*t + r] = convert_uchar((s_w[t] >> (8 * r)) & mFF);
        }
    }

    uint q_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    uint word[16];
    for (uint p = 0; p < 16; ++p) {
        word[p] = src_q[q_base + p * ne01];
    }
    for (uint t = 0; t < 16; ++t) {
        uint hpart = t >> 3;
        uint hi    = (t >> 2) & 1;
        uint l0    = (t & 3) << 2;
        uint w = 0;
        for (uint r = 0; r < 4; ++r) {
            uint l = l0 + r;
            uint byte = 0;
            for (uint j = 0; j < 4; ++j) {
                uint g = hpart * 8 + 2 * j + hi;
                uint q = (word[g] >> (2 * l)) & m03;
                byte |= q << (2 * j);
            }
            w |= byte << (8 * r);
        }
        for (uint r = 0; r < 4; ++r) {
            b->qs[4*t + r] = convert_uchar((w >> (8 * r)) & mFF);
        }
    }

    uint h_base = i02 * ne00_blk * ne01 * 8 + i00 * ne01 * 8 + i01;
    uint qh_word[8];
    for (uint p = 0; p < 8; ++p) {
        qh_word[p] = src_qh[h_base + p * ne01];
    }
    for (uint t = 0; t < 8; ++t) {
        uint w = 0;
        for (uint r = 0; r < 4; ++r) {
            uint bidx = 4 * t + r;
            uint hi   = bidx >> 4;
            uint k    = bidx & 15;
            uint byte = 0;
            for (uint p = 0; p < 8; ++p) {
                uint hm = (qh_word[p] >> ((hi << 4) + k)) & m01;
                byte |= hm << p;
            }
            w |= byte << (8 * r);
        }
        for (uint r = 0; r < 4; ++r) {
            b->hmask[4*t + r] = convert_uchar((w >> (8 * r)) & mFF);
        }
    }
}

// Dequant dump for one super-block. Unpack MUST match the GEMV consumer.
// Launch {16,1,1}/{16,1,1}. ORACLE=GEMV unpack
kernel void kernel_dequant_row_q2_k_trans4_ns(
    __global uint  * src0_q,
    __global half  * src0_d,
    __global half  * src0_dm,
    __global uchar * src0_s,
    int              ne00,
    int              ne01,
    int              i01,
    int              expert_id,
    int              sb,
    __global float * out256,
    uchar            mask_0F,
    uchar            mask_03
) {
    uint k = get_local_id(0);
    if (i01 >= ne01) {
        return;
    }
    uint n_sb = (uint)ne00 / QK_K;
    if ((uint)sb >= n_sb || k >= 16) {
        return;
    }
    uint m0F = (uint)mask_0F;
    uint m03 = (uint)mask_03;
    uint expert_d_offset = (uint)expert_id * n_sb * (uint)ne01;
    uint expert_q_offset = (uint)expert_id * ((uint)ne00 / 16) * (uint)ne01;
    uint s_row = ((uint)expert_id * (uint)ne01 + (uint)i01) * (n_sb * 16);

    float d    = convert_float(src0_d [expert_d_offset + (uint)sb * (uint)ne01 + (uint)i01]);
    float dmin = convert_float(src0_dm[expert_d_offset + (uint)sb * (uint)ne01 + (uint)i01]);

    for (uint g = 0; g < 16; ++g) {
        uint sc = (uint)src0_s[s_row + (uint)sb * 16 + g];
        float dl = d    * convert_float(sc & m0F);
        float ml = dmin * convert_float((sc >> 4) & m0F);
        uint bits = src0_q[expert_q_offset + ((uint)sb * 16 + g) * (uint)ne01 + (uint)i01];
        uint q = (bits >> (2 * k)) & m03;
        out256[16 * g + k] = dl * convert_float(q) - ml;
    }
}

// ORACLE=GEMV unpack
kernel void kernel_dequant_row_q3_k_trans4_ns(
    __global uint  * src0_q,
    __global half  * src0_d,
    __global uchar * src0_s,
    __global uint  * src0_qh,
    int              ne00,
    int              ne01,
    int              i01,
    int              expert_id,
    int              sb,
    __global float * out256,
    uchar            mask_03,
    uchar            mask_01,
    float            scale_zero
) {
    uint k = get_local_id(0);
    if (i01 >= ne01) {
        return;
    }
    uint n_sb = (uint)ne00 / QK_K;
    if ((uint)sb >= n_sb || k >= 16) {
        return;
    }
    uint m03 = (uint)mask_03;
    uint m01 = (uint)mask_01;
    uint expert_d_offset  = (uint)expert_id * n_sb * (uint)ne01;
    uint expert_q_offset  = (uint)expert_id * ((uint)ne00 / 16) * (uint)ne01;
    uint expert_qh_offset = (uint)expert_id * ((uint)ne00 / 32) * (uint)ne01;
    uint s_row = ((uint)expert_id * (uint)ne01 + (uint)i01) * (n_sb * 16);

    float d = convert_float(src0_d[expert_d_offset + (uint)sb * (uint)ne01 + (uint)i01]);

    for (uint g = 0; g < 16; ++g) {
        uint sc = (uint)src0_s[s_row + (uint)sb * 16 + g];
        float dl = d * (convert_float(sc) - scale_zero);
        uint bits = src0_q[expert_q_offset + ((uint)sb * 16 + g) * (uint)ne01 + (uint)i01];
        uint p    = g >> 1;
        uint hi   = g & 1;
        uint lane = hi << 4;
        uint qh_w = src0_qh[expert_qh_offset + ((uint)sb * 8 + p) * (uint)ne01 + (uint)i01];
        uint q_u  = (bits >> (2 * k)) & m03;
        uint hm   = (qh_w >> (lane + k)) & m01;
        int qhat  = (int)q_u;
        if (hm == 0) {
            qhat = (int)q_u - 4;
        }
        out256[16 * g + k] = dl * convert_float(qhat);
    }
}

kernel void kernel_convert_block_q5_k_trans4_ns(
    __global struct block_q5_K * src0,
    __global uint  * dst_qs,
    __global uint  * dst_qh,
    __global half  * dst_d,
    __global half  * dst_dm,
    __global uchar * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q5_K * b = src0 + src_blk_offset;

    dst_d [dst_blk_offset] = b->d;
    dst_dm[dst_blk_offset] = b->dm;

    for (int k = 0; k < 8; k++) {
        uchar b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (int bit = 0; bit < 8; bit++) {
            b0 |= (uchar)(((b->qh[bit]      >> k) & 1) << bit);
            b1 |= (uchar)(((b->qh[8  + bit] >> k) & 1) << bit);
            b2 |= (uchar)(((b->qh[16 + bit] >> k) & 1) << bit);
            b3 |= (uchar)(((b->qh[24 + bit] >> k) & 1) << bit);
        }
        uint packed = (uint)b0 | ((uint)b1 << 8) | ((uint)b2 << 16) | ((uint)b3 << 24);
        dst_qh[i01 + (i00 * 8 + k) * ne01 + i02 * ne00_blk * 8 * ne01] = packed;
    }

    uint4 qv[8];
    uchar * qv_bytes = (uchar *)qv;
    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->qs[i*32 + 2*j];
            uchar x1 = b->qs[i*32 + 2*j + 1];

            qv_bytes[i*32 + j     ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            qv_bytes[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
        }
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint4 v = qv[p];
        dst_qs[base + (p * 4 + 0) * ne01] = v.x;
        dst_qs[base + (p * 4 + 1) * ne01] = v.y;
        dst_qs[base + (p * 4 + 2) * ne01] = v.z;
        dst_qs[base + (p * 4 + 3) * ne01] = v.w;
    }

    __global uchar * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    #pragma unroll
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s_dst[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q5_k_trans4_ns(
    __global uint  * src_qs,
    __global uint  * src_qh,
    __global half  * src_d,
    __global half  * src_dm,
    __global uchar * src_s,
    __global struct block_q5_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q5_K * b = dst0 + dst_blk_offset;

    b->d  = src_d[src_blk_offset];
    b->dm = src_dm[src_blk_offset];

    for (int j = 0; j < 32; j++) b->qh[j] = 0;
    for (int k = 0; k < 8; k++) {
        uint packed = src_qh[i01 + (i00 * 8 + k) * ne01 + i02 * ne00_blk * 8 * ne01];
        uchar b0 = (uchar)(packed & 0xFF);
        uchar b1 = (uchar)((packed >> 8) & 0xFF);
        uchar b2 = (uchar)((packed >> 16) & 0xFF);
        uchar b3 = (uchar)((packed >> 24) & 0xFF);
        for (int bit = 0; bit < 8; bit++) {
            b->qh[bit]      |= (uchar)(((b0 >> bit) & 1) << k);
            b->qh[8  + bit] |= (uchar)(((b1 >> bit) & 1) << k);
            b->qh[16 + bit] |= (uchar)(((b2 >> bit) & 1) << k);
            b->qh[24 + bit] |= (uchar)(((b3 >> bit) & 1) << k);
        }
    }

    __global uchar * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s_src[i];
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;

    uint4 qv[8];
    for (int p = 0; p < 8; ++p) {
        qv[p].x = src_qs[base + (p * 4 + 0) * ne01];
        qv[p].y = src_qs[base + (p * 4 + 1) * ne01];
        qv[p].z = src_qs[base + (p * 4 + 2) * ne01];
        qv[p].w = src_qs[base + (p * 4 + 3) * ne01];
    }

    uchar * qv_bytes = (uchar *)qv;
    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = qv_bytes[i*32 + j];
            uchar hi = qv_bytes[i*32 + j + 16];
            b->qs[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->qs[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }
}

kernel void kernel_convert_block_q6_k_trans4_ns(
    __global struct block_q6_K * src0,
    __global uint  * dst_ql,
    __global uint  * dst_qh,
    __global half  * dst_d,
    __global char  * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q6_K * b = src0 + src_blk_offset;

    dst_d[dst_blk_offset] = b->d;

    uint4 qlv[8];
    uchar * qlv_bytes = (uchar *)qlv;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->ql[i*64 + 2*j];
            uchar x1 = b->ql[i*64 + 2*j + 1];
            uchar x2 = b->ql[i*64 + 32 + 2*j];
            uchar x3 = b->ql[i*64 + 32 + 2*j + 1];
            qlv_bytes[i*64 + j     ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            qlv_bytes[i*64 + j + 16] = convert_uchar(x2 & mask_0F) | convert_uchar((x3 & mask_0F) << 4);
            qlv_bytes[i*64 + j + 32] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
            qlv_bytes[i*64 + j + 48] = convert_uchar((x2 & mask_F0) >> 4) | convert_uchar(x3 & mask_F0);
        }
    }

    uint ql_base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint4 v = qlv[p];
        dst_ql[ql_base + (p * 4 + 0) * ne01] = v.x;
        dst_ql[ql_base + (p * 4 + 1) * ne01] = v.y;
        dst_ql[ql_base + (p * 4 + 2) * ne01] = v.z;
        dst_ql[ql_base + (p * 4 + 3) * ne01] = v.w;
    }

    uint qhv[16] = {0};

    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            uchar h = b->qh[n*32 + l];
            int u = l / 16;
            int bit_pos = (l % 16) * 2;
            qhv[(n*4 + 0)*2 + u] |= ((uint)((h >> 0) & 0x03)) << bit_pos;
            qhv[(n*4 + 1)*2 + u] |= ((uint)((h >> 2) & 0x03)) << bit_pos;
            qhv[(n*4 + 2)*2 + u] |= ((uint)((h >> 4) & 0x03)) << bit_pos;
            qhv[(n*4 + 3)*2 + u] |= ((uint)((h >> 6) & 0x03)) << bit_pos;
        }
    }

    uint qh_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;

    for (int p = 0; p < 16; ++p) {
        dst_qh[qh_base + p * ne01] = qhv[p];
    }

    __global char * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * 16 + i00 * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s_dst[i] = b->scales[i];
    }
}

kernel void kernel_restore_block_q6_k_trans4_ns(
    __global uint  * src_ql,
    __global uint  * src_qh,
    __global half  * src_d,
    __global char  * src_s,
    __global struct block_q6_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q6_K * b = dst0 + dst_blk_offset;

    b->d = src_d[src_blk_offset];

    uint ql_base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
    uint4 qlv[8];
    for (int p = 0; p < 8; ++p) {
        qlv[p].x = src_ql[ql_base + (p * 4 + 0) * ne01];
        qlv[p].y = src_ql[ql_base + (p * 4 + 1) * ne01];
        qlv[p].z = src_ql[ql_base + (p * 4 + 2) * ne01];
        qlv[p].w = src_ql[ql_base + (p * 4 + 3) * ne01];
    }

    uchar * qlv_bytes = (uchar *)qlv;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo_02 = qlv_bytes[i*64 + j];
            uchar lo_13 = qlv_bytes[i*64 + j + 16];
            uchar hi_02 = qlv_bytes[i*64 + j + 32];
            uchar hi_13 = qlv_bytes[i*64 + j + 48];
            b->ql[i*64 + 2*j]          = convert_uchar((lo_02 & mask_0F) | ((hi_02 & mask_0F) << 4));
            b->ql[i*64 + 2*j + 1]      = convert_uchar(((lo_02 & mask_F0) >> 4) | (hi_02 & mask_F0));
            b->ql[i*64 + 32 + 2*j]     = convert_uchar((lo_13 & mask_0F) | ((hi_13 & mask_0F) << 4));
            b->ql[i*64 + 32 + 2*j + 1] = convert_uchar(((lo_13 & mask_F0) >> 4) | (hi_13 & mask_F0));
        }
    }

    uint qh_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    uint qhv[16];
    for (int p = 0; p < 16; ++p) {
        qhv[p] = src_qh[qh_base + p * ne01];
    }

    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            int u = l / 16;
            int bit_pos = (l % 16) * 2;
            uchar v0 = (uchar)((qhv[(n*4 + 0)*2 + u] >> bit_pos) & 0x03);
            uchar v1 = (uchar)((qhv[(n*4 + 1)*2 + u] >> bit_pos) & 0x03);
            uchar v2 = (uchar)((qhv[(n*4 + 2)*2 + u] >> bit_pos) & 0x03);
            uchar v3 = (uchar)((qhv[(n*4 + 3)*2 + u] >> bit_pos) & 0x03);
            b->qh[n*32 + l] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
        }
    }

    __global char * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * 16 + i00 * 16;
    for (int i = 0; i < 16; ++i) {
        b->scales[i] = s_src[i];
    }
}

//------------------------------------------------------------------------------
// block_mxfp4
//------------------------------------------------------------------------------
#define QK_MXFP4 32
struct block_mxfp4 {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_mxfp4
// Convert the block_mxfp4 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_mxfp4(
    global struct block_mxfp4 * src0,
    global uchar * dst_q,
    global uchar * dst_e
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) dst_e + get_global_id(0);

    *e = b->e;

    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_convert_block_mxfp4_trans(
    global struct block_mxfp4 * src0,
    __global uint4 * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;

    dst_q[dst_blk_offset] = ((global uint4 *)(&(b->qs[0])))[0];
    dst_e[dst_blk_offset] = b->e;
}

kernel void kernel_restore_block_mxfp4(
    global uchar * src_q,
    global half  * src_e,
    global struct block_mxfp4 * dst
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) src_e + get_global_id(0);

    b->e = *e;
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_restore_block_mxfp4_trans(
    __global uint4 * src_q,
    __global uchar * src_e,
    global struct block_mxfp4 * dst,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = dst + dst_blk_offset;

    ((global uint4 *)(&(b->qs[0])))[0] = src_q[src_blk_offset];
    b->e = src_e[src_blk_offset];
}

kernel void kernel_convert_block_mxfp4_trans4_ns(
    global struct block_mxfp4 * src0,
    __global uint * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;
    dst_e[dst_blk_offset] = b->e;

    // ------------------------------------------------------------------
    // Output layout contract (unchanged, byte for byte):
    //   word w (w = 0..3) of block (i00,i01,i02) is the uint at
    //       dst_q[i02*ne00_blk*ne01*4 + i00*ne01*4 + i01 + w*ne01]
    //   whose little-endian bytes are
    //       byte[i]   = (qs[2i]  & 0x0F) | ((qs[2i+1] & 0x0F) << 4),  i = 0..7
    //       byte[i+8] = ((qs[2i] & 0xF0) >> 4) | (qs[2i+1] & 0xF0),   i = 0..7
    //   i.e. low nibbles fill words 0-1, high nibbles fill words 2-3.
    //   (mxfp4 keeps the same interleave as q4_0: the consumers of the
    //   repacked buffer were validated against exactly this byte order.)
    // qs sits at offset 1 in block_mxfp4: keep byte-wise reads (source may be
    // misaligned in the AoS block).
    // defect P, attempt 2 (scratchpad/agents/defectP-probe/REPORT.md): the
    // uchar byte-store loop was the WRITE-side defect (Adreno re-vectorizes
    // uchar-lane code into short lanes w/ arithmetic shifts; even-partner bit7
    // sign-fills the odd high nibble -> got=(want&0x0F)|0xF0). Rewritten in
    // scalar uint: per-byte qs loads widened to uint, one uint expression per
    // output word, ONE uint store (dst uint-aligned); no uchar/ushort in the
    // dataflow. Same form that fixed the restore; supersedes the old
    // "uint-in-registers miscompiled" note (the byte-store loop's fault).
    // ------------------------------------------------------------------
    uint dst_idx = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint stride  = ne01;

    for (int w = 0; w < 2; ++w) {
        uint lo_word = 0u;
        uint hi_word = 0u;
        for (int j = 0; j < 4; ++j) {
            uint ux0 = (uint) b->qs[2*(4*w + j) + 0];
            uint ux1 = (uint) b->qs[2*(4*w + j) + 1];
            lo_word |= (trans4_pack_lo_u32(ux0, ux1) << (j*8));
            hi_word |= (trans4_pack_hi_u32(ux0, ux1) << (j*8));
        }
        dst_q[dst_idx +         w * stride] = lo_word;  // word 4w+0 .. 4w+3
        dst_q[dst_idx + (2 + w) * stride] = hi_word;    // word 4w+8 .. 4w+11
    }
}

kernel void kernel_restore_block_mxfp4_trans4_ns(
    __global uint * src_q,
    __global uchar * src_e,
    __global struct block_mxfp4 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_mxfp4 * b = dst0 + dst_blk_offset;
    b->e = src_e[src_d_offset];

    // Restore qs with shifts and masks only; exponent load remains unchanged.
    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];

    for (int i = 0; i < QK_MXFP4 / 4; ++i) {
        uint byte_shift = 8u * (uint)(i & 3);
        uint lo_word = i < 4 ? q0 : q1;
        uint hi_word = i < 4 ? q2 : q3;
        uchar x0 = (uchar)((lo_word >> byte_shift) & 0xFFu);
        uchar x1 = (uchar)((hi_word >> byte_shift) & 0xFFu);

        b->qs[2 * i + 0] = convert_uchar((x0 & 0x0F) | ((x1 & 0x0F) << 4));
        b->qs[2 * i + 1] = convert_uchar(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
    }
}


//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

kernel void kernel_convert_block_q8_0(
    global block_q8_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q8_0 * b = (global block_q8_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK8_0; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q8_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst
) {
    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK8_0; ++i) {
        b->qs[i] = q[i];
    }
}

// View-aware AoS q8_0 -> f32 dequant (f32/f32 FA path).
kernel void kernel_dequant_q8_0_f32_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global float * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK8_0);
    float d = vload_half(0, (global half *)block);
    global char * qs = block + 2;

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global float * out = dst + (dst_row_base + blk_i0) * QK8_0;

    for (int i = 0; i < QK8_0; ++i) {
        out[i] = d * (float)qs[i];
    }
}

// View-aware AoS q8_0 -> f16 dequant. Rows tight, batch strides may be gapped.
kernel void kernel_dequant_q8_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK8_0);
    float d = vload_half(0, (global half *)block);
    global char * qs = block + 2;

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK8_0;

    for (int i = 0; i < QK8_0; ++i) {
        out[i] = (half)(d * (float)qs[i]);
    }
}

// View-aware AoS q4_0 -> f32 dequant (mirrors the q8_0 view variant).
kernel void kernel_dequant_q4_0_f32_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global float * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK4_0/2);
    float d = vload_half(0, (global half *)block);
    global uchar * qs = (global uchar *)(block + 2);

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global float * out = dst + (dst_row_base + blk_i0) * QK4_0;

    for (int i = 0; i < QK4_0/2; ++i) {
        uchar byte = qs[i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4)   - 8;
        out[i]            = d * (float)q0;
        out[i + QK4_0/2]  = d * (float)q1;
    }
}

// View-aware AoS q4_0 -> f16 dequant (mirrors the q8_0 view variant).
kernel void kernel_dequant_q4_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK4_0/2);
    float d = vload_half(0, (global half *)block);
    global uchar * qs = (global uchar *)(block + 2);

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK4_0;

    for (int i = 0; i < QK4_0/2; ++i) {
        uchar byte = qs[i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4)   - 8;
        out[i]          = (half)(d * (float)q0);
        out[i + QK4_0/2] = (half)(d * (float)q1);
    }
}

kernel void kernel_restore_block_q8_0_trans(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst,
    uint ne00,
    uint ne01
){
    uint num_blk_per_row = ne00 / QK8_0;

    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0) * num_blk_per_row;
    global uchar      * q = (global uchar *) src_q + get_global_id(0) * 4; // 4 8-bit packed
    global half       * d = (global half *) src_d + get_global_id(0);

    for (uint blk = 0; blk < num_blk_per_row; blk++) {
        b->d = *d;

        for (uint i = 0; i < QK8_0; i+=4) {
            b->qs[i]   = q[0];
            b->qs[i+1] = q[1];
            b->qs[i+2] = q[2];
            b->qs[i+3] = q[3];

            q += 4 * ne01; // M stride
        }

        d += ne01;

        b++;
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_K
// Convert the block_q4_K format to 4 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
// Each thread processes a super block.
// Mask args are just to keep the signature consistent with the no-shuffle
// version and they are not used in this kernel.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_K(
    global struct block_q4_K * src0,
    global uchar * dst_q,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2*get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K/2; ++i) {
        q[i] = b->q[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

// Restore block_q4_K from flattened arrays.
// Each thread processes a super block.
// Mask args are just to keep the signature consistent with the no-shuffle ones.
kernel void kernel_restore_block_q4_K(
    global uchar * src_q,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q4_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2*get_global_id(0);
    global uchar * s  = (global uchar *) src_s + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm  + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K/2; ++i) {
        b->q[i] = q[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

kernel void kernel_convert_block_q4_K_noshuffle(
    global struct block_q4_K * src0,
    global uchar * dst_q,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2 * get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->q[i*32 + 2*j];
            uchar x1 = b->q[i*32 + 2*j + 1];
            q[i*32 + j]      = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            q[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4)   | convert_uchar(x1 & mask_F0);
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q4_K_noshuffle(
    global uchar * src_q,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q4_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2 * get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = q[i*32 + j];
            uchar hi = q[i*32 + j + 16];
            b->q[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->q[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_K
// Convert the block_q5_K format to 5 separate arrays (AOS -> SOA).
// Each thread processes a super block.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_K(
    global struct block_q5_K * src0,
    global uchar * dst_q,
    global uchar * dst_qh,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/8*get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K/2; ++i) {
        q[i] = b->qs[i];
    }
    for (int i = 0; i < QK_K/8; ++i) {
        qh[i] = b->qh[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

// Restore block_q5_K from flattened arrays.
// Each thread processes a super block.
kernel void kernel_restore_block_q5_K(
    global uchar * src_q,
    global uchar * src_qh,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q5_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/8*get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d    = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K/2; ++i) {
        b->qs[i] = q[i];
    }
    for (int i = 0; i < QK_K/8; ++i) {
        b->qh[i] = qh[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

kernel void kernel_convert_block_q5_K_noshuffle(
    global struct block_q5_K * src0,
    global uchar * dst_q,
    global uchar * dst_qh,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2       * get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/8       * get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->qs[i*32 + 2*j];
            uchar x1 = b->qs[i*32 + 2*j + 1];
            q[i*32 + j]      = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            q[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
        }
    }

    for (int l = 0; l < QK_K/8; ++l) {
        uchar x0 = 0;
        for (int i = 0; i < 8; ++i) {
            x0 |= ((b->qh[(l%4)*8+i] >> (l/4)) & 0x01) << i;
        }
        qh[l] = x0;
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q5_K_noshuffle(
    global uchar * src_q,
    global uchar * src_qh,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q5_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2       * get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/8       * get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = q[i*32 + j];
            uchar hi = q[i*32 + j + 16];
            b->qs[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->qs[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }

    for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < 8; ++i) {
            uchar x0 = 0;
            for (int k = 0; k < 8; ++k) {
                x0 |= ((qh[4*k+g] >> i) & 0x01) << k;
            }
            b->qh[g*8+i] = x0;
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q6_K
// Convert the block_q6_K format to 3 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
// Each thread processes a super block.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q6_K(
    global struct block_q6_K * src0,
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) src0 + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK_K/2; ++i) {
        ql[i] = b->ql[i];
    }
    for (int i = 0; i < QK_K/4; ++i) {
        qh[i] = b->qh[i];
    }
    for (int i = 0; i < QK_K/16; ++i) {
        s[i] = b->scales[i];
    }
}

// Restore block_q6_K from flattened arrays.
// Each thread processes a super block.
kernel void kernel_restore_block_q6_K(
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    global struct block_q6_K * dst,
    uchar mask_lsb_8,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) dst + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK_K/2; ++i) {
        b->ql[i] = ql[i];
    }
    for (int i = 0; i < QK_K/4; ++i) {
        b->qh[i] = qh[i];
    }
    for (int i = 0; i < QK_K/16; ++i) {
        b->scales[i] = s[i];
    }
}

kernel void kernel_convert_block_q6_K_noshuffle(
    global struct block_q6_K * src0,
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) src0 + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK_K/2/4; ++i) {
        uchar x0 = b->ql[i*2 + 0] & mask_lsb_8;
        uchar x1 = b->ql[i*2 + 1] & mask_lsb_8;
        ql[i +  0] = (x0 & 0x0F)        | ((x1 & 0x0F) << 4);
        ql[i + 32] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);

        uchar x2 = b->ql[i*2 + 0 + 64] & mask_lsb_8;
        uchar x3 = b->ql[i*2 + 1 + 64] & mask_lsb_8;
        ql[i + 64] = (x2 & 0x0F)        | ((x3 & 0x0F) << 4);
        ql[i + 96] = ((x2 & 0xF0) >> 4) | (x3 & 0xF0);
    }

    for (int i = 0; i < QK_K/4/8; ++i) {
        uchar x0 = b->qh[i*4 + 0] & mask_lsb_8;
        uchar x1 = b->qh[i*4 + 1] & mask_lsb_8;
        uchar x2 = b->qh[i*4 + 2] & mask_lsb_8;
        uchar x3 = b->qh[i*4 + 3] & mask_lsb_8;
        qh[i +  0] = (x0 & 0x03)        | ((x1 & 0x03) << 2) | ((x2 & 0x03) << 4) | ((x3 & 0x03) << 6);
        qh[i +  8] = ((x0 & 0x0C) >> 2) | (x1 & 0x0C)        | ((x2 & 0x0C) << 2) | ((x3 & 0x0C) << 4);
        qh[i + 16] = ((x0 & 0x30) >> 4) | ((x1 & 0x30) >> 2) | (x2 & 0x30)        | ((x3 & 0x30) << 2);
        qh[i + 24] = ((x0 & 0xC0) >> 6) | ((x1 & 0xC0) >> 4) | ((x2 & 0xC0) >> 2) | (x3 & 0xC0);

        uchar x4 = b->qh[i*4 + 0 + 32] & mask_lsb_8;
        uchar x5 = b->qh[i*4 + 1 + 32] & mask_lsb_8;
        uchar x6 = b->qh[i*4 + 2 + 32] & mask_lsb_8;
        uchar x7 = b->qh[i*4 + 3 + 32] & mask_lsb_8;
        qh[i + 32] = (x4 & 0x03)        | ((x5 & 0x03) << 2) | ((x6 & 0x03) << 4) | ((x7 & 0x03) << 6);
        qh[i + 40] = ((x4 & 0x0C) >> 2) | (x5 & 0x0C)        | ((x6 & 0x0C) << 2) | ((x7 & 0x0C) << 4);
        qh[i + 48] = ((x4 & 0x30) >> 4) | ((x5 & 0x30) >> 2) | (x6 & 0x30)        | ((x7 & 0x30) << 2);
        qh[i + 56] = ((x4 & 0xC0) >> 6) | ((x5 & 0xC0) >> 4) | ((x6 & 0xC0) >> 2) | (x7 & 0xC0);
    }

    for (int i = 0; i < QK_K/16; ++i) {
        s[i] = b->scales[i];
    }
}

kernel void kernel_restore_block_q6_K_noshuffle(
    global uchar * src_ql,
    global uchar * src_qh,
    global char  * src_s,
    global half  * src_d,
    global struct block_q6_K * dst,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) dst + get_global_id(0);
    global uchar * ql = (global uchar *) src_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) src_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK_K/2/4; ++i) {
        uchar x0   = ql[i +  0] & mask_lsb_8;
        uchar x1   = ql[i + 32] & mask_lsb_8;
        b->ql[i*2 + 0] = (x0 & 0x0F)        | ((x1 & 0x0F) << 4);
        b->ql[i*2 + 1] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);

        uchar x2   = ql[i + 64] & mask_lsb_8;
        uchar x3   = ql[i + 96] & mask_lsb_8;
        b->ql[i*2 + 0 + 64] = (x2 & 0x0F)        | ((x3 & 0x0F) << 4);
        b->ql[i*2 + 1 + 64] = ((x2 & 0xF0) >> 4) | (x3 & 0xF0);
    }

    for (int i = 0; i < QK_K/4/8; ++i) {
        uchar x0 = qh[i +  0] & mask_lsb_8;
        uchar x1 = qh[i +  8] & mask_lsb_8;
        uchar x2 = qh[i + 16] & mask_lsb_8;
        uchar x3 = qh[i + 24] & mask_lsb_8;
        b->qh[i*4 + 0] = (x0 & 0x03)        | ((x1 & 0x03) << 2) | ((x2 & 0x03) << 4) | ((x3 & 0x03) << 6);
        b->qh[i*4 + 1] = ((x0 & 0x0C) >> 2) | (x1 & 0x0C)        | ((x2 & 0x0C) << 2) | ((x3 & 0x0C) << 4);
        b->qh[i*4 + 2] = ((x0 & 0x30) >> 4) | ((x1 & 0x30) >> 2) | (x2 & 0x30)        | ((x3 & 0x30) << 2);
        b->qh[i*4 + 3] = ((x0 & 0xC0) >> 6) | ((x1 & 0xC0) >> 4) | ((x2 & 0xC0) >> 2) | (x3 & 0xC0);

        uchar x4 = qh[i +  0 + 32] & mask_lsb_8;
        uchar x5 = qh[i +  8 + 32] & mask_lsb_8;
        uchar x6 = qh[i + 16 + 32] & mask_lsb_8;
        uchar x7 = qh[i + 24 + 32] & mask_lsb_8;
        b->qh[i*4 + 0 + 32] = (x4 & 0x03)        | ((x5 & 0x03) << 2) | ((x6 & 0x03) << 4) | ((x7 & 0x03) << 6);
        b->qh[i*4 + 1 + 32] = ((x4 & 0x0C) >> 2) | (x5 & 0x0C)        | ((x6 & 0x0C) << 2) | ((x7 & 0x0C) << 4);
        b->qh[i*4 + 2 + 32] = ((x4 & 0x30) >> 4) | ((x5 & 0x30) >> 2) | (x6 & 0x30)        | ((x7 & 0x30) << 2);
        b->qh[i*4 + 3 + 32] = ((x4 & 0xC0) >> 6) | ((x5 & 0xC0) >> 4) | ((x6 & 0xC0) >> 2) | (x7 & 0xC0);
    }

    for (int i = 0; i < QK_K/16; ++i) {
        b->scales[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_iq4_nl
// Convert the block_iq4_nl format to 2 separate arrays (AOS -> SOA).
//------------------------------------------------------------------------------
kernel void kernel_convert_block_iq4_nl(
    global struct block_iq4_nl * src0,
    global uchar * dst_q,
    global half  * dst_d,
    uchar          mask_0F,
    uchar          mask_F0,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_NL/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_iq4_nl(
    global uchar * src_q,
    global half  * src_d,
    global struct block_iq4_nl * dst,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK4_NL/2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_convert_block_iq4_nl_noshuffle(
    global struct block_iq4_nl * src0,
    global uchar * dst_q,
    global half  * dst_d,
    uchar          mask_0F,
    uchar          mask_F0,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_NL/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0       ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
        q[i + QK4_NL/4] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
    }
}

kernel void kernel_restore_block_iq4_nl_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global struct block_iq4_nl * dst,
    uchar mask_0F,
    uchar mask_F0,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_NL/4; ++i) {
        uchar x0 = q[i + 0       ];
        uchar x1 = q[i + QK4_NL/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}
