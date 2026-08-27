#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_extra_vector_types : enable

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32
#define QK_K 256

#define dotx8_reduce4(a_reg, b_lm, c_reg, lm_offset) \
    acc8.s0 = dot(a_reg.s0123, b_lm[lm_offset + 0]); \
    acc8.s1 = dot(a_reg.s0123, b_lm[lm_offset + 1]); \
    acc8.s2 = dot(a_reg.s0123, b_lm[lm_offset + 2]); \
    acc8.s3 = dot(a_reg.s0123, b_lm[lm_offset + 3]); \
    acc8.s4 = dot(a_reg.s0123, b_lm[lm_offset + 4]); \
    acc8.s5 = dot(a_reg.s0123, b_lm[lm_offset + 5]); \
    acc8.s6 = dot(a_reg.s0123, b_lm[lm_offset + 6]); \
    acc8.s7 = dot(a_reg.s0123, b_lm[lm_offset + 7]); \
    acc8.s0 += dot(a_reg.s4567, b_lm[lm_offset + 32]); \
    acc8.s1 += dot(a_reg.s4567, b_lm[lm_offset + 33]); \
    acc8.s2 += dot(a_reg.s4567, b_lm[lm_offset + 34]); \
    acc8.s3 += dot(a_reg.s4567, b_lm[lm_offset + 35]); \
    acc8.s4 += dot(a_reg.s4567, b_lm[lm_offset + 36]); \
    acc8.s5 += dot(a_reg.s4567, b_lm[lm_offset + 37]); \
    acc8.s6 += dot(a_reg.s4567, b_lm[lm_offset + 38]); \
    acc8.s7 += dot(a_reg.s4567, b_lm[lm_offset + 39]); \
    c_reg += convert_float8(acc8); \
    acc8.s0 = dot(a_reg.s89ab, b_lm[lm_offset + 64]); \
    acc8.s1 = dot(a_reg.s89ab, b_lm[lm_offset + 65]); \
    acc8.s2 = dot(a_reg.s89ab, b_lm[lm_offset + 66]); \
    acc8.s3 = dot(a_reg.s89ab, b_lm[lm_offset + 67]); \
    acc8.s4 = dot(a_reg.s89ab, b_lm[lm_offset + 68]); \
    acc8.s5 = dot(a_reg.s89ab, b_lm[lm_offset + 69]); \
    acc8.s6 = dot(a_reg.s89ab, b_lm[lm_offset + 70]); \
    acc8.s7 = dot(a_reg.s89ab, b_lm[lm_offset + 71]); \
    acc8.s0 += dot(a_reg.scdef, b_lm[lm_offset + 96]); \
    acc8.s1 += dot(a_reg.scdef, b_lm[lm_offset + 97]); \
    acc8.s2 += dot(a_reg.scdef, b_lm[lm_offset + 98]); \
    acc8.s3 += dot(a_reg.scdef, b_lm[lm_offset + 99]); \
    acc8.s4 += dot(a_reg.scdef, b_lm[lm_offset + 100]); \
    acc8.s5 += dot(a_reg.scdef, b_lm[lm_offset + 101]); \
    acc8.s6 += dot(a_reg.scdef, b_lm[lm_offset + 102]); \
    acc8.s7 += dot(a_reg.scdef, b_lm[lm_offset + 103]); \
    c_reg += convert_float8(acc8);

// ORACLE=GEMV unpack — same extracts as kernel_dequant_row_q2_k_trans4_ns
static inline half16 dequant16_q2(uint bits, uint sc, float d, float dmin, uint m0F, uint m03) {
    float dl = d    * convert_float(sc & m0F);
    float ml = dmin * convert_float((sc >> 4) & m0F);
    half16 a;
    a.s0 = (half)(dl * convert_float((bits >>  0) & m03) - ml);
    a.s1 = (half)(dl * convert_float((bits >>  2) & m03) - ml);
    a.s2 = (half)(dl * convert_float((bits >>  4) & m03) - ml);
    a.s3 = (half)(dl * convert_float((bits >>  6) & m03) - ml);
    a.s4 = (half)(dl * convert_float((bits >>  8) & m03) - ml);
    a.s5 = (half)(dl * convert_float((bits >> 10) & m03) - ml);
    a.s6 = (half)(dl * convert_float((bits >> 12) & m03) - ml);
    a.s7 = (half)(dl * convert_float((bits >> 14) & m03) - ml);
    a.s8 = (half)(dl * convert_float((bits >> 16) & m03) - ml);
    a.s9 = (half)(dl * convert_float((bits >> 18) & m03) - ml);
    a.sa = (half)(dl * convert_float((bits >> 20) & m03) - ml);
    a.sb = (half)(dl * convert_float((bits >> 22) & m03) - ml);
    a.sc = (half)(dl * convert_float((bits >> 24) & m03) - ml);
    a.sd = (half)(dl * convert_float((bits >> 26) & m03) - ml);
    a.se = (half)(dl * convert_float((bits >> 28) & m03) - ml);
    a.sf = (half)(dl * convert_float((bits >> 30) & m03) - ml);
    return a;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q2_k_f32_ns(
        __global     uint *           src0_q,
        __global     half *           src0_d,
        __global     half *           src0_dm,
        __global     uchar *          src0_s,
        __read_only  image1d_buffer_t src1,
        __global     uint *           src2,
        __global     ushort *         src2_emap,
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        uint is_ragged,
        uint skip_gran,
        uchar mask_0F,
        uchar mask_03
) {
    uint m0F = (uint)mask_0F;
    uint m03 = (uint)mask_03;
    uint block_id_m = get_global_id(1);
    uint block_id_n = get_global_id(2);

    if (block_id_n >= (uint)total_tiles[0]) {
        return;
    }

    uint n_active = TILESIZE_N;
    if (is_ragged && skip_gran < TILESIZE_N) {
        uint n_valid = TILESIZE_N;
        for (uint _t = 0; _t < TILESIZE_N; ++_t) {
            if (src2[block_id_n * TILESIZE_N + _t] == 0xFFFFFFFFu) { n_valid = _t; break; }
        }
        n_active = min((uint)TILESIZE_N, ((n_valid + skip_gran - 1) / skip_gran) * skip_gran);
    }
    bool skip_g1 = (8u  >= n_active);
    bool skip_g2 = (16u >= n_active);
    bool skip_g3 = (24u >= n_active);

    __private half16 reg_a;
    __private float32 reg_c = (float32)(0);
    __local half4 shared_b[128];

    const ushort expert_id = src2_emap[block_id_n];
    const uint row = block_id_m * TILESIZE_M;
    const uint col = block_id_n * TILESIZE_N;
    uint sub_block_id_m = get_local_id(0);
    uint2 b_global_offset;
    b_global_offset.x = ((sub_block_id_m & 3) << 2) + (sub_block_id_m >> 2) * ne00;
    b_global_offset.y = b_global_offset.x + (16 * ne00);
    uint2 b_local_offset;
    b_local_offset.x = (sub_block_id_m & 3) * 32 + (sub_block_id_m >> 2);
    b_local_offset.y = b_local_offset.x + 16;

    uint num_superblocks = ne00 / QK_K;
    uint row_idx = row + get_global_id(0);
    // unlike the q4/q5/q6 clones, no early return: all lanes must reach the
    // barriers (gate admits ne01%32, grid pads to 64)
    const bool valid = (row_idx < ne01);
    const uint row_load = valid ? row_idx : (ne01 - 1u);
    uint s_row = ((uint)expert_id * ne01 + row_load) * (num_superblocks * 16);
    uint expert_d_offset = (uint)expert_id * num_superblocks * ne01;
    uint expert_q_offset = (uint)expert_id * (ne00 / 16) * ne01;

    for (uint sb = 0; sb < num_superblocks; ++sb) {
        __private float d    = convert_float(src0_d [expert_d_offset + sb * ne01 + row_load]);
        __private float dmin = convert_float(src0_dm[expert_d_offset + sb * ne01 + row_load]);

        for (uint g = 0; g < 16; ++g) {
            uint step = sb * QK_K + g * TILESIZE_K;
            uint sc = (uint)src0_s[s_row + sb * 16 + g];
            uint bits = src0_q[expert_q_offset + (sb * 16 + g) * ne01 + row_load];
            reg_a = dequant16_q2(bits, sc, d, dmin, m0F, m03);

            uint b_sub_offset = col * ne00 + step;
            float8 bx8_f32;
            bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
            bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
            half8 bx8_f16 = convert_half8(bx8_f32);
            shared_b[b_local_offset.x] = bx8_f16.lo;
            shared_b[b_local_offset.y] = bx8_f16.hi;

            sub_group_barrier(CLK_LOCAL_MEM_FENCE);

            half8 acc8;
            dotx8_reduce4(reg_a, shared_b, reg_c.lo.lo, 0);
            if (!skip_g1) { dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8); }
            if (!skip_g2) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16); }
            if (!skip_g3) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24); }
        }
    }

    __local uint out_idx[TILESIZE_N];
    if (get_local_id(0) < TILESIZE_N) {
        uint idx = src2[block_id_n * TILESIZE_N + get_local_id(0)];
        if (idx == 0xFFFFFFFF) {
            idx = src2[block_id_n * TILESIZE_N + 0];
        }
        out_idx[get_local_id(0)] = idx * ne01;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint m_offset = row + get_local_id(0);
    if (valid) {
        write_imagef(dst, out_idx[1] + m_offset, (reg_c.s1));
        write_imagef(dst, out_idx[2] + m_offset, (reg_c.s2));
        write_imagef(dst, out_idx[3] + m_offset, (reg_c.s3));
        write_imagef(dst, out_idx[4] + m_offset, (reg_c.s4));
        write_imagef(dst, out_idx[5] + m_offset, (reg_c.s5));
        write_imagef(dst, out_idx[6] + m_offset, (reg_c.s6));
        write_imagef(dst, out_idx[7] + m_offset, (reg_c.s7));
        write_imagef(dst, out_idx[8] + m_offset, (reg_c.s8));
        write_imagef(dst, out_idx[9] + m_offset, (reg_c.s9));
        write_imagef(dst, out_idx[10] + m_offset, (reg_c.sa));
        write_imagef(dst, out_idx[11] + m_offset, (reg_c.sb));
        write_imagef(dst, out_idx[12] + m_offset, (reg_c.sc));
        write_imagef(dst, out_idx[13] + m_offset, (reg_c.sd));
        write_imagef(dst, out_idx[14] + m_offset, (reg_c.se));
        write_imagef(dst, out_idx[15] + m_offset, (reg_c.sf));
        write_imagef(dst, out_idx[16] + m_offset, (reg_c.sg));
        write_imagef(dst, out_idx[17] + m_offset, (reg_c.sh));
        write_imagef(dst, out_idx[18] + m_offset, (reg_c.si));
        write_imagef(dst, out_idx[19] + m_offset, (reg_c.sj));
        write_imagef(dst, out_idx[20] + m_offset, (reg_c.sk));
        write_imagef(dst, out_idx[21] + m_offset, (reg_c.sl));
        write_imagef(dst, out_idx[22] + m_offset, (reg_c.sm));
        write_imagef(dst, out_idx[23] + m_offset, (reg_c.sn));
        write_imagef(dst, out_idx[24] + m_offset, (reg_c.so));
        write_imagef(dst, out_idx[25] + m_offset, (reg_c.sp));
        write_imagef(dst, out_idx[26] + m_offset, (reg_c.sq));
        write_imagef(dst, out_idx[27] + m_offset, (reg_c.sr));
        write_imagef(dst, out_idx[28] + m_offset, (reg_c.ss));
        write_imagef(dst, out_idx[29] + m_offset, (reg_c.st));
        write_imagef(dst, out_idx[30] + m_offset, (reg_c.su));
        write_imagef(dst, out_idx[31] + m_offset, (reg_c.sv));
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (valid) {
        write_imagef(dst, out_idx[0] + m_offset, (reg_c.s0));
    }
}
