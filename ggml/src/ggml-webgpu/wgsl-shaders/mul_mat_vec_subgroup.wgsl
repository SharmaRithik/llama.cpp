diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;

#include "common_decls.tmpl"

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

// Each subgroup handles one output row.
// WG_SIZE threads / subgroup_size lanes = rows per workgroup.
// Within a subgroup, lanes split the K-dimension sub-blocks.

#ifdef IQ1_S_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 50u;
const NUM_WORK_ITEMS_IQ1_S = 16u; // 8 sub-blocks × 2 halves

fn dequant_dot_iq1_s(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let d = load_f16_as_f32_at(&src0, block_byte_base);
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        for (var wi = lane_id; wi < NUM_WORK_ITEMS_IQ1_S; wi += num_lanes) {
            let ib = wi / 2u;
            let half = wi % 2u;
            let qh = load_u32_at(&src0, block_byte_base + 34 + ib * 2) & 0xFFFF;
            let dl = d * (2.0 * f32((qh >> 12) & 7) + 1.0);
            let delta = select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x8000) != 0);
            let qs_w = load_u32_at(&src0, block_byte_base + 2 + ib * 4);

            var sub_sum = 0.0;
            for (var l = half * 2u; l < half * 2u + 2u; l++) {
                let ig = (get_byte(qs_w, l) | (((qh >> (3 * l)) & 7) << 8)) * 8;
                for (var j: u32 = 0; j < 8; j++) {
                    let gw = iq1_grid[(ig + j) / 16];
                    let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                    let gs = bitcast<i32>(g << 30) >> 30;
                    sub_sum += (f32(gs) + delta) * src1[src1_block_base + ib * 32 + l * 8 + j];
                }
            }
            local_sum += dl * sub_sum;
        }
    }
    return local_sum;
}
#endif

#ifdef IQ1_M_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 56u;
const NUM_WORK_ITEMS_IQ1_M = 16u;

fn dequant_dot_iq1_m(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        let scales0 = load_u32_at(&src0, block_byte_base + 48);
        let scales1 = load_u32_at(&src0, block_byte_base + 52);
        let scale_bits = ((scales0 >> 12) & 0xF) | ((scales0 >> 24) & 0x00F0) | ((scales1 >> 4) & 0x0F00) | ((scales1 >> 16) & 0xF000);
        let d = f32(bitcast<vec2<f16>>(scale_bits).x);

        for (var wi = lane_id; wi < NUM_WORK_ITEMS_IQ1_M; wi += num_lanes) {
            let ib = wi / 2u;
            let half = wi % 2u;

            let sw_offset = block_byte_base + 48 + (ib / 4) * 4;
            let sw = (load_u32_at(&src0, sw_offset) >> (16 * ((ib / 2) % 2))) & 0xFFFF;
            let s1 = (sw >> (6 * (ib % 2))) & 0x7;
            let s2 = (sw >> (6 * (ib % 2) + 3)) & 0x7;
            var dl = array<f32, 2>(
                d * f32(2 * s1 + 1),
                d * f32(2 * s2 + 1)
            );

            let qh_offset = block_byte_base + 32 + (ib / 2) * 4;
            let qh = load_u32_at(&src0, qh_offset) >> (16 * (ib % 2));
            let qs_w = load_u32_at(&src0, block_byte_base + ib * 4);

            var idx = array<u32, 4>(
                get_byte(qs_w, 0) | ((qh << 8) & 0x700),
                get_byte(qs_w, 1) | ((qh << 4) & 0x700),
                get_byte(qs_w, 2) | ((qh) & 0x700),
                get_byte(qs_w, 3) | ((qh >> 4) & 0x700)
            );
            var delta = array<f32, 4>(
                select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x08) != 0),
                select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x80) != 0),
                select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x08) != 0),
                select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x80) != 0)
            );

            var sub_sum = 0.0;
            for (var l = half * 2u; l < half * 2u + 2u; l++) {
                let ig = idx[l] * 8;
                for (var j: u32 = 0; j < 8; j++) {
                    let gw = iq1_grid[(ig + j) / 16];
                    let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                    let gs = bitcast<i32>(g << 30) >> 30;
                    sub_sum += dl[l / 2] * (f32(gs) + delta[l]) * src1[src1_block_base + ib * 32 + l * 8 + j];
                }
            }
            local_sum += sub_sum;
        }
    }
    return local_sum;
}
#endif

#ifdef IQ2_XXS_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 66u;
const NUM_WORK_ITEMS_IQ2_XXS = 16u; // 8 groups × 2 halves

fn dequant_dot_iq2_xxs(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let d = load_f16_as_f32_at(&src0, block_byte_base);
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        for (var wi = lane_id; wi < NUM_WORK_ITEMS_IQ2_XXS; wi += num_lanes) {
            let ig4 = wi / 2u;
            let half = wi % 2u;
            let ib = ig4 * 4u;
            let aux0 = load_u32_at(&src0, block_byte_base + 2 + ib * 2);
            let aux1 = load_u32_at(&src0, block_byte_base + 2 + (ib + 2) * 2);
            let db = d * (0.5 + f32(aux1 >> 28)) * 0.25;

            var sub_sum = 0.0;
            for (var l = half * 2u; l < half * 2u + 2u; l++) {
                let ig = get_byte(aux0, l) * 8;
                let is = (aux1 >> (7 * l)) & 127;
                let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
                for (var j: u32 = 0; j < 8; j++) {
                    let g = get_byte(iq2xxs_grid[(ig + j) / 4], (ig + j) % 4);
                    let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                    sub_sum += f32(g) * m * src1[src1_block_base + ib * 8 + l * 8 + j];
                }
            }
            local_sum += db * sub_sum;
        }
    }
    return local_sum;
}
#endif

#ifdef IQ2_S_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 82u;
const NUM_WORK_ITEMS_IQ2_S = 16u; // 8 sub-blocks × 2 halves

fn dequant_dot_iq2_s(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let d = load_f16_as_f32_at(&src0, block_byte_base);
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        for (var wi = lane_id; wi < NUM_WORK_ITEMS_IQ2_S; wi += num_lanes) {
            let ib = wi / 2u;
            let half = wi % 2u;
            let s = get_byte(load_u32_at(&src0, block_byte_base + 74 + (ib / 4) * 4), ib % 4);
            let db = array<f32, 2>(
                d * (0.5 + f32(s & 0xF)) * 0.25,
                d * (0.5 + f32(s >> 4)) * 0.25
            );
            let qs_w = load_u32_at(&src0, block_byte_base + 2 + ib * 4);

            var sub_sum = 0.0;
            for (var l = half * 2u; l < half * 2u + 2u; l++) {
                let qh_b = (get_byte(load_u32_at(&src0, block_byte_base + 66 + (ib / 4) * 4), ib % 4) << (8 - 2 * l)) & 0x300;
                let ig = (get_byte(qs_w, l) | qh_b) * 8;
                let signs = get_byte(load_u32_at(&src0, block_byte_base + 2 + (ib + 8) * 4), l);
                let dl = db[l / 2];
                for (var j: u32 = 0; j < 8; j++) {
                    let g = get_byte(iq2s_grid[(ig + j) / 4], (ig + j) % 4);
                    let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                    sub_sum += dl * f32(g) * m * src1[src1_block_base + ib * 32 + l * 8 + j];
                }
            }
            local_sum += sub_sum;
        }
    }
    return local_sum;
}
#endif

#ifdef IQ3_S_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 110u;
const NUM_WORK_ITEMS_IQ3_S = 16u; // 4 super-blocks × 2 halves × 2 group-pairs

fn dequant_dot_iq3_s(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let d = load_f16_as_f32_at(&src0, block_byte_base);
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        let scale_vals = load_u32_at(&src0, block_byte_base + 106);

        for (var wi = lane_id; wi < NUM_WORK_ITEMS_IQ3_S; wi += num_lanes) {
            let ib = wi / 4u;
            let k = (wi / 2u) % 2u;
            let group_half = wi % 2u;

            let s = get_byte(scale_vals, ib);
            let dl = select(
                d * (1.0 + 2.0 * f32(s & 0xF)),
                d * (1.0 + 2.0 * f32(s >> 4)),
                k >= 1u
            );

            let qh_byte = get_byte(load_u32_at(&src0, block_byte_base + 66 + (ib / 2) * 4), (ib % 2) * 2 + k);
            let sign_w = load_u32_at(&src0, block_byte_base + 74 + (ib * 2 + k) * 4);

            var sub_sum = 0.0;
            for (var l = group_half * 2u; l < group_half * 2u + 2u; l++) {
                let signs = get_byte(sign_w, l);
                let ig_val = load_u32_at(&src0, block_byte_base + 2 + (ib * 8 + k * 4 + l) * 2) & 0xFFFF;
                let ig1 = get_byte(ig_val, 0) | ((qh_byte << (8 - 2 * l)) & 256);
                let ig2 = get_byte(ig_val, 1) | ((qh_byte << (7 - 2 * l)) & 256);
                let sv_idx = src1_block_base + ib * 64 + k * 32 + l * 8;
                for (var j: u32 = 0; j < 4; j++) {
                    let g1 = get_byte(iq3s_grid[ig1], j);
                    let g2 = get_byte(iq3s_grid[ig2], j);
                    let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                    let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                    sub_sum += dl * f32(g1) * m1 * src1[sv_idx + j];
                    sub_sum += dl * f32(g2) * m2 * src1[sv_idx + j + 4];
                }
            }
            local_sum += sub_sum;
        }
    }
    return local_sum;
}
#endif

#ifdef Q4_K_SG
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 144u;

fn get_scale_min_raw(is: u32, scales_base: u32) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(load_u32_at(&src0, scales_base + is), 0u);
        let min_byte = get_byte(load_u32_at(&src0, scales_base + is + 4), 0u);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(load_u32_at(&src0, scales_base + is + 4), 0u);
        let sc_hi = get_byte(load_u32_at(&src0, scales_base + is - 4), 0u);
        let min_hi = get_byte(load_u32_at(&src0, scales_base + is), 0u);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}

fn dequant_dot_q4_k(lane_id: u32, num_lanes: u32, src0_idx_base: u32, src1_idx_base: u32) -> f32 {
    let num_blocks = params.k / BLOCK_SIZE;
    var local_sum = 0.0;

    for (var blk = 0u; blk < num_blocks; blk++) {
        let block_byte_base = (src0_idx_base + blk) * BLOCK_SIZE_BYTES;
        let d = load_f16_as_f32_at(&src0, block_byte_base);
        let m = load_f16_as_f32_at(&src0, block_byte_base + 2);
        let scales_base = block_byte_base + 4;
        let qs_base = block_byte_base + 16;
        let src1_block_base = src1_idx_base + blk * BLOCK_SIZE;

        // 8 sub-blocks of 32 elements, split across lanes
        for (var is = lane_id; is < 8u; is += num_lanes) {
            let scale_min = get_scale_min_raw(is, scales_base);
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            let q_b_idx = (is / 2) * 32u;
            let shift = (is % 2) * 4u;
            let k_base = is * 32u;

            var sub_sum = 0.0;
            for (var l = 0u; l < 32u; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(load_u32_at(&src0, qs_base + q_idx), 0u);
                let qs_val = (q_byte >> shift) & 0xF;
                sub_sum += (f32(qs_val) * dl - ml) * src1[src1_block_base + k_base + l];
            }
            local_sum += sub_sum;
        }
    }
    return local_sum;
}
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32,
        @builtin(subgroup_id) subgroup_id: u32) {

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let rows_per_wg = WG_SIZE / subgroup_size;
    let global_row = wg_linear * rows_per_wg + subgroup_id;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let total_rows = params.m * total_batches;

    if (global_row >= total_rows) {
        return;
    }

    let batch_idx = global_row / params.m;
    let col = global_row % params.m;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    var partial = 0.0;
#ifdef IQ1_S_SG
    partial = dequant_dot_iq1_s(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif
#ifdef IQ1_M_SG
    partial = dequant_dot_iq1_m(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif
#ifdef IQ2_XXS_SG
    partial = dequant_dot_iq2_xxs(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif
#ifdef IQ2_S_SG
    partial = dequant_dot_iq2_s(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif
#ifdef IQ3_S_SG
    partial = dequant_dot_iq3_s(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif
#ifdef Q4_K_SG
    partial = dequant_dot_q4_k(sg_inv_id, subgroup_size, src0_idx_base, src1_idx_base);
#endif

    let result = subgroupAdd(partial);

    if (sg_inv_id == 0u) {
        dst[params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + col] = result;
    }
}
