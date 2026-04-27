// SSM_SCAN — Mamba selective scan, F32 only.
// One workgroup of WG_SIZE threads per (head_dim i1, head h, seq i3); lane k
// handles d_state indices i0 in {k, k+WG_SIZE, k+2*WG_SIZE, ...}.
// Mamba-1: A is {d_state, n_head}    -> per-i0 dA = exp(dt_softplus * A[i0, h])
// Mamba-2: A is {1, n_head}           -> scalar  dA = exp(dt_softplus * A[h])

#ifdef USE_SUBGROUP_REDUCTION
diagnostic(off, subgroup_uniformity);
enable subgroups;
#endif

@group(0) @binding(0) var<storage, read>       src0: array<f32>;  // s
@group(0) @binding(1) var<storage, read>       src1: array<f32>;  // x
@group(0) @binding(2) var<storage, read>       src2: array<f32>;  // dt
@group(0) @binding(3) var<storage, read>       src3: array<f32>;  // A
@group(0) @binding(4) var<storage, read>       src4: array<f32>;  // B
@group(0) @binding(5) var<storage, read>       src5: array<f32>;  // C
@group(0) @binding(6) var<storage, read>       src6: array<i32>;  // ids
@group(0) @binding(7) var<storage, read_write> dst:  array<f32>;  // [y | s_out]

struct Params {
    offset_src0: u32, offset_src1: u32, offset_src2: u32,
    offset_src3: u32, offset_src4: u32, offset_src5: u32,
    offset_src6: u32, offset_dst:  u32,

    nc: u32,
    nr: u32,
    nh: u32,
    ng: u32,
    nt: u32,
    ns: u32,

    stride_s_3:  u32,
    stride_x_2:  u32,
    stride_x_3:  u32,
    stride_dt_1: u32,
    stride_dt_2: u32,
    stride_BC_2: u32,
    stride_BC_3: u32,

    is_mamba2: u32,
    s_off:     u32,
};

@group(0) @binding(8) var<uniform> params: Params;

// Sized for WG_SIZE: covers num_subgroups <= WG_SIZE for the subgroup path
// and the full power-of-two tree for the workgroup-reduction fallback.
var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn softplus(x: f32) -> f32 {
    return select(log(1.0 + exp(x)), x, x > 20.0);
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>
#ifdef USE_SUBGROUP_REDUCTION
      , @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let lane = lid.x;

    let i1 = wid.x % params.nr;
    let h  = wid.x / params.nr;
    let i3 = wid.y;

    let nc = params.nc;
    let nr = params.nr;
    let nh = params.nh;
    let ng = params.ng;
    let nt = params.nt;

    // Defensive: caller asserts nh % ng == 0 and ng >= 1, but a bad input
    // would otherwise hit GPU divide-by-zero.
    let heads_per_group = select(1u, nh / ng, ng > 0u && nh >= ng);
    let g  = h / heads_per_group;
    let ii = i1 + h * nr;

    let s_out_base = params.offset_dst + params.s_off + i3 * params.stride_s_3 + ii * nc;

    // Each lane writes only its own i0 slots, then reads only its own i0 slots
    // in the loop below, so no cross-lane barrier is needed before the loop.
    let id_idx = u32(src6[params.offset_src6 + i3]);
    let s_in_base = params.offset_src0 + id_idx * params.stride_s_3 + ii * nc;
    for (var i0 = lane; i0 < nc; i0 += WG_SIZE) {
        dst[s_out_base + i0] = src0[s_in_base + i0];
    }

    for (var i2: u32 = 0u; i2 < nt; i2++) {
        let B_token_base = params.offset_src4 + g * nc + i2 * params.stride_BC_2 + i3 * params.stride_BC_3;
        let C_token_base = params.offset_src5 + g * nc + i2 * params.stride_BC_2 + i3 * params.stride_BC_3;

        let dt_h = src2[params.offset_src2 + h + i2 * params.stride_dt_1 + i3 * params.stride_dt_2];
        let dt_softplus = softplus(dt_h);
        let x_val = src1[params.offset_src1 + ii + i2 * params.stride_x_2 + i3 * params.stride_x_3];
        let x_dt = x_val * dt_softplus;

        var lane_sum: f32 = 0.0;

        if (params.is_mamba2 != 0u) {
            let dA = exp(dt_softplus * src3[params.offset_src3 + h]);
            // vec4 fast path: each lane decodes 4 contiguous i0 per iteration.
            let nc_v4 = nc & ~3u;
            for (var base = lane * 4u; base < nc_v4; base += WG_SIZE * 4u) {
                let b_idx = B_token_base + base;
                let c_idx = C_token_base + base;
                let s_idx = s_out_base + base;
                let b4 = vec4<f32>(src4[b_idx], src4[b_idx+1u], src4[b_idx+2u], src4[b_idx+3u]);
                let c4 = vec4<f32>(src5[c_idx], src5[c_idx+1u], src5[c_idx+2u], src5[c_idx+3u]);
                let p4 = vec4<f32>(dst[s_idx], dst[s_idx+1u], dst[s_idx+2u], dst[s_idx+3u]);
                let st = p4 * dA + b4 * x_dt;
                dst[s_idx]      = st.x;
                dst[s_idx+1u]   = st.y;
                dst[s_idx+2u]   = st.z;
                dst[s_idx+3u]   = st.w;
                lane_sum += dot(st, c4);
            }
            // Scalar tail for nc % 4 != 0 (does not occur in real Mamba-2 models).
            for (var i0 = nc_v4 + lane; i0 < nc; i0 += WG_SIZE) {
                let b = src4[B_token_base + i0];
                let c = src5[C_token_base + i0];
                let prev = dst[s_out_base + i0];
                let state = prev * dA + b * x_dt;
                dst[s_out_base + i0] = state;
                lane_sum += state * c;
            }
        } else {
            for (var i0 = lane; i0 < nc; i0 += WG_SIZE) {
                let dA = exp(dt_softplus * src3[params.offset_src3 + i0 + h * nc]);
                let b = src4[B_token_base + i0];
                let c = src5[C_token_base + i0];
                let prev = dst[s_out_base + i0];
                let state = prev * dA + b * x_dt;
                dst[s_out_base + i0] = state;
                lane_sum += state * c;
            }
        }

        var sumf: f32;

#ifdef USE_SUBGROUP_REDUCTION
        // Two-pass reduction: subgroupAdd within each subgroup, then again
        // across the per-subgroup partials. Correct for any subgroup_size.
        let sg_total = subgroupAdd(lane_sum);
        if (subgroup_invocation_id == 0u) {
            partial_sums[subgroup_id] = sg_total;
        }
        workgroupBarrier();

        var cross_sg_acc = 0.0f;
        if (subgroup_id == 0u) {
            for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
                cross_sg_acc += partial_sums[k];
            }
        }
        let cross_sg_total = subgroupAdd(cross_sg_acc);
        if (lane == 0u) {
            sumf = cross_sg_total;
        }
#endif

#ifdef USE_WORKGROUP_REDUCTION
        partial_sums[lane] = lane_sum;
        workgroupBarrier();

        var stride = WG_SIZE >> 1u;
        while (stride > 0u) {
            if (lane < stride) {
                partial_sums[lane] += partial_sums[lane + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lane == 0u) {
            sumf = partial_sums[0];
        }
#endif

        if (lane == 0u) {
            let y_idx = params.offset_dst + ii + i2 * (nh * nr) + i3 * (nt * nh * nr);
            dst[y_idx] = sumf;
        }
        // Required: next iteration overwrites partial_sums.
        workgroupBarrier();
    }
}
