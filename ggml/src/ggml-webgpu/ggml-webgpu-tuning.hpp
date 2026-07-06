#ifndef GGML_WEBGPU_TUNING_HPP
#define GGML_WEBGPU_TUNING_HPP

// WebGPU tuning library.
//
// Central place for the compile-time shader knobs (workgroup sizes, tile
// sizes, outputs-per-workgroup, subgroup-matrix layout) used by the matmul
// family of kernels. The values are selected per device, per size class and
// per model, so a single build can carry tuned parameters for many targets.
//
// Selection is a most-specific-match lookup over a static table with a safe
// fallback: an unmatched selector returns the generic defaults, which are the
// values the backend shipped with, so behavior is unchanged unless a tuned
// entry applies.

#include "ggml-impl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

// Register-tile / subgroup-matrix matmul (prefill) knobs. Default values are
// the backend baseline and act as the generic profile.
struct ggml_webgpu_mul_mat_knobs {
    uint32_t tile_m                = 4;
    uint32_t tile_n                = 4;
    uint32_t wg_size_m             = 8;
    uint32_t wg_size_n             = 8;
    uint32_t reg_tile_k_float      = 8;
    uint32_t reg_tile_k_quant      = 32;
    uint32_t subgroup_m            = 2;
    uint32_t subgroup_n            = 4;
    uint32_t subgroup_matrix_m     = 4;
    uint32_t subgroup_matrix_n     = 2;
    uint32_t subgroup_tile_k_float = 32;
    uint32_t subgroup_tile_k_quant = 32;
    uint32_t wg_size               = 256;
};

// Matrix-vector matmul (decode) knobs.
struct ggml_webgpu_mul_mat_vec_knobs {
    uint32_t wg_size                 = 256;
    uint32_t float_outputs_per_wg    = 4;
    uint32_t legacy_q_outputs_per_wg = 4;
    uint32_t k_q_outputs_per_wg      = 4;
};

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------

// Problem size class, derived from the number of columns (tokens) of the
// matmul destination. Decode uses the mat-vec kernels, prefill the reg-tile /
// subgroup-matrix kernels; prefill is split so wide GEMMs can differ from
// short ones.
enum ggml_webgpu_size_class {
    GGML_WEBGPU_SIZE_DECODE = 0,     // N <= 4
    GGML_WEBGPU_SIZE_PREFILL_SMALL,  // 4 < N <= 128
    GGML_WEBGPU_SIZE_PREFILL_LARGE,  // N > 128
};

inline ggml_webgpu_size_class ggml_webgpu_classify_size(uint32_t n_tokens) {
    if (n_tokens <= 4) {
        return GGML_WEBGPU_SIZE_DECODE;
    }
    if (n_tokens <= 128) {
        return GGML_WEBGPU_SIZE_PREFILL_SMALL;
    }
    return GGML_WEBGPU_SIZE_PREFILL_LARGE;
}

inline const char * ggml_webgpu_size_class_name(ggml_webgpu_size_class sc) {
    switch (sc) {
        case GGML_WEBGPU_SIZE_DECODE:
            return "decode";
        case GGML_WEBGPU_SIZE_PREFILL_SMALL:
            return "prefill_small";
        case GGML_WEBGPU_SIZE_PREFILL_LARGE:
            return "prefill_large";
    }
    return "unknown";
}

// Everything the backend can observe about the target, collapsed into the
// keys the library selects on.
struct ggml_webgpu_tuning_selector {
    std::string            vendor;      // e.g. "nvidia"
    std::string            arch;        // e.g. "blackwell"
    std::string            device;      // e.g. "NVIDIA GeForce RTX 5080"
    std::string            model;       // e.g. "llama-3.2-1b" or a shape signature
    ggml_webgpu_size_class size_class = GGML_WEBGPU_SIZE_DECODE;
};

// ---------------------------------------------------------------------------
// Tuning database
// ---------------------------------------------------------------------------

// One row of the database. An empty ("") match string is a wildcard; a
// size_class of -1 matches any size. `device` matches as a substring of the
// selector device name so a family ("RTX 50") can cover several cards.
struct ggml_webgpu_mul_mat_row {
    const char *              vendor;
    const char *              device;
    const char *              model;
    int                       size_class;
    ggml_webgpu_mul_mat_knobs knobs;
};

struct ggml_webgpu_mul_mat_vec_row {
    const char *                  vendor;
    const char *                  device;
    const char *                  model;
    int                           size_class;
    ggml_webgpu_mul_mat_vec_knobs knobs;
};

// NVIDIA profile. Values below are the current baseline seeds; refine per
// (device, size class, model) as benchmark data becomes available. Rows are
// scanned top to bottom and the most specific match wins, so order does not
// matter for correctness, only wildcards vs. concrete fields.
static const ggml_webgpu_mul_mat_row GGML_WEBGPU_MUL_MAT_TABLE[] = {
    // vendor    device  model  size_class                       knobs
    { "nvidia",  "",     "",    GGML_WEBGPU_SIZE_PREFILL_SMALL,  {} },
    { "nvidia",  "",     "",    GGML_WEBGPU_SIZE_PREFILL_LARGE,  {} },
};

static const ggml_webgpu_mul_mat_vec_row GGML_WEBGPU_MUL_MAT_VEC_TABLE[] = {
    // vendor    device  model  size_class               knobs
    { "nvidia",  "",     "",    GGML_WEBGPU_SIZE_DECODE, {} },
};

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

namespace ggml_webgpu_tuning_detail {

// Returns -1 if the row does not match, otherwise a specificity score (higher
// is more specific) so the best match can be picked.
inline int match_score(const char *                        row_vendor,
                       const char *                        row_device,
                       const char *                        row_model,
                       int                                 row_size_class,
                       const ggml_webgpu_tuning_selector & sel) {
    int score = 0;
    if (row_vendor[0]) {
        if (sel.vendor != row_vendor) {
            return -1;
        }
        score += 1;
    }
    if (row_device[0]) {
        if (sel.device.find(row_device) == std::string::npos) {
            return -1;
        }
        score += 2;
    }
    if (row_model[0]) {
        if (sel.model != row_model) {
            return -1;
        }
        score += 4;
    }
    if (row_size_class >= 0) {
        if ((int) sel.size_class != row_size_class) {
            return -1;
        }
        score += 1;
    }
    return score;
}

}  // namespace ggml_webgpu_tuning_detail

inline ggml_webgpu_mul_mat_knobs ggml_webgpu_lookup_mul_mat(const ggml_webgpu_tuning_selector & sel) {
    ggml_webgpu_mul_mat_knobs best = {};  // generic default
    int                       best_score = -1;
    for (const auto & row : GGML_WEBGPU_MUL_MAT_TABLE) {
        int s = ggml_webgpu_tuning_detail::match_score(row.vendor, row.device, row.model, row.size_class, sel);
        if (s > best_score) {
            best_score = s;
            best       = row.knobs;
        }
    }
    return best;
}

inline ggml_webgpu_mul_mat_vec_knobs ggml_webgpu_lookup_mul_mat_vec(const ggml_webgpu_tuning_selector & sel) {
    ggml_webgpu_mul_mat_vec_knobs best = {};  // generic default
    int                           best_score = -1;
    for (const auto & row : GGML_WEBGPU_MUL_MAT_VEC_TABLE) {
        int s = ggml_webgpu_tuning_detail::match_score(row.vendor, row.device, row.model, row.size_class, sel);
        if (s > best_score) {
            best_score = s;
            best       = row.knobs;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Stable ids
// ---------------------------------------------------------------------------

// FNV-1a over the resolved knob values. Used to key the pipeline cache so that
// distinct knob sets compile distinct pipelines, while identical knobs share
// one. Two selectors that resolve to the same knobs collapse to one pipeline.
inline uint32_t ggml_webgpu_knobs_hash(const uint32_t * words, size_t count) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < count; i++) {
        h ^= words[i];
        h *= 16777619u;
    }
    return h;
}

inline uint32_t ggml_webgpu_knobs_id(const ggml_webgpu_mul_mat_knobs & k) {
    const uint32_t words[] = { k.tile_m,           k.tile_n,
                               k.wg_size_m,        k.wg_size_n,
                               k.reg_tile_k_float, k.reg_tile_k_quant,
                               k.subgroup_m,       k.subgroup_n,
                               k.subgroup_matrix_m, k.subgroup_matrix_n,
                               k.subgroup_tile_k_float, k.subgroup_tile_k_quant,
                               k.wg_size };
    return ggml_webgpu_knobs_hash(words, sizeof(words) / sizeof(words[0]));
}

inline uint32_t ggml_webgpu_knobs_id(const ggml_webgpu_mul_mat_vec_knobs & k) {
    const uint32_t words[] = { k.wg_size, k.float_outputs_per_wg, k.legacy_q_outputs_per_wg, k.k_q_outputs_per_wg };
    return ggml_webgpu_knobs_hash(words, sizeof(words) / sizeof(words[0]));
}

// ---------------------------------------------------------------------------
// Model identification
// ---------------------------------------------------------------------------

// Best-effort model fingerprint from a compute graph. The backend never sees
// the model name, so it is inferred from the matmul shapes: n_embd is the most
// common reduction dim, n_vocab the largest output dim. Known signatures map
// to friendly names; anything else returns a compact signature so it is still
// a stable, distinct key. An env override wins over inference.
inline std::string ggml_webgpu_resolve_model_id(const ggml_cgraph * cgraph, const std::string & vendor) {
    if (const char * env = std::getenv("GGML_WEBGPU_TUNING_MODEL")) {
        if (env[0]) {
            return env;
        }
    }
    GGML_UNUSED(vendor);

    // Collect matmul reduction dims (K) and output dims (M).
    uint32_t n_embd  = 0;  // most frequent K
    uint32_t n_ff    = 0;  // largest K seen (ffn projections)
    uint32_t n_vocab = 0;  // largest M seen (output projection)

    // Track the most frequent K with a tiny frequency table.
    uint32_t k_vals[16]  = {};
    uint32_t k_count[16] = {};
    int      n_k         = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID)) {
            continue;
        }
        const ggml_tensor * w = node->src[0];
        if (!w) {
            continue;
        }
        const uint32_t k = (uint32_t) w->ne[0];
        const uint32_t m = (uint32_t) w->ne[1];
        if (k > n_ff) {
            n_ff = k;
        }
        if (m > n_vocab) {
            n_vocab = m;
        }
        int slot = -1;
        for (int j = 0; j < n_k; j++) {
            if (k_vals[j] == k) {
                slot = j;
                break;
            }
        }
        if (slot < 0 && n_k < 16) {
            slot          = n_k++;
            k_vals[slot]  = k;
            k_count[slot] = 0;
        }
        if (slot >= 0) {
            k_count[slot]++;
        }
    }

    uint32_t best_count = 0;
    for (int j = 0; j < n_k; j++) {
        if (k_count[j] > best_count) {
            best_count = k_count[j];
            n_embd     = k_vals[j];
        }
    }

    if (n_embd == 0) {
        return "generic";
    }

    // Known small models (n_embd is the primary discriminator).
    if (n_embd == 2048 && n_vocab >= 128000) {
        return "llama-3.2-1b";
    }
    if (n_embd == 640) {
        return "gemma-3-270m";
    }
    if (n_embd == 1152) {
        return "gemma-3-1b";
    }
    if (n_embd == 1024 && n_vocab >= 150000) {
        return "qwen3-0.6b";
    }

    return std::string("e") + std::to_string(n_embd) + "_f" + std::to_string(n_ff) + "_v" + std::to_string(n_vocab);
}

#endif  // GGML_WEBGPU_TUNING_HPP
