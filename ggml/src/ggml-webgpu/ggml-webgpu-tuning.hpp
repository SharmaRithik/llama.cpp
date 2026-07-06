#ifndef GGML_WEBGPU_TUNING_HPP
#define GGML_WEBGPU_TUNING_HPP

// WebGPU tuning library.
//
// Central place for the compile-time shader knobs (workgroup sizes, tile
// sizes, outputs-per-workgroup, subgroup-matrix layout) used by the matmul
// family of kernels. Knobs are selected on three axes -- device, size class
// and model -- each an enum, and looked up in a hash map of overrides.
//
// A lookup builds the most-specific key and falls back through less-specific
// keys in a fixed order; anything not present in the map uses the compiled-in
// struct defaults, which are the values the backend shipped with. An empty
// map therefore means "baseline everywhere" and adding a row is how a target
// gets tuned.

#include "ggml-impl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>

// ============================================================================
// Knobs
// ============================================================================

// Register-tile / subgroup-matrix matmul (prefill) knobs. Defaults are the
// backend baseline and act as the generic profile.
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

// ============================================================================
// Selection axes
// ============================================================================

// `generic` is the wildcard / fallback value on each axis.
enum class ggml_webgpu_device : uint32_t {
    generic = 0,
    nvidia,
    intel,
    amd,
    apple,
};

enum class ggml_webgpu_model : uint32_t {
    generic = 0,
    llama_3_2_1b,
    gemma_3_270m,
    gemma_3_1b,
    qwen3_0_6b,
};

// Problem size, from the number of columns (tokens) of the matmul output.
// Decode uses the mat-vec kernels, prefill the reg-tile / subgroup-matrix
// kernels; prefill is split so wide GEMMs can differ from short ones.
enum class ggml_webgpu_size_class : uint32_t {
    decode = 0,     // N <= 4
    prefill_small,  // 4 < N <= 128
    prefill_large,  // N > 128
};

inline ggml_webgpu_size_class ggml_webgpu_classify_size(uint32_t n_tokens) {
    if (n_tokens <= 4) {
        return ggml_webgpu_size_class::decode;
    }
    if (n_tokens <= 128) {
        return ggml_webgpu_size_class::prefill_small;
    }
    return ggml_webgpu_size_class::prefill_large;
}

// What the backend observes at pipeline-creation time. Strings are mapped to
// the enum axes inside the lookup.
struct ggml_webgpu_tuning_selector {
    std::string            vendor;      // e.g. "nvidia"
    std::string            arch;        // e.g. "blackwell"
    std::string            device;      // e.g. "NVIDIA GeForce RTX 5080"
    std::string            model;       // e.g. "llama-3.2-1b"
    ggml_webgpu_size_class size_class = ggml_webgpu_size_class::decode;
};

// ============================================================================
// Axis classification (string / shape -> enum), all table-driven
// ============================================================================

// Single source of truth for model <-> canonical name.
struct ggml_webgpu_model_name_row {
    ggml_webgpu_model model;
    const char *      name;
};

static const ggml_webgpu_model_name_row GGML_WEBGPU_MODEL_NAMES[] = {
    { ggml_webgpu_model::llama_3_2_1b, "llama-3.2-1b" },
    { ggml_webgpu_model::gemma_3_270m, "gemma-3-270m" },
    { ggml_webgpu_model::gemma_3_1b,   "gemma-3-1b"   },
    { ggml_webgpu_model::qwen3_0_6b,   "qwen3-0.6b"   },
};

// Known model shape signatures. n_embd is the primary discriminator; n_vocab
// disambiguates families that share a hidden size.
struct ggml_webgpu_model_shape_row {
    ggml_webgpu_model model;
    uint32_t          n_embd;
    uint32_t          n_vocab_min;
};

static const ggml_webgpu_model_shape_row GGML_WEBGPU_MODEL_SHAPES[] = {
    { ggml_webgpu_model::llama_3_2_1b, 2048, 128000 },
    { ggml_webgpu_model::gemma_3_270m, 640,  0      },
    { ggml_webgpu_model::gemma_3_1b,   1152, 0      },
    { ggml_webgpu_model::qwen3_0_6b,   1024, 150000 },
};

inline ggml_webgpu_device ggml_webgpu_classify_device(const std::string & vendor, const std::string & name) {
    GGML_UNUSED(name);  // reserved for finer per-GPU tuning (e.g. by device name)
    static const struct {
        const char *       vendor;
        ggml_webgpu_device device;
    } table[] = {
        { "nvidia", ggml_webgpu_device::nvidia },
        { "intel",  ggml_webgpu_device::intel  },
        { "amd",    ggml_webgpu_device::amd    },
        { "apple",  ggml_webgpu_device::apple  },
    };
    for (const auto & e : table) {
        if (vendor == e.vendor) {
            return e.device;
        }
    }
    return ggml_webgpu_device::generic;
}

inline ggml_webgpu_model ggml_webgpu_model_from_name(const std::string & name) {
    for (const auto & e : GGML_WEBGPU_MODEL_NAMES) {
        if (name == e.name) {
            return e.model;
        }
    }
    return ggml_webgpu_model::generic;
}

inline const char * ggml_webgpu_model_to_name(ggml_webgpu_model model) {
    for (const auto & e : GGML_WEBGPU_MODEL_NAMES) {
        if (model == e.model) {
            return e.name;
        }
    }
    return "generic";
}

inline ggml_webgpu_model ggml_webgpu_model_from_shape(uint32_t n_embd, uint32_t n_vocab) {
    for (const auto & s : GGML_WEBGPU_MODEL_SHAPES) {
        if (n_embd == s.n_embd && n_vocab >= s.n_vocab_min) {
            return s.model;
        }
    }
    return ggml_webgpu_model::generic;
}

// ============================================================================
// Tuning registry (map of overrides, keyed by the three axes)
// ============================================================================

struct ggml_webgpu_tuning_key {
    ggml_webgpu_device     device;
    ggml_webgpu_model      model;
    ggml_webgpu_size_class size_class;

    bool operator==(const ggml_webgpu_tuning_key & o) const {
        return device == o.device && model == o.model && size_class == o.size_class;
    }
};

struct ggml_webgpu_tuning_key_hash {
    size_t operator()(const ggml_webgpu_tuning_key & k) const {
        return ((size_t) k.device << 16) ^ ((size_t) k.model << 8) ^ (size_t) k.size_class;
    }
};

// Overrides only. Anything absent falls back to the struct defaults, so these
// empty maps mean "baseline everywhere". Add rows as real per
// (device, model, size class) sweep data becomes available, e.g.:
//   { { ggml_webgpu_device::nvidia, ggml_webgpu_model::llama_3_2_1b,
//       ggml_webgpu_size_class::prefill_large }, { /*tile_m*/ 8, /*tile_n*/ 8 } },
static const std::unordered_map<ggml_webgpu_tuning_key, ggml_webgpu_mul_mat_knobs, ggml_webgpu_tuning_key_hash>
    GGML_WEBGPU_MUL_MAT_TUNING = {};

static const std::unordered_map<ggml_webgpu_tuning_key, ggml_webgpu_mul_mat_vec_knobs, ggml_webgpu_tuning_key_hash>
    GGML_WEBGPU_MUL_MAT_VEC_TUNING = {};

// ============================================================================
// Lookup
// ============================================================================

// Try the map from most- to least-specific key, then the compiled-in default.
// `generic` on an axis is the wildcard, so the fallback order is:
//   (device, model, size) -> (device, generic, size) -> (generic, generic, size).
template <typename Knobs, typename Map>
inline Knobs ggml_webgpu_tuning_lookup(const Map & table, const ggml_webgpu_tuning_selector & sel) {
    const ggml_webgpu_device device = ggml_webgpu_classify_device(sel.vendor, sel.device);
    const ggml_webgpu_model  model  = ggml_webgpu_model_from_name(sel.model);

    const ggml_webgpu_tuning_key candidates[] = {
        { device,                      model,                      sel.size_class },
        { device,                      ggml_webgpu_model::generic, sel.size_class },
        { ggml_webgpu_device::generic, ggml_webgpu_model::generic, sel.size_class },
    };
    for (const auto & key : candidates) {
        auto it = table.find(key);
        if (it != table.end()) {
            return it->second;
        }
    }
    return {};  // compiled-in defaults
}

inline ggml_webgpu_mul_mat_knobs ggml_webgpu_lookup_mul_mat(const ggml_webgpu_tuning_selector & sel) {
    return ggml_webgpu_tuning_lookup<ggml_webgpu_mul_mat_knobs>(GGML_WEBGPU_MUL_MAT_TUNING, sel);
}

inline ggml_webgpu_mul_mat_vec_knobs ggml_webgpu_lookup_mul_mat_vec(const ggml_webgpu_tuning_selector & sel) {
    return ggml_webgpu_tuning_lookup<ggml_webgpu_mul_mat_vec_knobs>(GGML_WEBGPU_MUL_MAT_VEC_TUNING, sel);
}

// ============================================================================
// Stable ids
// ============================================================================

// FNV-1a over the resolved knob values. Used to key the pipeline cache so that
// distinct knob sets compile distinct pipelines, while identical knobs share
// one.
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

// ============================================================================
// Model identification
// ============================================================================

// Best-effort model name from a compute graph. The backend never sees the
// model name, so it is inferred from the matmul shapes: n_embd is the most
// common reduction dim, n_vocab the largest output dim. Known signatures map
// to a canonical name; anything else returns a compact signature so it is
// still a stable, distinct key. An env override wins over inference.
inline std::string ggml_webgpu_resolve_model_id(const ggml_cgraph * cgraph, const std::string & vendor) {
    if (const char * env = std::getenv("GGML_WEBGPU_TUNING_MODEL")) {
        if (env[0]) {
            return env;
        }
    }
    GGML_UNUSED(vendor);

    uint32_t n_vocab = 0;  // largest output dim (M)

    // Most-frequent reduction dim (K) via a small frequency table.
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

    uint32_t n_embd     = 0;
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

    const ggml_webgpu_model model = ggml_webgpu_model_from_shape(n_embd, n_vocab);
    if (model != ggml_webgpu_model::generic) {
        return ggml_webgpu_model_to_name(model);
    }
    return std::string("e") + std::to_string(n_embd) + "_v" + std::to_string(n_vocab);
}

#endif  // GGML_WEBGPU_TUNING_HPP
