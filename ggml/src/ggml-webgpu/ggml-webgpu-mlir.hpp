#ifndef GGML_WEBGPU_MLIR_HPP
#define GGML_WEBGPU_MLIR_HPP

#include <mlir/Target/WGSL/MatmulCompiler.h>
#include <filesystem>
#include <fstream>
#include <sstream>

// The MLIR input for one matmul. The compiler chooses the schedule.
inline std::string ggml_webgpu_mlir_matmul_source(int64_t m, int64_t n, int64_t k, bool rhs_f16, bool acc_f16, bool gemv = false) {
    const std::string a = "tensor<" + std::to_string(m) + "x" + std::to_string(k) + "xf16>";
    const std::string b = "tensor<" + std::to_string(n) + "x" + std::to_string(k) + (rhs_f16 ? "xf16>" : "xf32>");
    const std::string h = "tensor<" + std::to_string(n) + "x" + std::to_string(k) + "xf16>";
    const std::string c = "tensor<" + std::to_string(m) + "x" + std::to_string(n) + "xf32>";
    const std::string acc = "tensor<" + std::to_string(m) + "x" + std::to_string(n) + (acc_f16 ? "xf16>" : "xf32>");
    const char * acc_type = acc_f16 ? "f16" : "f32";
    std::ostringstream ir;
    ir << "#id = affine_map<(i, j) -> (i, j)>\n"
       << "func.func @matmul(%a: " << a << ", %b: " << b << ") -> " << c << " {\n";
    if (!rhs_f16 && !gemv) {
        ir << "  %tmp = tensor.empty() : " << h << "\n"
           << "  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = [\"parallel\", \"parallel\"]} "
           << "ins(%b : " << b << ") outs(%tmp : " << h << ") {\n"
           << "  ^bb0(%x: f32, %unused: f16):\n"
           << "    %h = arith.truncf %x : f32 to f16\n"
           << "    linalg.yield %h : f16\n"
           << "  } -> " << h << "\n";
    }
    ir << "  %zero = arith.constant 0.0 : " << acc_type << "\n"
       << "  %empty = tensor.empty() : " << acc << "\n"
       << "  %init = linalg.fill ins(%zero : " << acc_type << ") outs(%empty : " << acc << ") -> " << acc << "\n"
       << "  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, " << (rhs_f16 || gemv ? "%b" : "%half") << " : " << a << ", " << (gemv ? b : h)
       << ") outs(%init : " << acc << ") -> " << acc << "\n";
    if (acc_f16) {
        ir << "  %out = tensor.empty() : " << c << "\n"
           << "  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = [\"parallel\", \"parallel\"]} "
           << "ins(%product : " << acc << ") outs(%out : " << c << ") {\n"
           << "  ^bb0(%x: f16, %unused: f32):\n"
           << "    %f = arith.extf %x : f16 to f32\n"
           << "    linalg.yield %f : f32\n"
           << "  } -> " << c << "\n";
    }
    ir << "  return " << (acc_f16 ? "%result" : "%product") << " : " << c << "\n}\n";
    return ir.str();
}

inline void ggml_webgpu_mlir_dump(const std::string & name, const std::string & input, const MlirWgslMatmulKernel & kernel) {
    const char * directory = getenv("GGML_WEBGPU_MLIR_DUMP_DIR");
    if (directory == nullptr) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        GGML_ABORT("Cannot create MLIR dump directory: %s", error.message().c_str());
    }
    const auto prefix = std::filesystem::path(directory) / name;
    auto write = [&](const char * extension, const char * text) {
        std::ofstream file(prefix.string() + extension);
        file << text;
        file.close();
        if (!file) {
            GGML_ABORT("Cannot write MLIR dump: %s", prefix.string().c_str());
        }
    };
    write(".input.mlir", input.c_str());
    write(".scheduled.mlir", kernel.scheduled_mlir);
    write(".lowered.mlir", kernel.lowered_mlir);
    write(".wgsl", kernel.wgsl);
}

#endif
