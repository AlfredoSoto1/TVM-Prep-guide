// #include <tvm/runtime/graph_executor.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>

// #include <fstream>
// #include <iostream>
// #include <stdexcept>
// #include <vector>

// static std::string ReadAll(const std::string& path) {
//   std::ifstream ifs(path, std::ios::binary);
//   if (!ifs) throw std::runtime_error("Failed to open: " + path);
//   return std::string((std::istreambuf_iterator<char>(ifs)),
//                      std::istreambuf_iterator<char>());
// }

// static int ArgMax(const float* data, int n) {
//   int idx = 0;
//   float best = data[0];
//   for (int i = 1; i < n; ++i)
//     if (data[i] > best) {
//       best = data[i];
//       idx = i;
//     }
//   return idx;
// }

// int main(int argc, char** argv) {
//   if (argc < 2) {
//     std::cerr << "Usage: " << argv[0] << " <model_dir>\n"
//               << "  model_dir contains resnet18_tvm.so/.json/.params\n";
//     return 1;
//   }
//   std::string model_dir = argv[1];
//   std::string so_path = model_dir + "/resnet18_tvm.so";
//   std::string graph_path = model_dir + "/resnet18_tvm.json";
//   std::string params_path = model_dir + "/resnet18_tvm.params";

//   try {
//     // Load module + graph + params
//     tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(so_path);
//     std::string graph_json = ReadAll(graph_path);
//     std::string params_blob = ReadAll(params_path);

//     // Device
//     DLDevice dev{kDLCPU, 0};

//     // Create executor
//     tvm::runtime::GraphExecutor gexec(graph_json, lib, dev);
//     gexec.LoadParams(params_blob);

//     // Allocate dummy input [1,3,224,224] float32 filled with 0
//     int64_t shape[4] = {1, 3, 224, 224};
//     tvm::runtime::NDArray input =
//         tvm::runtime::NDArray::Empty({shape[0], shape[1], shape[2],
//         shape[3]},
//                                      DLDataType{kDLFloat, 32, 1}, dev);
//     std::vector<float> zeros(1 * 3 * 224 * 224, 0.0f);
//     input.CopyFromBytes(zeros.data(), zeros.size() * sizeof(float));

//     // Try common input names
//     bool set_ok = false;
//     try {
//       gexec.SetInput("input0", input);
//       set_ok = true;
//     } catch (...) {
//     }
//     if (!set_ok) {
//       try {
//         gexec.SetInput("data", input);
//         set_ok = true;
//       } catch (...) {
//       }
//     }
//     if (!set_ok) {
//       std::cerr << "Could not set input. Try the real input name used in your
//       "
//                    "Relay graph.\n";
//       return 2;
//     }

//     // Run and fetch logits
//     gexec.Run();
//     tvm::runtime::NDArray out = gexec.GetOutput(0);

//     // Copy to host and argmax
//     int64_t n = 1;
//     for (int i = 0; i < out->ndim; ++i) n *= out.Shape()[i];
//     std::vector<float> logits(n);
//     out.CopyToBytes(logits.data(), logits.size() * sizeof(float));
//     int top1 = ArgMax(logits.data(), static_cast<int>(n));

//     std::cout << "Top-1 class id (with zero input): " << top1 << "\n";
//     return 0;
//   } catch (const std::exception& e) {
//     std::cerr << "Error: " << e.what() << "\n";
//     return 3;
//   }
// }

// ============================================================================

// #include <tvm/runtime/module.h>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>

// #include <fstream>
// #include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// static std::string ReadAll(const std::string& p) {
//   std::ifstream ifs(p, std::ios::binary);
//   if (!ifs) throw std::runtime_error("open fail: " + p);
//   return std::string((std::istreambuf_iterator<char>(ifs)),
//                      std::istreambuf_iterator<char>());
// }
// static int ArgMax(const float* d, int n) {
//   int i = 0;
//   float b = d[0];
//   for (int k = 1; k < n; ++k) {
//     if (d[k] > b) {
//       b = d[k];
//       i = k;
//     }
//   }
//   return i;
// }

// int main(int argc, char** argv) {
//   if (argc < 2) {
//     std::cerr << "Usage: " << argv[0] << " <model_dir>\n";
//     return 1;
//   }
//   std::string dir = argv[1];
//   try {
//     // 1) Load compiled module (.so)
//     tvm::runtime::Module lib =
//         tvm::runtime::Module::LoadFromFile(dir + "/resnet18_tvm.so");

//     // 2) Create graph executor from packed registry (no graph_executor.h
//     // needed)
//     std::string graph_json = ReadAll(dir + "/resnet18_tvm.json");
//     DLDevice dev{kDLCPU, 0};
//     auto fcreate = *tvm::runtime::Registry::Get("tvm.graph_executor.create");
//     tvm::runtime::Module gmod = fcreate(graph_json, lib, dev);

//     // 3) Load params
//     std::string params = ReadAll(dir + "/resnet18_tvm.params");
//     gmod.GetFunction("load_params")(tvm::runtime::String(params));

//     // 4) Build a dummy input (zeros) [1,3,224,224] float32 just to validate
//     run tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty(
//         {1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
//     std::vector<float> zeros(1 * 3 * 224 * 224, 0.f);
//     x.CopyFromBytes(zeros.data(), zeros.size() * sizeof(float));

//     // 5) Set input (try common names)
//     bool ok = false;
//     auto set_input = gmod.GetFunction("set_input");
//     try {
//       set_input("input0", x);
//       ok = true;
//     } catch (...) {
//     }
//     if (!ok) {
//       try {
//         set_input("data", x);
//         ok = true;
//       } catch (...) {
//       }
//     }
//     if (!ok) {
//       std::cerr << "Adjust input name ('input0'/'data').\n";
//       return 2;
//     }

//     // 6) Run & fetch output[0]
//     gmod.GetFunction("run")();
//     tvm::runtime::NDArray out;
//     gmod.GetFunction("get_output")(0, out);

//     int64_t n = 1;
//     for (int i = 0; i < out->ndim; ++i) n *= out.Shape()[i];
//     std::vector<float> logits(n);
//     out.CopyToBytes(logits.data(), n * sizeof(float));
//     int top1 = ArgMax(logits.data(), (int)n);
//     std::cout << "Top-1 id (zero input): " << top1 << "\n";
//     return 0;
//   } catch (const std::exception& e) {
//     std::cerr << "Error: " << e.what() << "\n";
//     return 3;
//   }
// }

// ==================================================================================

// #include <tvm/runtime/module.h>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>

// #include <fstream>
// #include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// static std::string ReadAll(const std::string& p) {
//   std::ifstream ifs(p, std::ios::binary);
//   if (!ifs) throw std::runtime_error("open fail: " + p);
//   return std::string((std::istreambuf_iterator<char>(ifs)),
//                      std::istreambuf_iterator<char>());
// }

// static bool LoadPPM224RGB(const std::string& path, std::vector<float>& chw) {
//   std::ifstream f(path, std::ios::binary);
//   if (!f) return false;

//   auto skip_comments = [&]() {
//     while (f.peek() == '#') {
//       std::string line;
//       std::getline(f, line);
//     }
//   };

//   std::string magic;
//   f >> magic;
//   if (magic != "P6") return false;
//   int w = 0, h = 0, maxv = 0;
//   skip_comments();
//   f >> w;
//   skip_comments();
//   f >> h;
//   skip_comments();
//   f >> maxv;
//   if (!f || w != 224 || h != 224 || maxv <= 0) return false;
//   f.get();  // consume single whitespace after header

//   std::vector<unsigned char> buf(224 * 224 * 3);
//   f.read(reinterpret_cast<char*>(buf.data()), buf.size());
//   if (!f) return false;

//   const float mean[3] = {0.485f, 0.456f, 0.406f};
//   const float stdv[3] = {0.229f, 0.224f, 0.225f};

//   chw.resize(1 * 3 * 224 * 224);
//   size_t cstride = 224 * 224;
//   for (int y = 0; y < 224; ++y) {
//     for (int x = 0; x < 224; ++x) {
//       size_t i = (y * 224 + x);
//       float r = buf[3 * i + 0] / 255.0f;
//       float g = buf[3 * i + 1] / 255.0f;
//       float b = buf[3 * i + 2] / 255.0f;
//       chw[0 * cstride + i] = (r - mean[0]) / stdv[0];
//       chw[1 * cstride + i] = (g - mean[1]) / stdv[1];
//       chw[2 * cstride + i] = (b - mean[2]) / stdv[2];
//     }
//   }
//   return true;
// }

// static int ArgMax(const float* d, int n) {
//   int idx = 0;
//   float best = d[0];
//   for (int i = 1; i < n; ++i)
//     if (d[i] > best) {
//       best = d[i];
//       idx = i;
//     }
//   return idx;
// }

// int main(int argc, char** argv) {
//   if (argc < 3) {
//     std::cerr << "Usage: " << argv[0] << " <artifacts_dir>
//     <image_224.ppm>\n"; return 1;
//   }
//   std::string dir = argv[1];
//   std::string img = argv[2];

//   try {
//     // Load module
//     tvm::runtime::Module lib =
//         tvm::runtime::Module::LoadFromFile(dir + "/resnet18_tvm.so");

//     // Create graph executor via packed API (no graph_executor.h needed)
//     std::string graph_json = ReadAll(dir + "/resnet18_tvm.json");
//     DLDevice dev{kDLCPU, 0};
//     auto fcreate = *tvm::runtime::Registry::Get("tvm.graph_executor.create");
//     tvm::runtime::Module gmod = fcreate(graph_json, lib, dev);

//     // Load params
//     std::string params = ReadAll(dir + "/resnet18_tvm.params");
//     gmod.GetFunction("load_params")(tvm::runtime::String(params));

//     // Load image → NCHW float32 normalized
//     std::vector<float> chw;
//     if (!LoadPPM224RGB(img, chw))
//       throw std::runtime_error("Failed to load 224x224 RGB PPM: " + img);

//     tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(
//         {1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
//     input.CopyFromBytes(chw.data(), chw.size() * sizeof(float));

//     // Set input (try common names)
//     bool ok = false;
//     auto set_input = gmod.GetFunction("set_input");
//     try {
//       set_input("input0", input);
//       ok = true;
//     } catch (...) {
//     }
//     if (!ok) {
//       try {
//         set_input("data", input);
//         ok = true;
//       } catch (...) {
//       }
//     }
//     if (!ok)
//       throw std::runtime_error("Unknown input name (try 'input0' or
//       'data').");

//     // Run
//     gmod.GetFunction("run")();

//     // Output[0]
//     tvm::runtime::NDArray out;
//     gmod.GetFunction("get_output")(0, out);

//     int64_t n = 1;
//     for (int i = 0; i < out->ndim; ++i) n *= out.Shape()[i];
//     std::vector<float> logits(n);
//     out.CopyToBytes(logits.data(), n * sizeof(float));
//     int top1 = ArgMax(logits.data(), (int)n);

//     bool is_cat = (top1 >= 281 && top1 <= 285);
//     std::cout << "Top-1 id: " << top1 << "\n"
//               << (is_cat ? "Prediction: CAT 🐱\n" : "Prediction: NOT CAT\n");
//     return 0;

//   } catch (const std::exception& e) {
//     std::cerr << "Error: " << e.what() << "\n";
//     return 3;
//   }
// }

// =============================================================================

// #include <tvm/runtime/module.h>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>

// #include <cmath>
// #include <fstream>
// #include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// // ---- stb_image (header-only) ----
// #define STB_IMAGE_IMPLEMENTATION
// #define STBI_NO_HDR
// #define STBI_NO_GIF
// #define STBI_NO_PSD
// #define STBI_NO_PIC
// #define STBI_NO_PNM
// #include "stb_image.h"

// // ---- utils ----
// static std::string ReadAll(const std::string& p) {
//   std::ifstream ifs(p, std::ios::binary);
//   if (!ifs) throw std::runtime_error("open fail: " + p);
//   return std::string((std::istreambuf_iterator<char>(ifs)),
//                      std::istreambuf_iterator<char>());
// }

// static void ResizeBilinearRGB8(const unsigned char* src, int sw, int sh,
//                                unsigned char* dst, int dw, int dh) {
//   // src/dst are RGB, tightly packed
//   const float x_scale = static_cast<float>(sw) / dw;
//   const float y_scale = static_cast<float>(sh) / dh;
//   for (int y = 0; y < dh; ++y) {
//     float sy = (y + 0.5f) * y_scale - 0.5f;
//     int y0 = static_cast<int>(std::floor(sy));
//     int y1 = y0 + 1;
//     float wy1 = sy - y0;
//     float wy0 = 1.0f - wy1;
//     if (y0 < 0) {
//       y0 = 0;
//       wy0 = 1.0f;
//       wy1 = 0.0f;
//     }
//     if (y1 >= sh) {
//       y1 = sh - 1;
//       wy1 = 1.0f - wy0;
//     }
//     for (int x = 0; x < dw; ++x) {
//       float sx = (x + 0.5f) * x_scale - 0.5f;
//       int x0 = static_cast<int>(std::floor(sx));
//       int x1 = x0 + 1;
//       float wx1 = sx - x0;
//       float wx0 = 1.0f - wx1;
//       if (x0 < 0) {
//         x0 = 0;
//         wx0 = 1.0f;
//         wx1 = 0.0f;
//       }
//       if (x1 >= sw) {
//         x1 = sw - 1;
//         wx1 = 1.0f - wx0;
//       }

//       const unsigned char* p00 = src + (y0 * sw + x0) * 3;
//       const unsigned char* p01 = src + (y0 * sw + x1) * 3;
//       const unsigned char* p10 = src + (y1 * sw + x0) * 3;
//       const unsigned char* p11 = src + (y1 * sw + x1) * 3;

//       for (int c = 0; c < 3; ++c) {
//         float v = wx0 * (wy0 * p00[c] + wy1 * p10[c]) +
//                   wx1 * (wy0 * p01[c] + wy1 * p11[c]);
//         dst[(y * dw + x) * 3 + c] = static_cast<unsigned
//         char>(std::round(v));
//       }
//     }
//   }
// }

// static int ArgMax(const float* d, int n) {
//   int idx = 0;
//   float best = d[0];
//   for (int i = 1; i < n; ++i)
//     if (d[i] > best) {
//       best = d[i];
//       idx = i;
//     }
//   return idx;
// }

// int main(int argc, char** argv) {
//   if (argc < 3) {
//     std::cerr << "Usage: " << argv[0] << " <artifacts_dir>
//     <image.(jpg|png)>\n"; return 1;
//   }
//   std::string dir = argv[1];
//   std::string img = argv[2];

//   try {
//     // 1) Load compiled module
//     tvm::runtime::Module lib =
//         tvm::runtime::Module::LoadFromFile(dir + "/resnet18_tvm.so");

//     // 2) Create graph executor via packed registry (no graph_executor.h
//     needed) std::string graph_json = ReadAll(dir + "/resnet18_tvm.json");
//     DLDevice dev{kDLCPU, 0};
//     auto fcreate = *tvm::runtime::Registry::Get("tvm.graph_executor.create");
//     // tvm::runtime::Module gmod = fcreate(graph_json, lib, dev);
//     tvm::runtime::Module gmod = fcreate(graph_json, lib, (int)kDLCPU, 0);

//     // 3) Load params
//     std::string params = ReadAll(dir + "/resnet18_tvm.params");
//     gmod.GetFunction("load_params")(tvm::runtime::String(params));

//     // 4) Load image using stb_image -> RGB
//     int w = 0, h = 0, channels = 0;
//     unsigned char* raw = stbi_load(img.c_str(), &w, &h, &channels, 3);
//     if (!raw)
//       throw std::runtime_error(std::string("stbi_load failed: ") +
//                                stbi_failure_reason());

//     // 5) Resize to 224x224
//     std::vector<unsigned char> rgb224(224 * 224 * 3);
//     ResizeBilinearRGB8(raw, w, h, rgb224.data(), 224, 224);
//     stbi_image_free(raw);

//     // 6) Normalize (ImageNet) and convert to NCHW float32
//     const float mean[3] = {0.485f, 0.456f, 0.406f};
//     const float stdv[3] = {0.229f, 0.224f, 0.225f};
//     std::vector<float> chw(1 * 3 * 224 * 224);
//     size_t cstride = 224 * 224;
//     for (int y = 0; y < 224; ++y) {
//       for (int x = 0; x < 224; ++x) {
//         size_t i = y * 224 + x;
//         float r = rgb224[3 * i + 0] / 255.0f;
//         float g = rgb224[3 * i + 1] / 255.0f;
//         float b = rgb224[3 * i + 2] / 255.0f;
//         // Assuming model expects RGB order; swap if your export used BGR.
//         chw[0 * cstride + i] = (r - mean[0]) / stdv[0];
//         chw[1 * cstride + i] = (g - mean[1]) / stdv[1];
//         chw[2 * cstride + i] = (b - mean[2]) / stdv[2];
//       }
//     }

//     // 7) Create TVM NDArray input
//     tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(
//         {1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
//     input.CopyFromBytes(chw.data(), chw.size() * sizeof(float));

//     // Set input (common names)
//     auto set_input = gmod.GetFunction("set_input");
//     bool ok = false;
//     try {
//       set_input("input0", input);
//       ok = true;
//     } catch (...) {
//     }
//     if (!ok) {
//       try {
//         set_input("data", input);
//         ok = true;
//       } catch (...) {
//       }
//     }
//     if (!ok)
//       throw std::runtime_error("Unknown input name (try 'input0' or
//       'data').");

//     // 8) Run & fetch logits
//     gmod.GetFunction("run")();
//     tvm::runtime::NDArray out;
//     gmod.GetFunction("get_output")(0, out);

//     int64_t n = 1;
//     for (int i = 0; i < out->ndim; ++i) n *= out.Shape()[i];
//     std::vector<float> logits(n);
//     out.CopyToBytes(logits.data(), n * sizeof(float));
//     int top1 = ArgMax(logits.data(), (int)n);

//     bool is_cat = (top1 >= 281 && top1 <= 285);
//     std::cout << "Top-1 id: " << top1 << "\n"
//               << (is_cat ? "Prediction: CAT 🐱\n" : "Prediction: NOT CAT\n");
//     return 0;

//   } catch (const std::exception& e) {
//     std::cerr << "Error: " << e.what() << "\n";
//     return 3;
//   }
// }

// run_resnet_cat_stb.cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"

// ---------- helpers ----------
static std::string ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("open fail: " + p);
  return std::string((std::istreambuf_iterator<char>(f)), {});
}

static void ResizeBilinearRGB8(const unsigned char* src, int sw, int sh,
                               unsigned char* dst, int dw, int dh) {
  float xs = float(sw) / dw, ys = float(sh) / dh;
  for (int y = 0; y < dh; ++y) {
    float sy = (y + 0.5f) * ys - 0.5f;
    int y0 = (int)std::floor(sy), y1 = y0 + 1;
    float wy1 = sy - y0, wy0 = 1.f - wy1;
    y0 = std::max(0, std::min(sh - 1, y0));
    y1 = std::max(0, std::min(sh - 1, y1));
    for (int x = 0; x < dw; ++x) {
      float sx = (x + 0.5f) * xs - 0.5f;
      int x0 = (int)std::floor(sx), x1 = x0 + 1;
      float wx1 = sx - x0, wx0 = 1.f - wx1;
      x0 = std::max(0, std::min(sw - 1, x0));
      x1 = std::max(0, std::min(sw - 1, x1));
      const unsigned char* p00 = src + (y0 * sw + x0) * 3;
      const unsigned char* p01 = src + (y0 * sw + x1) * 3;
      const unsigned char* p10 = src + (y1 * sw + x0) * 3;
      const unsigned char* p11 = src + (y1 * sw + x1) * 3;
      for (int c = 0; c < 3; ++c) {
        float v = wx0 * (wy0 * p00[c] + wy1 * p10[c]) +
                  wx1 * (wy0 * p01[c] + wy1 * p11[c]);
        dst[(y * dw + x) * 3 + c] = (unsigned char)std::lround(v);
      }
    }
  }
}

static std::vector<std::string> LoadLabels(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};  // ok: we’ll fall back to ids only
  std::vector<std::string> L;
  std::string s;
  while (std::getline(f, s))
    if (!s.empty()) L.push_back(s);
  return L;
}

static std::vector<float> Softmax(const std::vector<float>& x) {
  std::vector<float> p(x.size());
  float m = *std::max_element(x.begin(), x.end());
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    p[i] = std::exp(x[i] - m);
    sum += p[i];
  }
  for (auto& v : p) v = float(v / sum);
  return p;
}

static std::vector<int> TopK(const std::vector<float>& v, int k) {
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  if (k > (int)idx.size()) k = (int)idx.size();
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return v[a] > v[b]; });
  idx.resize(k);
  return idx;
}

// ---------- main ----------
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <artifacts_dir> <image.(jpg|png)> [labels.txt]\n";
    return 1;
  }
  std::string dir = argv[1], img = argv[2];
  std::string labels_path = (argc >= 4) ? argv[3] : "labels.txt";

  try {
    // 1) Load module
    tvm::runtime::Module lib =
        tvm::runtime::Module::LoadFromFile(dir + "/resnet18_tvm.so");

    // 2) Create graph executor (4 args: json, lib, dev_type, dev_id)
    std::string graph_json = ReadAll(dir + "/resnet18_tvm.json");
    auto fcreate = tvm::runtime::Registry::Get("tvm.graph_executor.create");
    if (!fcreate)
      throw std::runtime_error(
          "tvm.graph_executor.create not found (build TVM with "
          "USE_GRAPH_EXECUTOR=ON)");
    tvm::runtime::Module gmod = (*fcreate)(graph_json, lib, (int)kDLCPU, 0);

    // 3) Load params
    std::string params = ReadAll(dir + "/resnet18_tvm.params");
    gmod.GetFunction("load_params")(tvm::runtime::String(params));

    // 4) Load + resize + normalize
    int w = 0, h = 0, c = 0;
    unsigned char* raw = stbi_load(img.c_str(), &w, &h, &c, 3);
    if (!raw)
      throw std::runtime_error(std::string("stbi_load failed: ") +
                               stbi_failure_reason());
    std::vector<unsigned char> rgb224(224 * 224 * 3);
    ResizeBilinearRGB8(raw, w, h, rgb224.data(), 224, 224);
    stbi_image_free(raw);

    const float mean[3] = {0.485f, 0.456f, 0.406f},
                stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(1 * 3 * 224 * 224);
    size_t cs = 224 * 224;
    for (int y = 0; y < 224; ++y) {
      for (int x = 0; x < 224; ++x) {
        size_t i = y * 224 + x;
        float r = rgb224[3 * i + 0] / 255.f, g = rgb224[3 * i + 1] / 255.f,
              b = rgb224[3 * i + 2] / 255.f;
        chw[0 * cs + i] = (r - mean[0]) / stdv[0];
        chw[1 * cs + i] = (g - mean[1]) / stdv[1];
        chw[2 * cs + i] = (b - mean[2]) / stdv[2];
      }
    }

    // 5) Create input on CPU(0)
    DLDevice dev{kDLCPU, 0};
    tvm::runtime::NDArray input =
        tvm::runtime::NDArray::Empty({1, 3, 224, 224}, {kDLFloat, 32, 1}, dev);
    input.CopyFromBytes(chw.data(), chw.size() * sizeof(float));

    // 6) Set input
    auto set_input = gmod.GetFunction("set_input");
    bool set = false;
    try {
      set_input("input0", input);
      set = true;
    } catch (...) {
    }
    if (!set) {
      try {
        set_input("data", input);
        set = true;
      } catch (...) {
      }
    }
    if (!set)
      throw std::runtime_error("Unknown input name (try 'input0' or 'data').");

    // 7) Run
    gmod.GetFunction("run")();

    // 8) Get output (returns NDArray)
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::NDArray out = get_output(0);

    // 9) Copy logits
    int64_t n = 1;
    for (int i = 0; i < out->ndim; ++i) n *= out.Shape()[i];
    std::vector<float> logits(n);
    out.CopyToBytes(logits.data(), n * sizeof(float));

    // --- NEW: labels + top-5 ---
    auto probs = Softmax(logits);
    auto labels = LoadLabels(labels_path);  // optional file; empty if not found
    auto top = TopK(probs, 5);

    std::cout << "Top-5:\n";
    for (int i = 0; i < (int)top.size(); ++i) {
      int id = top[i];
      const char* name =
          (id < (int)labels.size()) ? labels[id].c_str() : "<no-label>";
      std::cout << "  " << (i + 1) << ") id=" << id << "  p=" << probs[id]
                << "  " << name << "\n";
    }

    int top1 = top[0];
    bool is_cat = (top1 >= 281 && top1 <= 285);
    std::cout << (is_cat ? "Prediction: CAT 🐱\n" : "Prediction: NOT CAT\n");
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 3;
  }
}
