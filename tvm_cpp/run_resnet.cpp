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
