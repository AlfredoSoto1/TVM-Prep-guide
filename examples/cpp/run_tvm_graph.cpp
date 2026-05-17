#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#define STB_IMAGE_IMPLEMENTATION
#include "tvm_cpp/stb_image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
  std::string artifact_dir;
  std::string image_path;
  std::string labels_path;
  std::string input_name = "input0";
  std::string library = "model.so";
  bool normalize = true;
};

struct Prediction {
  int id;
  float prob;
  std::string label;
};

std::string ReadAll(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)), {});
}

std::vector<std::string> LoadLabels(const std::string& path) {
  std::ifstream file(path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    labels.push_back(line);
  }
  return labels;
}

std::string MetadataValue(const std::string& metadata, const std::string& key, const std::string& fallback) {
  const std::string needle = "\"" + key + "\":";
  size_t pos = metadata.find(needle);
  if (pos == std::string::npos) {
    return fallback;
  }
  pos = metadata.find('"', pos + needle.size());
  if (pos == std::string::npos) {
    return fallback;
  }
  size_t end = metadata.find('"', pos + 1);
  if (end == std::string::npos) {
    return fallback;
  }
  return metadata.substr(pos + 1, end - pos - 1);
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--artifact-dir") {
      args.artifact_dir = require_value(arg);
    } else if (arg == "--image") {
      args.image_path = require_value(arg);
    } else if (arg == "--labels") {
      args.labels_path = require_value(arg);
    } else if (arg == "--input-name") {
      args.input_name = require_value(arg);
    } else if (arg == "--library") {
      args.library = require_value(arg);
    } else if (arg == "--no-normalize") {
      args.normalize = false;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (args.artifact_dir.empty() || args.image_path.empty()) {
    throw std::runtime_error(
        "Usage: run_tvm_graph --artifact-dir DIR --image IMAGE [--labels labels.txt] "
        "[--input-name input0] [--library model.so] [--no-normalize]");
  }
  return args;
}

std::vector<float> LoadImageNCHW(const std::string& path, bool normalize) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* image = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + path);
  }

  int new_width = 0;
  int new_height = 0;
  if (width < height) {
    new_width = 256;
    new_height = static_cast<int>(height * 256.0 / width);
  } else {
    new_height = 256;
    new_width = static_cast<int>(width * 256.0 / height);
  }

  std::vector<unsigned char> resized(new_width * new_height * 3);
  for (int y = 0; y < new_height; ++y) {
    int src_y = y * height / new_height;
    for (int x = 0; x < new_width; ++x) {
      int src_x = x * width / new_width;
      for (int c = 0; c < 3; ++c) {
        resized[(y * new_width + x) * 3 + c] = image[(src_y * width + src_x) * 3 + c];
      }
    }
  }
  stbi_image_free(image);

  constexpr int H = 224;
  constexpr int W = 224;
  int start_x = (new_width - W) / 2;
  int start_y = (new_height - H) / 2;
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stddev[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> output(3 * H * W);

  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float value = resized[((y + start_y) * new_width + (x + start_x)) * 3 + c];
        if (normalize) {
          value = (value / 255.0f - mean[c]) / stddev[c];
        }
        output[c * H * W + y * W + x] = value;
      }
    }
  }
  return output;
}

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    const std::string metadata_path = args.artifact_dir + "/metadata.json";
    std::ifstream metadata_file(metadata_path);
    if (metadata_file) {
      const std::string metadata((std::istreambuf_iterator<char>(metadata_file)), {});
      args.input_name = MetadataValue(metadata, "input_name", args.input_name);
      args.library = MetadataValue(metadata, "library", args.library);
      if (args.labels_path.empty()) {
        std::string labels = MetadataValue(metadata, "labels", "");
        if (!labels.empty()) {
          args.labels_path = args.artifact_dir + "/" + labels;
        }
      }
    }

    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(args.artifact_dir + "/" + args.library);
    std::string graph_json = ReadAll(args.artifact_dir + "/model.json");
    std::string params = ReadAll(args.artifact_dir + "/model.params");

    auto fcreate = tvm::runtime::Registry::Get("tvm.graph_executor.create");
    if (!fcreate) {
      throw std::runtime_error("tvm.graph_executor.create not found");
    }
    tvm::runtime::Module graph = (*fcreate)(graph_json, lib, static_cast<int>(kDLCPU), 0);
    graph.GetFunction("load_params")(tvm::runtime::String(params));

    std::vector<float> input = LoadImageNCHW(args.image_path, args.normalize);
    tvm::runtime::NDArray input_nd = tvm::runtime::NDArray::Empty(
        {1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
    input_nd.CopyFromBytes(input.data(), input.size() * sizeof(float));

    graph.GetFunction("set_input")(args.input_name, input_nd);
    graph.GetFunction("run")();
    tvm::runtime::NDArray output_nd = graph.GetFunction("get_output")(0);

    int output_size = 1;
    for (int i = 0; i < output_nd->ndim; ++i) {
      output_size *= static_cast<int>(output_nd->shape[i]);
    }
    std::vector<float> logits(output_size);
    output_nd.CopyToBytes(logits.data(), logits.size() * sizeof(float));

    float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
      probs[i] = std::exp(logits[i] - max_logit);
      sum += probs[i];
    }
    for (float& prob : probs) {
      prob = static_cast<float>(prob / sum);
    }

    std::vector<std::string> labels;
    if (!args.labels_path.empty()) {
      labels = LoadLabels(args.labels_path);
    }
    std::vector<Prediction> predictions;
    for (int i = 0; i < output_size; ++i) {
      std::string label = i < static_cast<int>(labels.size()) ? labels[i] : "<no-label>";
      predictions.push_back({i, probs[i], label});
    }
    std::sort(predictions.begin(), predictions.end(), [](const Prediction& a, const Prediction& b) {
      return a.prob > b.prob;
    });

    for (int i = 0; i < 5 && i < static_cast<int>(predictions.size()); ++i) {
      std::cout << i + 1 << ": id=" << predictions[i].id << " p=" << predictions[i].prob << " "
                << predictions[i].label << "\n";
    }
  } catch (const std::exception& exc) {
    std::cerr << "Error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
