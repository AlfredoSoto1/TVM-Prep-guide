// #include <fstream>
// #include <iostream>
// #include <numeric>
// #include <vector>

// // Include TVM runtime headers
// #include <tvm/runtime/c_runtime_api.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/registry.h>

// void RunTVMModel(const std::string &model_path, const std::string &lib_path,
//                  const std::string &params_path, int device_type, int device_id)
// {
//     // 1. Load the TVM graph, library, and parameters

//     // Load graph JSON
//     std::ifstream graph_file(model_path);
//     std::string graph_json((std::istreambuf_iterator<char>(graph_file)),
//                            std::istreambuf_iterator<char>());

//     // Load compiled library
//     tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);

//     // Load parameters
//     std::ifstream params_file(params_path, std::ios::binary);
//     std::string params_bytes((std::istreambuf_iterator<char>(params_file)),
//                              std::istreambuf_iterator<char>());

//     tvm::runtime::TVMByteArray params_arr;
//     params_arr.data = params_bytes.c_str();
//     params_arr.size = params_bytes.length();

//     // 2. Create TVM runtime module
//     tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
//         graph_json, lib, device_type, device_id);

//     // Set parameters
//     mod.GetFunction("load_params")(params_arr);

//     // 3. Prepare Input Data (Example: a simple 1x3x224x224 float32 tensor)
//     std::vector<float> input_data(1 * 3 * 224 * 224);
//     // Fill with some dummy data (e.g., random values or all zeros)
//     std::iota(input_data.begin(), input_data.end(), 0.0f); // Example: fill with increasing floats

//     // Input shape
//     std::vector<int64_t> input_shape = {1, 3, 224, 224};
//     // Create DLTensor for input
//     DLTensor input_tensor;
//     input_tensor.data = input_data.data();
//     input_tensor.device = {static_cast<DLDeviceType>(device_type), device_id};
//     input_tensor.ndim = input_shape.size();
//     input_tensor.dtype = {kDLFloat, 32, 1};
//     input_tensor.shape = input_shape.data();
//     input_tensor.strides = nullptr; // Let TVM infer
//     input_tensor.byte_offset = 0;

//     // 4. Set Input
//     // Assuming your model has an input named "input" (common for ONNX)
//     mod.GetFunction("set_input")("input", &input_tensor);

//     // 5. Run Inference
//     mod.GetFunction("run")();

//     // 6. Get Output (Example: a single output tensor)
//     DLTensor output_tensor;
//     // Get number of outputs if unknown
//     int num_outputs = mod.GetFunction("get_num_outputs")();
//     if (num_outputs == 0)
//     {
//         std::cerr << "Error: Model has no outputs." << std::endl;
//         return;
//     }
//     // Assuming we want the first output
//     mod.GetFunction("get_output")(0, &output_tensor);

//     // Now, output_tensor contains the result.
//     // You can copy it to a CPU buffer if it's on a different device.
//     // For demonstration, let's assume it's on CPU or copy it to CPU.

//     // Calculate output size
//     int64_t output_size = 1;
//     for (int i = 0; i < output_tensor.ndim; ++i)
//     {
//         output_size *= output_tensor.shape[i];
//     }
//     std::vector<float> output_data(output_size);

//     // If output is on GPU/other device, copy it back to CPU
//     if (output_tensor.device.device_type != kDLCPU)
//     {
//         DLTensor cpu_output_tensor = output_tensor; // Create a DLTensor pointing to CPU memory
//         cpu_output_tensor.data = output_data.data();
//         cpu_output_tensor.device = {kDLCPU, 0};
//         TVMArrayCopyFromTo(&output_tensor, &cpu_output_tensor, nullptr);
//     }
//     else
//     {
//         // If already on CPU, just cast and copy
//         memcpy(output_data.data(), output_tensor.data, output_size * sizeof(float));
//     }

//     // Print a few output values (for demonstration)
//     std::cout << "Inference successful! First 10 output values: ";
//     for (int i = 0; i < std::min((int)output_data.size(), 10); ++i)
//     {
//         std::cout << output_data[i] << " ";
//     }
//     std::cout << std::endl;
// }

// int main()
// {
//     // These paths should point to your compiled TVM artifacts
//     std::string model_json_path = "model.json";     // Path to graph JSON
//     std::string model_lib_path = "model.so";        // Path to compiled shared library
//     std::string model_params_path = "model.params"; // Path to parameters

//     // Device configuration
//     // kDLCPU = 1, kDLGPU = 2, kDLVulkan = 10, etc.
//     // See include/tvm/runtime/c_runtime_api.h for DLDeviceType
//     int device_type = kDLCPU; // Or kDLGPU, kDLVulkan, etc.
//     int device_id = 0;

//     RunTVMModel(model_json_path, model_lib_path, model_params_path, device_type, device_id);

//     return 0;
// }

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// STB for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ---------- Helpers ----------
static void ResizeBilinearRGB8(const unsigned char* src, int sw, int sh,
                               unsigned char* dst, int dw, int dh) {
    float xs = float(sw) / dw, ys = float(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        float sy = (y + 0.5f) * ys - 0.5f;
        int y0 = std::max(0, std::min(sh-1, (int)std::floor(sy)));
        int y1 = std::max(0, std::min(sh-1, y0+1));
        float wy1 = sy - y0, wy0 = 1.f - wy1;
        for (int x = 0; x < dw; ++x) {
            float sx = (x + 0.5f) * xs - 0.5f;
            int x0 = std::max(0, std::min(sw-1, (int)std::floor(sx)));
            int x1 = std::max(0, std::min(sw-1, x0+1));
            float wx1 = sx - x0, wx0 = 1.f - wx1;
            for (int c = 0; c < 3; ++c) {
                float v = wx0 * (wy0*src[(y0*sw+x0)*3 + c] + wy1*src[(y1*sw+x0)*3 + c])
                        + wx1 * (wy0*src[(y0*sw+x1)*3 + c] + wy1*src[(y1*sw+x1)*3 + c]);
                dst[(y*dw + x)*3 + c] = (unsigned char)std::round(v);
            }
        }
    }
}

static std::vector<float> Softmax(const std::vector<float>& x) {
    std::vector<float> p(x.size());
    float m = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i=0;i<x.size();i++) { p[i] = std::exp(x[i]-m); sum += p[i]; }
    for (auto& v: p) v = float(v / sum);
    return p;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.so> <image.jpg>\n";
        return 1;
    }

    std::string so_path = argv[1];
    std::string img_path = argv[2];

    try {
        // 1) Load compiled module (.so)
        tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(so_path);

        // 2) Create graph executor
        auto fcreate = tvm::runtime::Registry::Get("tvm.graph_executor.create");
        if (!fcreate) throw std::runtime_error("graph_executor.create not found");
        tvm::runtime::Module gmod = (*fcreate)("", mod, 1, 0); // device_type=CPU, dev_id=0

        // 3) Load image
        int w,h,c;
        unsigned char* raw = stbi_load(img_path.c_str(), &w, &h, &c, 3);
        if (!raw) throw std::runtime_error("Failed to load image");
        std::vector<unsigned char> rgb224(224*224*3);
        ResizeBilinearRGB8(raw, w, h, rgb224.data(), 224, 224);
        stbi_image_free(raw);

        // 4) Normalize and create input NDArray
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};
        std::vector<float> chw(1*3*224*224);
        for (int y=0;y<224;y++)
            for (int x=0;x<224;x++) {
                size_t i = y*224+x;
                chw[0*224*224 + i] = (rgb224[i*3+0]/255.f - mean[0])/stdv[0];
                chw[1*224*224 + i] = (rgb224[i*3+1]/255.f - mean[1])/stdv[1];
                chw[2*224*224 + i] = (rgb224[i*3+2]/255.f - mean[2])/stdv[2];
            }

        tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({1,3,224,224},{kDLFloat,32,1},{kDLCPU,0});
        input.CopyFromBytes(chw.data(), chw.size()*sizeof(float));

        // 5) Set input and run
        gmod.GetFunction("set_input")("input0", input);
        gmod.GetFunction("run")();

        // 6) Get output
        tvm::runtime::NDArray out = gmod.GetFunction("get_output")(0);
        int64_t n = 1; for (int i=0;i<out->ndim;i++) n*=out.Shape()[i];
        std::vector<float> logits(n);
        out.CopyToBytes(logits.data(), n*sizeof(float));

        // 7) Softmax + top-5
        auto probs = Softmax(logits);
        std::cout << "Top-5 probabilities:\n";
        for (int i=0;i<5;i++) std::cout << "  " << i+1 << ": " << probs[i] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
