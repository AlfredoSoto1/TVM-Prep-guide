#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h> // Includes kDLCPU

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Simple image loading library

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <stdexcept> // For std::runtime_error

// --- Helper Struct and Functions ---

struct Prediction {
    int id;
    float prob;
    std::string label;
};

// Function to load ImageNet labels from a text file.
std::vector<std::string> LoadLabels(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);
    return labels;
}

// Function to read an entire file into a string (used for graph JSON and params).
std::string ReadAll(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    // Read stream to string using iterators
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

// --- Main Execution Function ---

int main(int argc, char** argv) {
    // Check for minimum required arguments
    if (argc < 5) {
        std::cerr << "Usage: ./run_model <model_name> <model_dir> <image_path> <labels.txt> [input_name] [--normalize] [--bgr] [--mean MEAN MEAN MEAN] [--std STD STD STD]\n";
        std::cerr << "Examples:\n";
        std::cerr << "  (PyTorch/ImageNet) ./run_model resnet18 ./artifacts/ResNet18 ./images/cat.png labels.txt input0 --normalize\n";
        std::cerr << "  (Caffe-style)      ./run_model squeezenet ./artifacts/SqueezeNet ./images/cat.png labels.txt data --bgr --no-normalize --mean 104 117 123 --std 1 1 1\n";
        std::cerr << "  (Simple 0-1)       ./run_model custom ./artifacts/Custom ./images/cat.png labels.txt input0 --no-normalize\n";
        return -1;
    }

    // Assign required positional arguments
    std::string model_name = argv[1];
    std::string model_dir  = argv[2];
    std::string image_path = argv[3];
    std::string labels_path = argv[4];
    
    // --- Argument Parsing and Preprocessing Setup ---

    std::string input_name = "input0";   // Default input name for TVM graph
    bool normalize = false; // Controls 0-1 scaling (PyTorch-style)
    bool bgr = false;       // Controls RGB vs BGR channel order
    float mean[3] = {0.0f, 0.0f, 0.0f};   // Channel-wise mean subtraction values (always in R,G,B order)
    float std_dev[3] = {1.0f, 1.0f, 1.0f}; // Channel-wise standard deviation (always in R,G,B order)

    // Pre-pass: Check for --normalize to set ImageNet defaults (PyTorch standard)
    for (int i = 5; i < argc; i++) {
        if (std::string(argv[i]) == "--normalize") {
            normalize = true;
            mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f; // R, G, B
            std_dev[0] = 0.229f; std_dev[1] = 0.224f; std_dev[2] = 0.225f; // R, G, B
            break; 
        }
    }

    // Main pass: Parse all flags and allow custom values to override defaults/presets
    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--normalize") {
            continue; 
        } else if (arg == "--no-normalize") {
            normalize = false;
            // Reset mean/std to no-op for Caffe-style (0-255 scaling is implicit)
            mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
            std_dev[0] = 1.0f; std_dev[1] = 1.0f; std_dev[2] = 1.0f;
        } else if (arg == "--bgr") {
            bgr = true; // Use BGR channel order in the final tensor
        } else if (arg == "--rgb") {
            bgr = false; // Use RGB channel order (default)
        } else if (arg == "--mean" && i + 3 < argc) {
            mean[0] = std::stof(argv[++i]);
            mean[1] = std::stof(argv[++i]);
            mean[2] = std::stof(argv[++i]);
        } else if (arg == "--std" && i + 3 < argc) {
            std_dev[0] = std::stof(argv[++i]);
            std_dev[1] = std::stof(argv[++i]);
            std_dev[2] = std::stof(argv[++i]);
        } else if (arg.find("--") != 0) {
            // If it's not a flag, assume it's the custom input name
            input_name = arg;
        }
    }

    // --- TVM Runtime Initialization ---

    try {
        // Load compiled model library (.so file)
        std::string lib_path = model_dir + "/" + model_name + "_tvm.so";
        tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);

        // Load graph JSON
        std::string json_path = model_dir + "/" + model_name + "_tvm.json";
        std::string graph_json = ReadAll(json_path);

        // --- FIX: Robust Graph Executor Creation (Addresses all-zero output bug) ---
        
        // 1. Explicitly ensure the LLVM runtime is loaded.
        // This is necessary to correctly link the graph to the compiled CPU code.
        const auto* f_load_target_ptr = tvm::runtime::Registry::Get("runtime.SystemLib");
        if (!f_load_target_ptr) {
            throw std::runtime_error("runtime.SystemLib not found. TVM runtime may not be correctly linked.");
        }
        tvm::runtime::Module graph_runtime_module = (*f_load_target_ptr)();

        // 2. Create the graph executor
        auto fcreate = tvm::runtime::Registry::Get("tvm.graph_executor.create");
        if (!fcreate) throw std::runtime_error("tvm.graph_executor.create not found");
        
        // Create graph executor (gmod) using the JSON, the loaded library (lib), CPU device, and device ID 0.
        tvm::runtime::Module gmod = (*fcreate)(graph_json, lib, (int)kDLCPU, 0);

        // --- END FIX ---
        
        // Load parameters (.params file)
        std::string params_path = model_dir + "/" + model_name + "_tvm.params";
        std::string params = ReadAll(params_path);
        gmod.GetFunction("load_params")(tvm::runtime::String(params));

        // Try to get input names from module (for models that include this metadata)
        try {
            auto get_input_names = gmod.GetFunction("get_input_names");
            if (get_input_names != nullptr) {
                tvm::runtime::Array<tvm::runtime::String> names = get_input_names();
                if (names.size() > 0) {
                    input_name = names[0];
                    std::cout << "Detected input name: " << input_name << "\n";
                }
            }
        } catch (...) {
            std::cout << "Using provided input name: " << input_name << "\n";
        }

        // --- Image Loading and Preprocessing ---

        // Load image (stb_image returns RGB, interleaved: HWC)
        int w, h, c;
        unsigned char* img = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!img) { std::cerr << "Failed to load image\n"; return -1; }

        std::cout << "Original image: " << w << "x" << h << "x" << c << "\n";

        // Step 1: Resize (Scale shortest side to 256)
        int new_w, new_h;
        if (w < h) { 
            new_w = 256; 
            new_h = static_cast<int>(h * 256.0 / w); 
        } else { 
            new_h = 256; 
            new_w = static_cast<int>(w * 256.0 / h); 
        }
        std::cout << "Resized to: " << new_w << "x" << new_h << "\n";
        // ... (resize implementation using nearest neighbor) ...
        unsigned char* resized = new unsigned char[new_w * new_h * 3];
        for (int y = 0; y < new_h; y++) {
            int in_y = y * h / new_h;
            for (int x = 0; x < new_w; x++) {
                int in_x = x * w / new_w;
                for (int ch = 0; ch < 3; ch++) {
                    resized[y * new_w * 3 + x * 3 + ch] = img[in_y * w * 3 + in_x * 3 + ch];
                }
            }
        }
        stbi_image_free(img);

        // Step 2: Center crop (224x224)
        const int H = 224, W = 224;
        int start_x = (new_w - W) / 2;
        int start_y = (new_h - H) / 2;
        
        if (start_x < 0 || start_y < 0 || start_x + W > new_w || start_y + H > new_h) {
            std::cerr << "Invalid crop dimensions\n";
            delete[] resized;
            return -1;
        }

        unsigned char* img_cropped = new unsigned char[H * W * 3];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                for (int ch = 0; ch < 3; ch++) {
                    img_cropped[y * W * 3 + x * 3 + ch] =
                        resized[(y + start_y) * new_w * 3 + (x + start_x) * 3 + ch];
                }
            }
        }
        delete[] resized;

        std::cout << "Final image size: " << W << "x" << H << "x3\n";
        std::cout << "Color order: " << (bgr ? "BGR" : "RGB") << "\n";
        std::cout << "Pixel scaling: " << (normalize ? "0.0 to 1.0 (PyTorch-style)" : "0 to 255 (Caffe-style)") << "\n";
        std::cout << "Mean (R,G,B): [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]\n";
        std::cout << "Std (R,G,B):  [" << std_dev[0] << ", " << std_dev[1] << ", " << std_dev[2] << "]\n";


        // Step 3: Create TVM Input Tensor (NCHW float32)
        tvm::runtime::NDArray input_nd = tvm::runtime::NDArray::Empty(
            {1, 3, H, W}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
        float* input_data = static_cast<float*>(input_nd->data);


        // --- Final Preprocessing: HWC uint8 -> NCHW float32 (with normalization) ---

        for (int ch = 0; ch < 3; ch++) { // Iterate over output channels (C)
            for (int y = 0; y < H; y++) { // Iterate over height (H)
                for (int x = 0; x < W; x++) { // Iterate over width (W)
                    
                    // Determine the source channel (src_ch) in the HWC data (img_cropped)
                    // The mean/std arrays are always indexed by (R=0, G=1, B=2)
                    int src_ch = bgr ? (2 - ch) : ch; // FIX: Correct BGR/RGB channel swap and align with mean/std

                    // Get the raw pixel value
                    float v = img_cropped[y * W * 3 + x * 3 + src_ch];

                    // Apply 0-1 scaling if requested
                    if (normalize) {
                        v /= 255.0f;
                    }

                    // Apply mean subtraction and standard deviation division
                    // We use src_ch to look up the correct mean/std for the pixel being processed.
                    input_data[ch * H * W + y * W + x] = (v - mean[src_ch]) / std_dev[src_ch];
                }
            }
        }
        delete[] img_cropped;

        // --- Inference Execution ---

        // Debug: Print input tensor range (should be ~[-2, 2] for ImageNet normalized)
        float min_val = input_data[0], max_val = input_data[0];
        double sum = 0.0;
        for (int i = 0; i < 3 * H * W; i++) {
            if (input_data[i] < min_val) min_val = input_data[i];
            if (input_data[i] > max_val) max_val = input_data[i];
            sum += input_data[i];
        }
        std::cout << "Input tensor range: [" << min_val << ", " << max_val << "], mean: " << sum / (3 * H * W) << "\n";

        // Set input tensor
        gmod.GetFunction("set_input")(input_name, input_nd);

        // Run inference
        gmod.GetFunction("run")();

        // Get output tensor
        tvm::runtime::NDArray output_nd = gmod.GetFunction("get_output")(0);
        
        // Debug output shape
        std::cout << "Output shape: ";
        for (int i = 0; i < output_nd->ndim; i++) {
            std::cout << output_nd->shape[i] << " ";
        }
        std::cout << "\n";

        // Copy output data from TVM NDArray to std::vector
        int out_size = 1;
        for (int i = 0; i < output_nd->ndim; i++) { out_size *= output_nd->shape[i]; }
        
        std::vector<float> logits(out_size);
        output_nd.CopyToBytes(logits.data(), out_size * sizeof(float));

        // Debug: Print output statistics
        float min_logit = logits[0], max_logit = logits[0];
        double logit_sum = 0.0;
        for (int i = 0; i < out_size; i++) {
            if (logits[i] < min_logit) min_logit = logits[i];
            if (logits[i] > max_logit) max_logit = logits[i];
            logit_sum += logits[i];
        }
        std::cout << "Logits range: [" << min_logit << ", " << max_logit << "], mean: " << logit_sum / out_size << "\n";

        // WARNING: Catches the "all-zero" bug if it returns (usually means compilation/load failure)
        if (max_logit == 0.0f && min_logit == 0.0f) {
            std::cerr << "WARNING: All output logits are zero! Model may not be processing correctly.\n";
            std::cerr << "Possible issues: 1. Input name mismatch. 2. Model weights loaded as zeros (compilation error).\n";
        }

        // --- Post-processing: Softmax and Top-5 ---

        // Softmax to convert logits to probabilities
        std::vector<float> probs(out_size);
        max_logit = *std::max_element(logits.begin(), logits.end());
        double exp_sum = 0.0;
        for (int i = 0; i < out_size; i++) {
            // Using logit - max_logit for numerical stability
            probs[i] = std::exp(logits[i] - max_logit); 
            exp_sum += probs[i];
        }
        for (int i = 0; i < out_size; i++) {
            probs[i] /= exp_sum;
        }

        // Top-5 predictions
        std::vector<Prediction> preds;
        std::vector<std::string> labels = LoadLabels(labels_path);
        
        if (labels.size() > 0 && labels.size() != out_size) {
            std::cerr << "Warning: Model output size (" << out_size 
                      << ") does not match number of labels (" << labels.size() << ").\n";
        }

        for (int i = 0; i < out_size; i++) {
            std::string label = (i < (int)labels.size()) ? labels[i] : "<no-label>";
            preds.push_back({i, probs[i], label});
        }

        std::sort(preds.begin(), preds.end(), [](const Prediction& a, const Prediction& b) {
            return a.prob > b.prob;
        });

        std::cout << "\n===== Running " << model_name << " =====\nTop-5 Predictions:\n";
        for (int i = 0; i < 5 && i < (int)preds.size(); i++) {
            std::cout << "  " << i + 1 << ") id=" << preds[i].id 
                      << "  p=" << preds[i].prob 
                      << "  " << preds[i].label << "\n";
        }
        std::cout << "Predicted class: " << preds[0].id << " (" << preds[0].label << ")\n";

    } catch (const tvm::runtime::Error& e) {
        std::cerr << "TVM Runtime Error:\n" << e.what() << "\n";
        return -1;
    } catch(const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}