#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// Include TVM runtime headers
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

void RunTVMModel(const std::string &model_path, const std::string &lib_path,
                 const std::string &params_path, int device_type, int device_id)
{
    // 1. Load the TVM graph, library, and parameters

    // Load graph JSON
    std::ifstream graph_file(model_path);
    std::string graph_json((std::istreambuf_iterator<char>(graph_file)),
                           std::istreambuf_iterator<char>());

    // Load compiled library
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);

    // Load parameters
    std::ifstream params_file(params_path, std::ios::binary);
    std::string params_bytes((std::istreambuf_iterator<char>(params_file)),
                             std::istreambuf_iterator<char>());

    tvm::runtime::TVMByteArray params_arr;
    params_arr.data = params_bytes.c_str();
    params_arr.size = params_bytes.length();

    // 2. Create TVM runtime module
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
        graph_json, lib, device_type, device_id);

    // Set parameters
    mod.GetFunction("load_params")(params_arr);

    // 3. Prepare Input Data (Example: a simple 1x3x224x224 float32 tensor)
    std::vector<float> input_data(1 * 3 * 224 * 224);
    // Fill with some dummy data (e.g., random values or all zeros)
    std::iota(input_data.begin(), input_data.end(), 0.0f); // Example: fill with increasing floats

    // Input shape
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    // Create DLTensor for input
    DLTensor input_tensor;
    input_tensor.data = input_data.data();
    input_tensor.device = {static_cast<DLDeviceType>(device_type), device_id};
    input_tensor.ndim = input_shape.size();
    input_tensor.dtype = {kDLFloat, 32, 1};
    input_tensor.shape = input_shape.data();
    input_tensor.strides = nullptr; // Let TVM infer
    input_tensor.byte_offset = 0;

    // 4. Set Input
    // Assuming your model has an input named "input" (common for ONNX)
    mod.GetFunction("set_input")("input", &input_tensor);

    // 5. Run Inference
    mod.GetFunction("run")();

    // 6. Get Output (Example: a single output tensor)
    DLTensor output_tensor;
    // Get number of outputs if unknown
    int num_outputs = mod.GetFunction("get_num_outputs")();
    if (num_outputs == 0)
    {
        std::cerr << "Error: Model has no outputs." << std::endl;
        return;
    }
    // Assuming we want the first output
    mod.GetFunction("get_output")(0, &output_tensor);

    // Now, output_tensor contains the result.
    // You can copy it to a CPU buffer if it's on a different device.
    // For demonstration, let's assume it's on CPU or copy it to CPU.

    // Calculate output size
    int64_t output_size = 1;
    for (int i = 0; i < output_tensor.ndim; ++i)
    {
        output_size *= output_tensor.shape[i];
    }
    std::vector<float> output_data(output_size);

    // If output is on GPU/other device, copy it back to CPU
    if (output_tensor.device.device_type != kDLCPU)
    {
        DLTensor cpu_output_tensor = output_tensor; // Create a DLTensor pointing to CPU memory
        cpu_output_tensor.data = output_data.data();
        cpu_output_tensor.device = {kDLCPU, 0};
        TVMArrayCopyFromTo(&output_tensor, &cpu_output_tensor, nullptr);
    }
    else
    {
        // If already on CPU, just cast and copy
        memcpy(output_data.data(), output_tensor.data, output_size * sizeof(float));
    }

    // Print a few output values (for demonstration)
    std::cout << "Inference successful! First 10 output values: ";
    for (int i = 0; i < std::min((int)output_data.size(), 10); ++i)
    {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    // These paths should point to your compiled TVM artifacts
    std::string model_json_path = "model.json";     // Path to graph JSON
    std::string model_lib_path = "model.so";        // Path to compiled shared library
    std::string model_params_path = "model.params"; // Path to parameters

    // Device configuration
    // kDLCPU = 1, kDLGPU = 2, kDLVulkan = 10, etc.
    // See include/tvm/runtime/c_runtime_api.h for DLDeviceType
    int device_type = kDLCPU; // Or kDLGPU, kDLVulkan, etc.
    int device_id = 0;

    RunTVMModel(model_json_path, model_lib_path, model_params_path, device_type, device_id);

    return 0;
}