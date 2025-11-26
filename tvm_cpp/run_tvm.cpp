#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <cstring>
#include <iostream>
#include <vector>

int main() {
  // 1) Load the compiled module (.so / .dll)
  tvm::runtime::Module mod_factory =
      tvm::runtime::Module::LoadFromFile("resnet18_pi64.so");  // your file

  // 2) Choose device (CPU on target)
  //    kDLCPU for CPU; (kDLCUDA/kDLVulkan etc. if you built for GPU runtime)
  TVMDevice dev;
  dev.device_type = kDLCPU;
  dev.device_id = 0;

  // 3) Create a graph executor from the factory ("default" constructor)
  tvm::runtime::PackedFunc make = mod_factory.GetFunction("default");
  tvm::runtime::Module gmod = make(dev);  // same as Python: lib["default"](dev)

  // 4) Prepare input tensor
  //    Adjust name/dtype/shape as per your model
  const char* input_name = "input";  // or whatever you used in export
  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;
  std::vector<int64_t> shape = {1, 3, 224, 224};

  tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(shape, dtype, dev);

  // fill input with dummy data
  {
    float* ptr = static_cast<float*>(input->data);
    size_t n = 1ull * shape[0] * shape[1] * shape[2] * shape[3];
    for (size_t i = 0; i < n; ++i) ptr[i] = 0.0f;
  }

  // 5) Set input, run, get output
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

  set_input(input_name, input);
  run();

  // Assume single output at index 0, e.g., (1, 1000)
  tvm::runtime::NDArray out = get_output(0);
  std::vector<int64_t> out_shape(out->shape, out->shape + out->ndim);
  std::cout << "Output ndim=" << out->ndim << " shape=[";
  for (size_t i = 0; i < out_shape.size(); ++i) {
    std::cout << out_shape[i] << (i + 1 < out_shape.size() ? "," : "");
  }
  std::cout << "]\n";

  // (Optional) access output data
  float* out_ptr = static_cast<float*>(out->data);
  std::cout << "out[0]=" << out_ptr[0] << "\n";
  return 0;
}