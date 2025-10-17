#include <dlpack/dlpack.h>
#include <cstring>
#include <vector>
#include <iostream>

extern "C" int32_t tvmgen_default_run(DLTensor *input0, DLTensor *output0);

static void MakeDLTensor(float *data, int64_t *shape, int ndim, DLTensor *t)
{
  t->data = data;
  t->device = DLDevice{kDLCPU, 0};
  t->ndim = ndim;
  t->dtype = DLDataType{kDLFloat, 32, 1};
  t->shape = shape;
  t->strides = nullptr;
  t->byte_offset = 0;
}

int main()
{
  // Input/output host buffers
  std::vector<float> in(1 * 3 * 224 * 224), out(1 * 1000);
  int64_t ishape[4] = {1, 3, 224, 224};
  int64_t oshape[2] = {1, 1000};

  // Fill input
  std::fill(in.begin(), in.end(), 0.0f);

  // Wrap as DLTensor views
  DLTensor tin{}, tout{};
  MakeDLTensor(in.data(), ishape, 4, &tin);
  MakeDLTensor(out.data(), oshape, 2, &tout);

  // Call the AOT entrypoint
  int32_t rc = tvmgen_default_run(&tin, &tout);
  if (rc != 0)
  {
    std::cerr << "run failed: " << rc << "\n";
    return 1;
  }

  std::cout << "ok, out[0..4]: " << out[0] << " " << out[1] << " " << out[2]
            << " " << out[3] << " " << out[4] << "\n";
  return 0;
}
