# Target Profiles

Target profiles live in `examples/python/tvm_prep/targets.py`.

Use `--target-profile` for the common path and `--target` only when you need a raw TVM target override.

## Profiles

- `native`: local CPU with `llvm -mcpu=native`; best for quick validation only.
- `x86_64`: portable x86_64 Linux CPU artifact.
- `raspi4_aarch64`: Raspberry Pi 4 with 64-bit Linux.
- `raspi_armv7`: Raspberry Pi or ARM Linux device with 32-bit hard-float userspace.
- `wasm32`: WebAssembly output path.
- `vulkan`: Vulkan-capable GPU target; preferred over CUDA for this guide.
- `c`: C source export path for microTVM-style work.

## Target vs Host

The TVM target describes generated model code. The host target describes host-side runtime code that packages and calls the generated kernels.

For normal CPU builds these are often the same. For GPU targets like Vulkan, the target is `vulkan` and the host is usually an LLVM CPU target.

## Cross-Compilation Notes

Cross-compilation requires a compatible compiler and usually a sysroot for the target device. The devcontainers include:

```bash
aarch64-linux-gnu-gcc
aarch64-linux-gnu-g++
arm-linux-gnueabihf-gcc
arm-linux-gnueabihf-g++
emcc
em++
```

For production embedded deployments, match the compiler and sysroot to the exact OS image on the device.
