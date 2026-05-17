# Cross-Compilation

The compile host is the devcontainer. The target device is where the compiled artifact will run.

## Compile For x86_64

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile x86_64
```

## Compile For Raspberry Pi 4 64-bit

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile raspi4_aarch64
```

This uses:

```text
target = llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -mattr=+neon
cc     = aarch64-linux-gnu-gcc
cxx    = aarch64-linux-gnu-g++
```

## Compile For Raspberry Pi 32-bit

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile raspi_armv7
```

This uses:

```text
target = llvm -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a72 -mattr=+neon,+vfp3
cc     = arm-linux-gnueabihf-gcc
cxx    = arm-linux-gnueabihf-g++
```

## Raw Target Override

Use raw target overrides only after a named profile works:

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile x86_64 \
  --target "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 -mattr=+neon" \
  --host "llvm -mtriple=aarch64-linux-gnu"
```

## Deployment Boundary

This repository compiles artifacts and builds target-side examples. It does not automate copying files to devices. Transfer the compiled artifacts and target executable manually using your preferred method.
