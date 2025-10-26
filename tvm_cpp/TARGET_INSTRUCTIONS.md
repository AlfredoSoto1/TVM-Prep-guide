# Apache TVM Target Architecture Cheat Sheet

This README summarizes how to define target architectures in Apache TVM for various devices including Raspberry Pi, Jetson, x86 CPUs, and microcontrollers. It uses official TVM documentation and helper methods to simplify target selection.

---

## 1. Target Basics

A TVM target string has the general format:

```
<target_kind> [-mtriple=<triple>] [-mcpu=<cpu_name>] [-mattr=<features>] [-keys=<key>] [-model=<model>]
```

* `<target_kind>`: Backend (e.g., `llvm`, `cuda`, `rocm`, `metal`, `opencl`, `c`)
* `-mtriple`: CPU/OS/ABI triple (used mainly for cross-compilation)
* `-mcpu`: CPU microarchitecture (or `native` for auto-detect)
* `-mattr`: Optional CPU features (e.g., `+neon`, `+sse4.2`)
* `-keys`, `-model`: Optional runtime keys or device models

Helper functions in TVM provide a simpler way to create targets without remembering full strings.

---

## 2. Common Devices and Target Definitions

### Raspberry Pi (ARM CPU)

| Model                  | TVM Helper                           | Equivalent Target String                                                  |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------- |
| Pi 4 (64-bit)          | `tvm.target.arm_cpu(model="raspi4")` | `llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -mattr=+neon`           |
| Pi 3 / Zero 2 (64-bit) | `tvm.target.arm_cpu(model="raspi3")` | `llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 -mattr=+neon`           |
| Pi (32-bit)            | `tvm.target.arm_cpu(model="raspi3")` | `llvm -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a72 -mattr=+neon,+vfp3` |

### x86 CPU (Desktop/Laptop)

| CPU            | TVM Helper                       | Target String                                  |
| -------------- | -------------------------------- | ---------------------------------------------- |
| Generic x86_64 | `tvm.target.llvm(mcpu="native")` | `llvm -mcpu=native`                            |
| Intel Haswell  | None (manual string)             | `llvm -mtriple=x86_64-linux-gnu -mcpu=haswell` |
| Intel Core2    | None (manual string)             | `llvm -mtriple=i686-linux-gnu -mcpu=core2`     |

### NVIDIA Jetson (ARM CPU + CUDA GPU)

| Device           | CPU Target                                  | GPU Target                      |
| ---------------- | ------------------------------------------- | ------------------------------- |
| Jetson Nano      | `tvm.target.arm_cpu(model="jetson-nano")`   | `tvm.target.cuda(arch="sm_53")` |
| Jetson Xavier NX | `tvm.target.arm_cpu(model="jetson-xavier")` | `tvm.target.cuda(arch="sm_75")` |

### Microcontrollers (STM32 examples)

| MCU     | TVM Helper                             |
| ------- | -------------------------------------- |
| STM32H7 | `tvm.target.stm32(series="stm32H7xx")` |
| STM32F4 | `tvm.target.stm32(series="stm32F4xx")` |

---

## 3. How to Find `-mcpu` and `-mtriple`

TVM passes these parameters to LLVM. To find valid options:

### For `-mcpu`

```bash
llc -mcpu=help  # lists CPU names LLVM recognizes
```

* Examples: `native`, `haswell`, `cortex-a72`, `cortex-a53`

### For `-mtriple`

```bash
# Linux
gcc -dumpmachine
uname -m
# Windows (MinGW)
x86_64-w64-windows-gnu
```

* Format: `<arch>-<vendor>-<os>`
* Examples: `x86_64-linux-gnu`, `armv7-linux-gnueabihf`, `aarch64-linux-gnu`

---

## 4. Using Helper Functions in Code

```python
import tvm

# Raspberry Pi 4
target = tvm.target.arm_cpu(model="raspi4")

# Jetson Xavier NX GPU
gpu_target = tvm.target.cuda(arch="sm_75")

# Desktop CPU (auto detect)
cpu_target = tvm.target.llvm(mcpu="native")

# Build Relay module
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

---

## 5. References

* TVM Target API: [https://tvm.apache.org/docs/reference/api/python/target.html](https://tvm.apache.org/docs/reference/api/python/target.html)
* Raspberry Pi Deployment Tutorial: [https://daobook.github.io/tvm/docs/how_to/deploy_models/deploy_model_on_rasp.html](https://daobook.github.io/tvm/docs/how_to/deploy_models/deploy_model_on_rasp.html)
* LLVM CPU and Triple documentation: [https://llvm.org/docs/CommandGuide/llc.html#cmdoption-mcpu](https://llvm.org/docs/CommandGuide/llc.html#cmdoption-mcpu)

