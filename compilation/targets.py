"""Plain TVM target profiles.

Each profile is a plain dict with these keys:
  target  - TVM target string passed to relay.build
  host    - optional host target string for cross-compilation
  cc      - optional C compiler used when exporting the library
  ext     - output file extension: so, wasm, or tar
"""

TARGETS = {
    "x86_64": {
        "target": "llvm -mtriple=x86_64-linux-gnu -mcpu=x86-64",
        "cc":     "gcc",
        "ext":    "so",
    },
    "native": {
        "target": "llvm -mcpu=native",
        "cc":     "gcc",
        "ext":    "so",
        # Native artifacts are not portable across machines.
    },
    "raspi4_aarch64": {
        "target": "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -mattr=+neon",
        "host":   "llvm -mtriple=aarch64-linux-gnu",
        "cc":     "aarch64-linux-gnu-gcc",
        "ext":    "so",
    },
    "raspi_armv7": {
        "target": "llvm -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a72 -mattr=+neon,+vfp3",
        "host":   "llvm -mtriple=armv7-linux-gnueabihf",
        "cc":     "arm-linux-gnueabihf-gcc",
        "ext":    "so",
    },
    "wasm32": {
        "target": "llvm -mtriple=wasm32-unknown-unknown-wasm",
        "cc":     "emcc",
        "ext":    "wasm",
        # Requires Emscripten 3.x from emsdk, not the apt package.
    },
    "vulkan": {
        "target": "vulkan",
        "host":   "llvm -mtriple=x86_64-linux-gnu",
        "ext":    "so",
        # Requires TVM built with USE_VULKAN=ON.
    },
    "c": {
        "target": "c",
        "ext":    "tar",
        # C source export, useful as a bridge to microTVM-style bare-metal deployment.
    },
    "raspi5_aarch64": {
    "target": "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a76",
    "host":   "llvm -mtriple=aarch64-linux-gnu",
    "cc":     "aarch64-linux-gnu-gcc",
    "ext":    "so",
    },
}
