from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TargetProfile:
    name: str
    target: str
    host: Optional[str] = None
    cc: Optional[str] = None
    cxx: Optional[str] = None
    output_format: str = "so"
    notes: str = ""


TARGET_PROFILES: Dict[str, TargetProfile] = {
    "x86_64": TargetProfile(
        name="x86_64",
        target="llvm -mtriple=x86_64-linux-gnu -mcpu=x86-64",
        cc="gcc",
        cxx="g++",
        notes="Native Linux x86_64 CPU target for the devcontainer.",
    ),
    "native": TargetProfile(
        name="native",
        target="llvm -mcpu=native",
        cc="gcc",
        cxx="g++",
        notes="Best for local validation only; do not use for portable artifacts.",
    ),
    "raspi4_aarch64": TargetProfile(
        name="raspi4_aarch64",
        target="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -mattr=+neon",
        host="llvm -mtriple=aarch64-linux-gnu",
        cc="aarch64-linux-gnu-gcc",
        cxx="aarch64-linux-gnu-g++",
        notes="Raspberry Pi 4 running a 64-bit Linux OS.",
    ),
    "raspi_armv7": TargetProfile(
        name="raspi_armv7",
        target="llvm -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a72 -mattr=+neon,+vfp3",
        host="llvm -mtriple=armv7-linux-gnueabihf",
        cc="arm-linux-gnueabihf-gcc",
        cxx="arm-linux-gnueabihf-g++",
        notes="32-bit Raspberry Pi Linux target.",
    ),
    "wasm32": TargetProfile(
        name="wasm32",
        target="llvm -mtriple=wasm32-unknown-unknown-wasm",
        cc="emcc",
        cxx="em++",
        output_format="wasm",
        notes="WebAssembly target. Runtime packaging differs from Linux shared libraries.",
    ),
    "vulkan": TargetProfile(
        name="vulkan",
        target="vulkan",
        host="llvm -mtriple=x86_64-linux-gnu",
        output_format="so",
        notes="GPU target for Vulkan-capable hosts. Use instead of CUDA in this guide.",
    ),
    "c": TargetProfile(
        name="c",
        target="c",
        output_format="tar",
        notes="C source export path used as the bridge toward microTVM-style deployment.",
    ),
}


def list_target_profiles() -> list[str]:
    return sorted(TARGET_PROFILES)


def get_target_profile(name: str) -> TargetProfile:
    try:
        return TARGET_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(list_target_profiles())
        raise ValueError(f"Unknown target profile '{name}'. Known profiles: {known}") from exc
