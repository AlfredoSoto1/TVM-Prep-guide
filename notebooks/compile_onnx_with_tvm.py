#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor

try:
    import onnx  # noqa: F401
except Exception as e:
    print("ERROR: onnx is not installed. pip install onnx", file=sys.stderr)
    raise


def parse_input_shape(shape_str: str) -> List[int]:
    """Parse input shape like "1,3,224,224" into a list of ints."""
    parts = [s.strip() for s in shape_str.split(",") if s.strip()]
    return [int(p) for p in parts]


def default_output_lib_name(out_dir: Path, base: str) -> Path:
    """Choose platform-specific extension for the exported shared library."""
    if sys.platform.startswith("win"):
        ext = ".dll"
    elif sys.platform == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    return out_dir / f"{base}{ext}"


def build_target_string(backend: str,
                        arch: Optional[str],
                        mtriple: Optional[str],
                        mcpu: Optional[str],
                        extra: Optional[List[str]]) -> str:
    """Build a TVM Target string with common knobs.

    Examples:
      - LLVM x86: backend='llvm', mcpu='skylake', extra=['-mattr=+avx2,+fma']
      - LLVM AArch64 cross: backend='llvm', mtriple='aarch64-linux-gnu', mcpu='cortex-a72'
      - CUDA: backend='cuda', arch='sm_86'
      - ROCm: backend='rocm'
      - Metal: backend='metal'
      - Vulkan: backend='vulkan'
    """
    items: List[str] = [backend]
    if arch and backend in ("cuda",):
        items.append(f"-arch={arch}")
    if mtriple and backend in ("llvm",):
        items.append(f"-mtriple={mtriple}")
    if mcpu and backend in ("llvm",):
        items.append(f"-mcpu={mcpu}")
    if extra:
        items.extend(extra)
    return " ".join(items)


def main():
    p = argparse.ArgumentParser(
        description="Compile ONNX to TVM shared library (+ graph.json, params.bin).")
    p.add_argument("--onnx", required=True,
                   help="Path to ONNX model (e.g., resnet18.onnx)")
    p.add_argument("--input-name", default="input",
                   help="Primary input tensor name (default: input)")
    p.add_argument("--input-shape", default="1,3,224,224",
                   help="Comma-separated shape (default: 1,3,224,224)")
    p.add_argument("--backend", default="llvm", choices=["llvm", "cuda", "rocm", "metal", "vulkan"],
                   help="TVM codegen backend / device target")
    p.add_argument("--arch", default=None,
                   help="Architecture string (e.g., sm_86 for CUDA)")
    p.add_argument("--mtriple", default=None,
                   help="LLVM target triple (e.g., aarch64-linux-gnu)")
    p.add_argument("--mcpu", default=None,
                   help="LLVM CPU model (e.g., skylake or cortex-a72)")
    p.add_argument("--extra", default=None, nargs="*",
                   help="Additional target options like -mattr=+avx2,+fma")
    p.add_argument("--opt-level", type=int, default=3,
                   help="Relay compiler opt level (0-3)")
    p.add_argument("--output-dir", default="tvm_out",
                   help="Directory to place compiled artifacts")
    p.add_argument("--basename", default="model_tvm",
                   help="Base filename for outputs")
    p.add_argument("--executor", default="graph", choices=["graph", "aot"],
                   help="Executor kind. 'graph' yields (lib + graph.json + params.bin). 'aot' builds AOT module.")
    args = p.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        print(f"ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_shape = parse_input_shape(args.input_shape)
    input_name = args.input_name

    # Load ONNX into Relay
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # Build target
    target_str = build_target_string(
        args.backend, args.arch, args.mtriple, args.mcpu, args.extra or [])
    target = tvm.target.Target(target_str)

    # Choose device kind (for testing/run)
    if args.backend == "cuda":
        dev = tvm.cuda(0)
    elif args.backend == "rocm":
        dev = tvm.rocm(0)
    elif args.backend == "metal":
        dev = tvm.metal(0)
    elif args.backend == "vulkan":
        dev = tvm.vulkan(0)
    else:
        dev = tvm.cpu()

    print(f"[TVM] Target: {target}\n[TVM] Device: {dev}")

    # Select executor
    if args.executor == "graph":
        exec_cfg = relay.backend.Executor("graph")
        # Optional: runtime can be "cpp" for a C++ runtime (requires TVM built with that runtime).
        runtime = relay.backend.Runtime("cpp")
    else:
        exec_cfg = relay.backend.Executor(
            "aot", {"interface-api": "c", "unpacked-api": True, "link-params": True})
        runtime = relay.backend.Runtime("crt")  # C runtime for AOT

    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build(mod, target=target, params=params,
                          executor=exec_cfg, runtime=runtime)

    # Export shared library
    lib_path = default_output_lib_name(out_dir, args.basename)
    lib.export_library(str(lib_path))

    if args.executor == "graph":
        # Save graph + params (needed by graph executor runtime)
        graph_json_path = out_dir / f"{args.basename}.graph.json"
        params_path = out_dir / f"{args.basename}.params.bin"
        with open(graph_json_path, "w", encoding="utf-8") as f:
            f.write(lib.get_graph_json())
        with open(params_path, "wb") as f:
            f.write(relay.save_param_dict(lib.get_params()))
        print(
            f"[OK] Exported:\n  {lib_path}\n  {graph_json_path}\n  {params_path}")
        print("\nTo run:\n  python run_tvm_module.py --lib {} --graph {} --params {} --backend {} --input-shape {}\n"
              .format(lib_path, graph_json_path, params_path, args.backend, args.input_shape))
    else:
        # AOT produces a single shared lib; typically integrate with TVM's CRT in C.
        metadata_path = out_dir / f"{args.basename}.metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("{}")  # Placeholder
        print(
            f"[OK] Exported AOT shared lib:\n  {lib_path}\n  (Integrate with TVM CRT for embedding.)")


if __name__ == "__main__":
    main()
