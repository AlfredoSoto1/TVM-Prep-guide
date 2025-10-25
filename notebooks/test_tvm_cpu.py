import numpy as np
import tvm
from tvm import te


def main():
    print("TVM version:", tvm.__version__)

    # 1) Check that the LLVM target is enabled (CPU JIT)
    has_llvm = tvm.runtime.enabled("llvm")
    print("LLVM available:", has_llvm)
    if not has_llvm:
        raise RuntimeError(
            "TVM CPU JIT (llvm) is not available. Check your TVM install.")

    target = "llvm"
    dev = tvm.cpu(0)

    # 2) Build and run a tiny vector-add kernel
    n = 1024
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)

    print("Building kernel for target:", target)
    fadd = tvm.build(s, [A, B, C], target=target, name="vadd")
    print("Lowered IR built OK. Running...")

    a_np = np.random.rand(n).astype("float32")
    b_np = np.random.rand(n).astype("float32")
    c_ref = a_np + b_np

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(np.zeros(n, dtype="float32"), device=dev)

    fadd(a_tvm, b_tvm, c_tvm)

    # 3) Validate numerics
    np.testing.assert_allclose(c_tvm.numpy(), c_ref, rtol=1e-5, atol=1e-5)
    print("✅ Vector add correctness check passed.")

    # 4) (Optional) Quick perf smoke check
    import time
    iters = 200
    t0 = time.time()
    for _ in range(iters):
        fadd(a_tvm, b_tvm, c_tvm)
    t1 = time.time()
    print(f"Ran {iters} iters in {t1 - t0:.4f}s ({iters/(t1-t0):.1f} iters/sec)")


if __name__ == "__main__":
    main()
