import tvm
from tvm.contrib import graph_executor
import numpy as np

# 1) Load compiled operator library (.so)
lib = tvm.runtime.load_module("deploy_lib_cpu.so")

# 2) Load execution graph (.json) and parameters (.params)
with open("deploy_graph.json") as f:
    graph_json = f.read()
with open("deploy_params.params", "rb") as f:
    param_bytes = f.read()

# 3) Create executor on a device
dev = tvm.cpu(0)
module = graph_executor.create(graph_json, lib, dev)

# 4) Load parameters into the executor
module.load_params(param_bytes)

# 5) Prepare and set input (must match name/shape/dtype)
input_data = np.random.randn(1, 3, 224, 224).astype("float32")
module.set_input("input", input_data)

# 6) Run and fetch output(s)
module.run()

out = module.get_output(0).asnumpy()
print("✅ Output shape:", out.shape)
