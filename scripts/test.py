
from safetensors import safe_open

tensors = {}
with safe_open("/home/ubuntu/TensorRT-Model-Optimizer/llm_ptq/saved_models_suzuka-llama3-1_dense_int4_awq_tp1_pp1/rank0.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

tensorsa = {}
with safe_open("/home/ubuntu/suzuka_models/suzuka-llama3-1.3B-awq/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensorsa[k] = f.get_tensor(k)


print()