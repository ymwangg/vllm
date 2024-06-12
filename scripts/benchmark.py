"""
This scripts show how to benchmark speculative decoding on a dataset.
"""
from vllm import LLM, SamplingParams
import sys
import time
import json
import numpy as np
import json
from transformers import AutoTokenizer;
import pdb
# use_speculate = False
use_cudagraph = True

# output file name
outname = "tmp.json"
args = sys.argv[1:]
bs = int(args[1]) # batch size
use_speculate = bool(int(args[2]))
# bs = 1
tp_size = 1 # tensor parallel degree for target model
draft_model_tp_size = 1
speculate_length = 5 # speculate length
max_tokens = 256 # maximum number of genrated tokens

# dataset
dataset = "/home/ubuntu/vllm-aws/scripts/human_eval.json"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=1.0,  max_tokens=max_tokens)

# target_model = "mistralai/Mistral-7B-Instruct-v0.2"
# target_model = "/home/ubuntu/Llama-2-70B-chat-hf"
target_model = "/home/ubuntu/Llama3/Meta-Llama-3-8B"

draft_model = args[0]
enforce_eager = not use_cudagraph

# Sample prompts.
with open(dataset) as fh:
    data = json.load(fh)

# Prepare batches
prompts_list = []
step = 0
offset = 0
while True:
    prompt = [item['prompt'] for item in data[offset:offset+bs]]
    prompts_list.append(prompt)
    step += 1
    offset += bs
    if len(prompt) < bs:
        break

# Create an LLM.
engine_kwargs = {
    "model": target_model,
    "tensor_parallel_size": tp_size,
    "enforce_eager": enforce_eager,
    "gpu_memory_utilization": 0.75,
    "quantization": None,
    "disable_custom_all_reduce": False,
}

if use_speculate:
    engine_kwargs.update({
                "speculative_model":
                draft_model,
                "num_speculative_tokens":
                speculate_length,
                "use_v2_block_manager":
                True,
                "draft_model_tp_size":
                draft_model_tp_size,
            })

llm = LLM(**engine_kwargs)

# pdb.set_trace()
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

# warm up
llm.generate(prompts_list[0], sampling_params)

outputs_list = []
t0 = time.monotonic()
step = 0
while step < len(prompts_list):
    t2 = time.monotonic()
    prompts = prompts_list[step % len(prompts_list)]
    if len(prompts) == 0:
        break
    outputs = llm.generate(prompts, sampling_params)
    outputs_list.append(outputs)
    step += 1
    t3 = time.monotonic()
    print(f"batch_{step} done in {t3-t2} sec")
t1 = time.monotonic()
prompt_lens = []
text_lens = []
history = []
records = []

for outputs in outputs_list:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        prompt_len = len(output.prompt_token_ids)
        prompt_lens.append(prompt_len)
        text_len = len(output.outputs[0].token_ids)
        text_lens.append(text_len)
        history.append(output.outputs[0].acceptance_history)
        m = np.mean(history[-1])
        token_ids = output.prompt_token_ids + output.outputs[0].token_ids
        records.append({'prompt': prompt, 'response': generated_text, 'acceptance': m, 'prompt_len': prompt_len, 'response_len': text_len, 'token_ids': token_ids})
dt = t1-t0
with open("results.txt", "a") as text_file:
    print(f"======= batch: {bs}, target:{target_model} draft:{draft_model} =========", file=text_file)
    print(f"finished in {t1-t0} seconds", file=text_file)
    print(f"Mean avg prompt length = {np.mean(prompt_lens)}", file=text_file)
    print(f"Mean avg response length = {np.mean(text_lens)}", file=text_file)
    print(f"Throughput = {np.sum(text_lens)/dt} tokens/s", file=text_file)
    avg_accept = np.mean([np.mean(x) for x in history if x])
    print(f"Avg accepted = {avg_accept} \n", file=text_file)
with open(outname,'w') as fh:
    json.dump(records, fh)