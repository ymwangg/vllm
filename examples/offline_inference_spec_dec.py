import numpy as np
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Write a Python function to tell me what the date is today."
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512, logprobs=5, prompt_logprobs=2)

# Create an LLM.
llm = LLM(model="/home/ubuntu/models/Llama-2-7b-chat-hf/",
          draft_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
          enforce_eager=True,
          speculate_length=5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
    )
