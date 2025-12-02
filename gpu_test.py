from vllm import LLM, SamplingParams
import time
import subprocess

# Initialize with optimal settings
# Using 7B model - 8B model still too large for 4x 11.6GB GPUs
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",  # 7B model will fit
    tensor_parallel_size=4,
    dtype="float16",
    gpu_memory_utilization=0.75,
    max_model_len=2048,
    trust_remote_code=True,
    disable_custom_all_reduce=True,
)

# Batch multiple prompts for better GPU utilization
prompts = [
    "Hello world! Explain multi-GPU LLM inference.",
    "What are the benefits of tensor parallelism?",
    "How does KV cache affect inference speed?",
    "Describe the role of PagedAttention in vLLM.",
    "What is continuous batching?",
]

# Optimized sampling parameters
params = SamplingParams(
    max_tokens=64,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

# Benchmark inference
print(f"Processing {len(prompts)} prompts...")
print("="*60)
start_time = time.time()

results = llm.generate(prompts, sampling_params=params)

end_time = time.time()
elapsed = end_time - start_time

# Calculate metrics
total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
tokens_per_second = total_tokens / elapsed
prompts_per_second = len(prompts) / elapsed

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)
print(f"Total prompts: {len(prompts)}")
print(f"Total time: {elapsed:.2f}s")
print(f"Throughput: {prompts_per_second:.2f} prompts/sec")
print(f"Total tokens generated: {total_tokens}")
print(f"Tokens/second: {tokens_per_second:.2f}")
print(f"Average tokens per prompt: {total_tokens/len(prompts):.1f}")
print("="*60)

# Print results
for i, result in enumerate(results):
    print(f"\n[Prompt {i+1}] {prompts[i][:50]}...")
    print(f"[Output] {result.outputs[0].text}")
    print(f"[Tokens] {len(result.outputs[0].token_ids)}")

# GPU memory stats
print("\n" + "="*60)
print("GPU MEMORY USAGE")
print("="*60)
try:
    gpu_stats = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits']
    ).decode('utf-8')
    for line in gpu_stats.strip().split('\n'):
        gpu_id, mem_used, mem_total, util = line.split(',')
        print(f"GPU {gpu_id}: {mem_used.strip()}MB / {mem_total.strip()}MB "
              f"({float(mem_used)/float(mem_total)*100:.1f}%) | Util: {util.strip()}%")
except Exception as e:
    print(f"Could not retrieve GPU stats: {e}")
print("="*60)
