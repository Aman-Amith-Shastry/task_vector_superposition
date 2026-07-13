cd /Users/amanshastry/Desktop/Python/langraph_basics && \
for g in arithmetic entity mmlu; do python experiments/sweep_$g.py --null-check --quantize int8 --model meta-llama/Llama-3.1-8B-Instruct; done && \
for g in arithmetic entity mmlu; do python experiments/sweep_$g.py --null-check --model Qwen/Qwen2.5-3B-Instruct; done && \
for g in arithmetic entity mmlu; do python experiments/sweep_$g.py --null-check --model meta-llama/Llama-3.2-3B-Instruct; done && \
for g in arithmetic entity mmlu; do python experiments/sweep_$g.py --null-check --model google/gemma-2-2b-it; done && \
for g in arithmetic entity mmlu; do python experiments/sweep_$g.py --null-check --model meta-llama/Llama-3.2-1B-Instruct; done