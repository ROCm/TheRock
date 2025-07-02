python3 -m sglang.bench_one_batch \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch 32 --input-len 2048 --output-len 32 \
  --tp 8 \
  --attention-backend wave
