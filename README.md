# SCOPE
SCOPE: Optimizing KV Cache Compression in Long-context Generation

> **Note:** This repo demonstrates the core principles of the SCOPE method and we will make all the code publicly available upon the acceptance of the paper.
>
> **Three decoding strategies:** Slide, Adaptive, Discontinuous
>
> **Setting:** Slide(decoding_metric="fixed"), Adaptive(decoding_metric="linear"), Discontinuous(decoding_metric="jump")


## Inference in GSM8K+(LongGenBench_examples/gsm8k_30.jsonl)

```bash
export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support ALLKV, PyramidKV, SnapKV, H2O, StreamingLLM
max_capacity_prompts=$3
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
source_path=$5
model_path=$6
decoding_metric=$7 # H2O Support None,h2o,(fixed,linear,jump)---SCOPE
decoding_window_size=$8
save_dir=$9 # path to result save_dir
K=$10 #30,60
T=$11

python3 run_longgenbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --K ${K}\
    --decoding_window_size ${decoding_window_size} \
    --decoding_recent_size ${decoding_recent_size} \
    --decoding_metric ${decoding_metric} \
    --max_num_examples ${T} \
```

## Eval Acc in GSM8K+

```bash
results_dir=$1

python3 eval_gen.py \
    --results_dir ${results_dir}
```
