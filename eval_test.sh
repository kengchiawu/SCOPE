export CUDA_VISIBLE_DEVICES=4,5

method=H2O # Support ALLKV, PyramidKV, SnapKV, H2O, StreamingLLM
max_capacity_prompts=512
attn_implementation=flash_attention_2 # Support "flash_attention_2", "sdpa", "eager".
#source_path=$5
model_path='meta-llama/Llama-3.2-1B-Instruct'
decoding_metric=h2o # H2O Support None,h2o,(fixed,linear,jump)---SCOPE
decoding_window_size=64
decoding_recent_size=64
save_dir='~/SCOPE/results' # path to result save_dir
K=30 #30,60
T=20

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