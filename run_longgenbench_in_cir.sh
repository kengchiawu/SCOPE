export CUDA_VISIBLE_DEVICES=4,5

method=Quest # Support ALLKV, PyramidKV, SnapKV, H2O, StreamingLLM, Quest
max_capacity_prompts=2048
attn_implementation=eager # Support "flash_attention_2", "sdpa", "eager".
#source_path=$5
model_path='meta-llama/Llama-3.1-8B-Instruct'
decoding_metric=None # H2O Support None,h2o,(fixed,linear,jump)---SCOPE
decoding_window_size=512
#decoding_window_size指的是decoding阶段KV Cache的新增长度上限
decoding_recent_size=256
#decoding_recent_size指的是decoding阶段采用的h2o方法中local window size
save_dir='./results' # path to result save_dir
K=30 #30,60
T=20
#Quest
chunk_size=16
page_select_strategy='amax'
# number of shots in data
shot_number=5 # in gsm8K 8, csqa 5
same_strategy=True
# 如果设置same_strategy=Ture，则max_capacity_prompts应该设置为max_capacity_prompts+decoding_window_size，重复使用update_kv

for method in ALLKV PyramidKV SnapKV H2O StreamingLLM Quest
do
    for decoding_metric in None # fixed linear jump
    do
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
            --chunk_size ${chunk_size} \
            --page_select_strategy ${page_select_strategy} \
            --shot_number ${shot_number} \
            --same_strategy ${same_strategy} 
    done
done
