results_dir="./results/llama-3.1-8b-instruct_2048_eager"
decoding_metric="None"

python3 eval_longgenbench.py \
    --results_dir ${results_dir} \
    --decoding_metric ${decoding_metric}  \
    --same_strategy 
