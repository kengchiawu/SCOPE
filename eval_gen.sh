results_dir="./results/llama-3.1-8b-instruct_1024_eager"
decoding_metric="None"
same_strategy=True
for decoding_metric in None fixed linear jump
do
    python3 eval_longgenbench.py \
        --results_dir ${results_dir} \
        --decoding_metric ${decoding_metric}  \
        --same_strategy ${same_strategy}
done