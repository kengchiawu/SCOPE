import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

datasets = ["gsm8k"]#, "csqa", "mmlu"]

dataset2maxlen_8K = {
    "gsm8k": 7950,
    "mmlu": 7950,
    "csqa": 7950,
}

dataset2maxlen_4K = {
    "gsm8k": 4096,
    "mmlu": 4096,
    "csqa": 4096,
}

model2prompt = {
    "gsm8k": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number.",
    "mmlu": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number. The following are multiple choice questions (with answers) about ",
    "csqa": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number.",
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3-": 7950,
    "llama-3-": 7950,
    "llama3.1": 130000,
    "llama-3.1": 130000,
    "llama3.2": 130000,
    "llama-3.2": 130000,
    "mistral": 31500
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat_llama2(system_prompt, prompt):
    return f"[INST] <<SYS>>\n {system_prompt} \n<</SYS>>\n\n{prompt} [/INST]"

def build_chat_llama3_modify(system_prompt, prompt):
    return f"<<SYS>>\n {system_prompt} \n<</SYS>>\n\n{prompt}"

def build_chat_llama3(system_prompt, prompt):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def build_chat_llama3_wo_system(system_prompt, prompt):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{system_prompt}\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


def main(args):
    print("Loading data...")
    test_data = []
    prompts = []
    questionss = []
    answerss = []
    lengths = []
    
    model_path = args.model_path.lower()

    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    if args.K == 30:
        output_max_len = dataset2maxlen_4K[args.dataset]
    else:
        output_max_len = dataset2maxlen_8K[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            template = model2prompt[args.dataset]
            system_prompt = template.format(**example)
            #print(list(example.keys()))
            #raise RuntimeError("i need stop!")
            
            # mmlu
            if "task" in example:
                template = template + example["task"] + "."
                # print(template)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat_llama2(system_prompt, example["prompt"])
            
            elif "llama-3" in args.model_path.lower() and (
                # chat models are better off without build prompts on these tasks
                args.dataset not in [""] # "gsm8k"... , 
            ):
                print(f"using template for {args.dataset}")
                # prompt = build_chat_llama3(system_prompt, example["prompt"])
                # prompt = build_chat_llama3_wo_system(system_prompt, example["prompt"])
                prompt = build_chat_llama3_modify(system_prompt, example["prompt"])
            else:
                print(f"NOT using template for {args.dataset}")
                prompt = system_prompt + "\n\n" + example["prompt"]

            example["prompt"] = prompt
                
            test_data.append(example)


    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        prompts.append(example["prompt"])
        questionss.append(example["questions"])
        answerss.append(example["answers"])
        lengths.append(len(example["prompt"]))

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}_{args.attn_implementation}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}_{args.attn_implementation}", args.dataset, f"pre_{args.method}_dec_{args.decoding_metric}.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_questionss = questionss[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask


        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        # # default to True
        # if args.method == "DynamicKV":
        #     args.output_attentions = True
        # else:
        #     args.output_attentions=False

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        
        if args.method != "FullKV":
            if args.method.lower() in ["snapkv","pyramidkv","h2o","allkv", "quest"]:
                window_sizes = 8
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts//2

            kernel_sizes = 7
            pooling = "maxpool"

            chunk_size = args.chunk_size
            page_select_strategy = args.page_select_strategy

            same_strategy = args.same_strategy

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.decoding_metric = args.decoding_metric
                model.model.layers[i].self_attn.config.decoding_window_size = args.decoding_window_size
                model.model.layers[i].self_attn.config.decoding_recent_size = args.decoding_recent_size
                # model.model.layers[i].self_attn.config.delta = args.delta
                model.model.layers[i].self_attn.config.delta = (output_max_len - model.model.layers[i].self_attn.config.decoding_recent_size) // (model.model.layers[i].self_attn.config.decoding_window_size - model.model.layers[i].self_attn.config.decoding_recent_size)
                # print(f"layer {i} delta {model.model.layers[i].self_attn.config.delta}")

                # new config for Quest
                model.model.layers[i].self_attn.config.chunk_size = chunk_size
                model.model.layers[i].self_attn.config.page_select_strategy = page_select_strategy

                # new config for differ_strategy
                model.model.layers[i].self_attn.config.same_strategy = same_strategy

        context_length = batch_input_ids.shape[-1]
                
        output = model.generate(
            **tokenized_prompts,
            output_attentions = args.output_attentions,
            max_new_tokens=output_max_len,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )


        batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        
        # print(f"debbug batch_outputs {batch_outputs}")
        
        batch_generations = batch_outputs

        torch.cuda.empty_cache()

        for j in range(args.eval_batch_size):
            
            example = {}
            
            example["prompt"] = batch_prompts[j]
            example["questions"] = batch_questionss[j]
            example["answers"] = batch_answerss[j]
            example["length"] = batch_lengths[j]
            example["pred"] = batch_generations[j]

            fout.write(json.dumps(example) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--decoding_metric", type=str, default="None", help="")
    parser.add_argument("--decoding_window_size", type=int, default=1024, help="")
    parser.add_argument("--decoding_recent_size", type=int, default=128, help="")
    

    # parser.add_argument("--delta", type=int, default=15, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")

    parser.add_argument("--K", type=int, default=-1, help="number of questions parallel")
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    '''
    此后为添加的参数
    '''
    # new parameters for Quest
    parser.add_argument("--chunk_size",type=int,default=16,help='')
    parser.add_argument("--page_select_strategy",type=str,default='amax',help='')

    # new parameters for the number of shots
    parser.add_argument("--shot_number",type=int,default=8,help='the number of shots, we got 8 shots in gsm8k and 5shots in csqa')

    # new parameter to define whether using diferent strategy in prefill and decoding phase
    parser.add_argument("--same_strategy", type=bool, default=False, help="")

    args = parser.parse_args()
    
    set_seed(args.seed)
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from model.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    save_dir = args.save_dir
        
    max_capacity_prompts = args.max_capacity_prompts
    
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        if args.dataset == "csqa":
            args.K = int(args.K / 3 * 4) # GSM8K/MMLU has 30,60 questions in a single long input;CSQA has 40,80 questions
        args.data_file = f"./data/longgenbench_examples/{args.dataset}_{args.K}_{args.shot_number}shot.jsonl"
        
        main(args)
