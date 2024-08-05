import argparse
import os
import json
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

def read_jsonl(file_path):
    json_list = []
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            e = json.loads(line.strip())
            json_list.append(e)
    return json_list

if __name__ == "__main__":
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    device = "cuda" # the device to load the model onto
    assert args.batch_size == 1
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)

    examples = read_jsonl(args.test_file)
    if "qwen" in args.model_path.strip('/').split('/')[-2].lower() or "chatglm" in args.model_path.strip('/').split('/')[-2].lower():
        save_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-2], args.model_path.strip('/').split('/')[-1])
    else:
        save_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-1])

    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    answer_file = os.path.join(save_dir,  "eval_{}".format(args.test_file.split('/')[-1]))
    ans_file = os.path.expanduser(answer_file)

    if os.path.exists(ans_file): #从上一次预测的地方继续预测
        target_examples = read_jsonl(ans_file)
        examples = examples[len(target_examples):]

    for e in examples:
        prompt = e['instruction']
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        e['output'] = response
        with open(ans_file,'a+',encoding='utf-8') as f:
            # print(in_item)
            f.write(json.dumps(e, ensure_ascii=False)+'\n')
            f.flush()


'''

####################################### Qwen1.5-7B-Chat ####################################### 
# baseline
CUDA_VISIBLE_DEVICES=0 python evaluate_qwen15_singlegpu.py --model_path ../../checkpoint/Qwen1.5-7B-Chat --batch_size 1 --test_file ../../data/test.ape.jsonl --output_dir ../../results/sftnum_ablation/
'''