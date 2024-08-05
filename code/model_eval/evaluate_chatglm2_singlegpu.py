#!/usr/bin/env python
# _*_coding:utf-8_*_
import argparse
import os
import json
import codecs
from transformers import AutoModel, AutoTokenizer
def read_jsonl(file_path):
    json_list = []
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            e = json.loads(line.strip())
            # e = eval(line.strip())
            json_list.append(e)
    return json_list

def process_item(in_item, model, tokenizer, ans_file):
    raw_input_str = in_item.get("instruction","")
    history = []
    response, history = model.chat(
        tokenizer,
        raw_input_str,
        history,
        max_length=1024,
        temperature=0.75,
        top_p=0.95,
    )
    in_item['chatglm2_cot_output'] = response
    with open(ans_file,'a+',encoding='utf-8') as f:
        # print(in_item)
        f.write(json.dumps(in_item, ensure_ascii=False)+'\n')
        f.flush()
    # print(json.dumps(in_item, ensure_ascii=False), file=ans_file)


if __name__ == "__main__":
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    args = parser.parse_args()
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    # 数据文件
    # infile = codecs.open(os.path.expanduser(args.test_file), "r", encoding="utf-8")
    examples = read_jsonl(os.path.expanduser(args.test_file))
    if "qwen" in args.model_path.strip('/').split('/')[-2].lower() or "chatglm" in args.model_path.strip('/').split('/')[-2].lower():
        save_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-2], args.model_path.strip('/').split('/')[-1])
    else:
        save_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-1])
    # save_dir = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-1])
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    answer_file = os.path.join(save_dir,  "eval_{}".format(args.test_file.split('/')[-1]))
    # ans_file = codecs.open(os.path.expanduser(answer_file), "w", encoding="utf-8")
    ans_file = os.path.expanduser(answer_file)
    if os.path.exists(ans_file): #从上一次预测的地方继续预测
        target_examples = read_jsonl(ans_file)
        examples = examples[len(target_examples):]
    # ignore_cnt = 2808
    # now_id = 0
    # for line in infile:
    #     # now_id += 1
    #     # if now_id <= ignore_cnt: continue
    #     line = line.strip()
    #     if not line:
    #         continue
    #     item = json.loads(line)
    for item in examples:
        process_item(item, model, tokenizer, ans_file)

'''
CUDA_VISIBLE_DEVICES=0 python evaluate_chatglm2_singlegpu.py --model_path /pfs-LLM/common/edu_workspace/Math/checkpoint/chatglm2-6b --batch_size 1 --test_file ../../data/test.ape.jsonl --output_dir ../../results/chatglm2 
'''