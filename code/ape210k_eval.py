import re
import os
import json
import copy
from collections import Counter
import argparse
from utils import WRITE_FUNC

xuanze_name_list = ["选择","选择题","单选题"]
duoxuan_name_list = ["多选题"]
panduan_name_list = ["判断","判断题"]
jianda_name_list = ["简答","解答","解答题","应用题"]
tiankong_name_list = ["填空","填空题"]
jisuan_name_list = ["计算","计算题"]

panduan_correct_list = ["√","正确","对","yes","Yes","YES","True","true"]
panduan_wrong_list = ["×","错误","错","no","NO","No","False","false"]
def read_jsonl(file_path):
    json_list = []
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            e = json.loads(line.strip())
            json_list.append(e)
    return json_list

def preprocess_examples(examples):
    for e in examples:
        if "Human" in e['output']:
            e['output'] = e['output'].lstrip("Human: {}\nAssistant: ".format(e['input'])).replace('⁇',"").strip()
        e['target'] = str(e['target']).strip().rstrip(".").rstrip("。").rstrip('．').strip().lstrip("[解答过程]\n")
    return examples


# evaluate different task types differently
def get_merged_answerspan(output):
    ################# 答：################
    if output.count("答：")>1:
        merged_answer = ""
        splited_output = output.split("答：")[1:]
        for x in splited_output:
            merged_answer += x.split("\n")[0] + "\n"
        return merged_answer.replace(" ","")
    
    ################# （1），（2），（3）################
    if "（1）" in output and "（2）" in output:
        merged_answer = ""
        last_sentence = ""
        splited_output = output.split("\n")
        for x in splited_output:
            if x.startswith("（2）") or x.startswith("（3）") or x.startswith("（4）"):
                merged_answer += last_sentence + "\n"
            if len(x.strip())!=0:
                last_sentence = x
        merged_answer += last_sentence
        return merged_answer.strip().replace(" ","")
    return None


def get_last_sentence(e):
    output = e['output'].strip()
    # if e['question_type'] not in xuanze_name_list and "故选" in output:
    #     output = output.split("故选")[0].strip()
    # if e['question_type'] in xuanze_name_list and "故选" in output:
    #     output = output.split("故选")[1].strip()
    if "故答案为" in output: 
        output = output.split("故答案为")[1]
    if "故选" in output: 
        output = output.split("故选")[1]
    if "答：" in output: 
        output = output.split("答：")[1]
    output = output.replace(" ","")
    # first_sentence = output.split("。")[0].split("，")[0].split("\n")[0]
    # last_sentence = output.split("。")[-1].split("，")[-1].split("\n")[-1]
    # first_sentence = output.split("。")[0].split("\n")[0]
    # last_sentence = output.split("。")[-1].split("\n")[-1]
    last_sentence = output.split("\n")[-1]
    return last_sentence

def is_answer_matched(target, sentence):
    anss = target.split("；")
    flag = 1
    index = 0
    for x in anss:
        matched = False
        while(matched==False):
            if x in sentence[index:]:
                index = sentence.index(x,index)
                end_index = index + len(x)
                matched = True
                if index>=1 and sentence[index].isdigit() and sentence[index-1].isdigit(): 
                    matched = False
                if end_index<len(sentence) and ((sentence[end_index-1].isdigit() and sentence[end_index].isdigit()) or (sentence[end_index-1]!='}' and sentence[end_index]=='$')): 
                    matched = False
                index = end_index
            else:
                return 0
    return 1

def evaluate_general(e):
    last_sentence = get_merged_answerspan(e['output'])
    if last_sentence is None:
        last_sentence = get_last_sentence(e)
    if is_answer_matched(e['target'].lstrip("(").rstrip(")").replace("/","；"), last_sentence):
        return 1
    else:
        return 0

def cal_acc(examples):
    examples = preprocess_examples(examples)
    tot = len(examples)
    # xuanze_name_list = ["选择","选择题"]
    # duoxuan_name_list = ["多选题"]
    # panduan_name_list = ["判断","判断题"]
    # jianda_name_list = ["简答","解答","解答题","应用题"]
    # tiankong_name_list = ["填空","填空题"]
    for e in examples:
        e["is_correct"] = evaluate_general(e)

    correct = sum([e["is_correct"] for e in examples])
    print("acc: {}".format(correct/tot))
    print("#####################################################\n\n\n")
    return correct/tot, examples



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_output_path")
    parser.add_argument("--eval_file")
    parser.add_argument("--ckpt_name")
    args = parser.parse_args()

    base_output_path, eval_file, ckpt_name = args.base_output_path, args.eval_file, args.ckpt_name
    json_list = read_jsonl(os.path.join(base_output_path, ckpt_name, eval_file))
    for e in json_list:
        if 'answer' in e:
            e['target'] = e['answer'][0]
            e['task'] = e['question_type']
        if "output" not in e:
            e['output'] = e['chatglm2_cot_output']

    acc, predictions = cal_acc(json_list)
    write_func = WRITE_FUNC["csv"]
    write_func(predictions, os.path.join(base_output_path, ckpt_name, "{}.autoevaluated.csv".format(eval_file.rstrip(".jsonl"))))
    print("Finished!")
    
    

'''
####################################### Chatglm2-6B ####################################### 
python evaluate_ape210k.py --base_output_path base_dir --ckpt_name chatglm2-6b --eval_file eval_test.ape.jsonl
'''