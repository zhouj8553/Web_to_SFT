import numpy as np
import re
import sys
import json
import copy
import random
random.seed(1)
from tqdm import tqdm

import utils

# \n
def replace_newline_between_numbers(input_string):
    # 使用正则表达式查找两个数字之间的换行符，并将其替换为斜杠
    pattern = r'(\d+)\n(\d+)'
    replaced_string = re.sub(pattern, r'\1/\3', input_string)
    return replaced_string

#\\n
def replace_multiple_newlines_between_numbers(input_string):
    # 使用正则表达式查找两个数字之间的一个或多个连续的\\n，并将其替换为/
    pattern = r'(\d)(\\n)+(\d)'
    replaced_string = re.sub(pattern, r'\1/\3', input_string)
    return replaced_string
# 多\\n换1
def remove_multiple_newlines(input_string):
    # 使用正则表达式将一个或多个连续的\\n替换为一个\\n
    pattern = r'(\\n)+'
    replaced_string = re.sub(pattern, '\\n', input_string)
    return replaced_string

def clean_division(text):
    text = replace_multiple_newlines_between_numbers(text)
    text = remove_multiple_newlines(text)
    return text 
    
def clean_formula(analysis):
    analysis = analysis.replace("\xa0"," ").strip()
    orig_analysis = copy.deepcopy(analysis)
    pattern = r'，(=|≈)'
    matches = [x for x in re.finditer(pattern, analysis)]  
    pattern = r'， (=|≈)'
    matches2 = [x for x in re.finditer(pattern, analysis)]
    matches = matches + matches2 

    # match = re.find(pattern, analysis)
    now_pos = 0
    new_analysis = ""
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        if analysis[start_pos-1] not in ['<','>','=',"≈","＜","＞"]:
            new_analysis += analysis[now_pos:start_pos]+analysis[end_pos-1]
            now_pos = end_pos
    new_analysis += analysis[now_pos:]
    analysis = new_analysis
    return analysis

def clean_analysis_sp_pattern1(analysis):
    # "此题主要考查 xxx"，前一半是solvestep，后一半是analysis
    splited_analysis = analysis.split("此题主要考查")
    return {"extracted_solvestep":splited_analysis[0], "extracted_analysis":"此题主要考查"+splited_analysis[1],"merged_extracted_solvestep":splited_analysis[0]}


def clean_analysis_sp_pattern2(merged_extracted_solvestep):
    # 如果抽取出来的结果里“故答案为：”出现在"据此解答" or "直接得出结论即可" or "本题考查" or "此题考查" or "据此判断" 之前，则意味着有solvestep和analysis的混合，从“故答案为”后接的第一个“。/.”开始进行断句，且保证断句后的第一个字符不是数字（防止切断小数）。
    if "故答案为" not in merged_extracted_solvestep: return merged_extracted_solvestep
    tmp = merged_extracted_solvestep.split("故答案为")[1]
    if "据此解答" in tmp or "直接得出结论即可" in tmp or "本题考查" in tmp or "此题考查" in tmp or "据此判断" in tmp:
        index = merged_extracted_solvestep.index("故答案为")
        for i in range(index, len(merged_extracted_solvestep)):
            if merged_extracted_solvestep[i]=='。': 
                merged_extracted_solvestep = merged_extracted_solvestep[:i+1]
                if "本题考查" in merged_extracted_solvestep: merged_extracted_solvestep = merged_extracted_solvestep.split("本题考查")[0]
                return merged_extracted_solvestep
            if merged_extracted_solvestep[i]=='．' and i+1<len(merged_extracted_solvestep) and merged_extracted_solvestep[i+1].isdigit()==False: 
                merged_extracted_solvestep = merged_extracted_solvestep[:i+1]
                if "本题考查" in merged_extracted_solvestep: merged_extracted_solvestep = merged_extracted_solvestep.split("本题考查")[0]
                return merged_extracted_solvestep
            if merged_extracted_solvestep[i]=='.' and i+1<len(merged_extracted_solvestep) and merged_extracted_solvestep[i+1].isdigit()==False: 
                merged_extracted_solvestep = merged_extracted_solvestep[:i+1]
                if "本题考查" in merged_extracted_solvestep: merged_extracted_solvestep = merged_extracted_solvestep.split("本题考查")[0]
                return merged_extracted_solvestep
    return merged_extracted_solvestep

def clean_analysis_sp_pattern3(merged_extracted_solvestep):
    # 去除solvestep最前面的“【】”或者“[]”
    while merged_extracted_solvestep.startswith("[") and "]" in merged_extracted_solvestep:
        index = merged_extracted_solvestep.index("]")
        merged_extracted_solvestep = merged_extracted_solvestep[index+1:]
    while merged_extracted_solvestep.startswith("【") and "】" in merged_extracted_solvestep:
        index = merged_extracted_solvestep.index("】")
        merged_extracted_solvestep = merged_extracted_solvestep[index+1:]
    return merged_extracted_solvestep


def clean_analysis(analysis):
    # 从raw_analysis中提取solvestep
    analysis = analysis.replace("\xa0"," ").strip()
    analysis_keywords = ["【分析】", "[分析]", "试题分析：","试题分析:", "分析：","分析:",'【思路点拨】']
    solvestep_keywords = ["【详解】", "[详解]", "试题解析：", "试题解析:", "解答：", "解答:","解:","解：","【解答】","[解答]",'[题目详解]']
    comments_keywords = ["【点睛】","[点睛]", "点评：","点评:","[点评]","【点评】","【剖析】","[剖析]",'【解题思路】',"\n点睛:"]
    if ("此题主要考查" in analysis):
        flag = True
        for keyword in analysis_keywords + solvestep_keywords + comments_keywords:
            if keyword in analysis: flag = False
        if analysis.startswith("此题主要考查"): falg = False
        if len(analysis.split("此题主要考查")[0])<10: flag = False
        if flag == True: return clean_analysis_sp_pattern1(analysis)

    
    tmp = ""
    now_type = "extracted_default"
    ret_ans = {}
    for ch in analysis:
        for keyword in analysis_keywords:
            if tmp.endswith(keyword) and (tmp.endswith("根据分析:")==False and tmp.endswith("根据分析：")==False and tmp.endswith("根据以上分析:")==False and tmp.endswith("根据以上分析：")==False and tmp.endswith("经分析:")==False and tmp.endswith("经分析：")==False and tmp.endswith("根据题干分析:")==False and tmp.endswith("根据题干分析：")==False and tmp.endswith("依照分析:")==False and tmp.endswith("依照分析：")==False  and tmp.endswith("由分析:")==False and tmp.endswith("由分析：")==False and tmp.endswith("由题目分析:")==False and tmp.endswith("由题目分析：")==False and tmp.endswith("结合分析:")==False and tmp.endswith("结合分析：")==False and tmp.endswith("步分析:")==False and tmp.endswith("步分析：")==False and tmp.endswith("进行分析:")==False and tmp.endswith("进行分析：")==False):
                if len(tmp)>10:
                    ret_ans[now_type] = tmp.rstrip(keyword)
                    tmp = ""
                    now_type = "extracted_analysis"
        for keyword in solvestep_keywords:
            if tmp.endswith(keyword): 
                tmp = tmp.rstrip(keyword)
                ret_ans[now_type] = tmp
                tmp = ""
                if keyword in ["解:","解："]: 
                    tmp += keyword
                now_type = "extracted_solvestep"
        for keyword in comments_keywords:
            if tmp.endswith(keyword): 
                ret_ans[now_type] = tmp.rstrip(keyword)
                tmp = ""  
                now_type = "extracted_comments"              
        tmp += ch
    ret_ans[now_type] = tmp
    if "extracted_default" in ret_ans and len(ret_ans["extracted_default"])<=5:
        del ret_ans["extracted_default"]
    for (x,y) in ret_ans.items():
        y = y.strip()
    merged_solvestep = ret_ans.get("extracted_solvestep","")
    if len(merged_solvestep)==0: merged_solvestep = ret_ans.get("extracted_default","")
    if len(merged_solvestep)==0: merged_solvestep = ret_ans.get("extracted_analysis","")
    if len(merged_solvestep)==0: merged_solvestep = ret_ans.get("extracted_comments","")
    if len(merged_solvestep)<10: merged_solvestep = analysis
    tmp = copy.deepcopy(merged_solvestep)
    ret_ans['merged_extracted_solvestep'] = clean_analysis_sp_pattern2(merged_solvestep)
    ret_ans['merged_extracted_solvestep'] = clean_analysis_sp_pattern3(ret_ans['merged_extracted_solvestep'])
    return ret_ans


def clean_examples_with_rules():
    examples = utils.read_jsonl("../data/filtered_web_train_data.jsonl")
    for e in examples:
        # new_q = replace_multiple_newlines_between_numbers(e['question'])
        # new_ans = replace_multiple_newlines_between_numbers(e['answer'])
        # new_raw_answer = replace_multiple_newlines_between_numbers(e['raw_answer'])
        analysis = replace_multiple_newlines_between_numbers(e['raw_analysis'])
        analysis = analysis.replace("\xa0"," ").strip()
        orig_analysis = copy.deepcopy(analysis)
        pattern = r'，(=|≈)'
        matches = [x for x in re.finditer(pattern, analysis)]  
        pattern = r'， (=|≈)'
        matches2 = [x for x in re.finditer(pattern, analysis)]
        matches = matches + matches2 

        # match = re.find(pattern, analysis)
        now_pos = 0
        new_analysis = ""
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            if analysis[start_pos-1] not in ['<','>','=',"≈","＜","＞"]:
                new_analysis += analysis[now_pos:start_pos]+analysis[end_pos-1]
                now_pos = end_pos
        new_analysis += analysis[now_pos:]
        e['rule_processed_analysis'] = new_analysis
        web_solvestep = clean_analysis(remove_multiple_newlines(new_analysis))['merged_extracted_solvestep']
        e['rule_processed_solvestep'] = web_solvestep
    random.shuffle(examples)
    for i in range(3):
        print(examples[i]['rule_processed_solvestep'])
        import pdb 
        pdb.set_trace()
    utils.save_to_jsonl(examples,"../filtered_web_cleaned_analysis_train_data.jsonl")

if __name__ == "__main__":
    clean_examples_with_rules()

# python rule_cleaning.py 


