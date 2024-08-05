from utils import read_jsonl, save_to_jsonl, read_csv

import time
import re  
import string
import random
random.seed(1)
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool


def remove_punctuation_and_newlines(text):
    # print(text)
    # 删除中文标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 删除英文标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 删除换行符
    text = text.replace('\n', '')
    text = text.replace(" ","").replace("\u3000","").replace("frac","").replace("【题文】","")
    text = re.sub(r'[a-zA-Z]{2,}', '', text) 
    return text

def remove_punctuation_and_newlines_v2(text):
    text = text.replace("【题文】","")
    text = re.sub(r'[a-zA-Z]{2,}', '', text) 
    text = re.sub(r'[^\u4e00-\u9fa50-9a-zA-Z]', '', text)
    return text


def remove_multiple_newlines(input_string):
    # 使用正则表达式将一个或多个连续的\\n替换为一个\\n
    pattern = r'(\\n)+'
    replaced_string = re.sub(pattern, '\\n', input_string)
    return replaced_string

def find_deduplicate_ids(questions, examples2, key2):
    pairwise_keys = []
    for e in tqdm(examples2):
        question = remove_punctuation_and_newlines(remove_multiple_newlines(e[key2]))
        for highquality_idx,q in enumerate(questions):
            if question == q:
                pairwise_keys.append({"highquality_idx":highquality_idx, "web57w_idx":e['web57w_idx']})
                break 
    return pairwise_keys


def get_web_highquality_pair_idxs_single(packed_input):
    def is_subsequence(A, B):  
        a_index, b_index = 0, 0  
        while a_index < len(A) and b_index < len(B):  
            if A[a_index] == B[b_index]:  
                a_index += 1  
            b_index += 1  
        return a_index == len(A)

    def counter_diff(sentence_a, sentence_b):  
        # 将句子分割成单词列表  
        words_a = [x for x in sentence_a] 
        words_b = [x for x in sentence_b]
        
        # 使用Counter计算每个句子中单词的频率  
        counter_a = Counter(words_a)  
        counter_b = Counter(words_b)  
        
        # 找出A句子有而B句子没有的单词及其频率  
        diff_a_b = counter_a - counter_b  
        
        # 找出B句子有而A句子没有的单词及其频率  
        diff_b_a = counter_b - counter_a
        
        # 返回两个差异字典  
        return diff_a_b, diff_b_a 
    [web_train_examples, highquality_train_examples] = packed_input
    highquality_train_questions = [remove_punctuation_and_newlines_v2(e['input'][0]) for e in highquality_train_examples]
    highquality_train_solvesteps = [remove_punctuation_and_newlines_v2(e['solve_step'][0]) for e in highquality_train_examples]
    pairwise_keys = []
    for web57w_idx in tqdm(range(len(web_train_examples))):
        e = web_train_examples[web57w_idx]
        question = remove_punctuation_and_newlines_v2(remove_multiple_newlines(e['question']))
        answer = remove_punctuation_and_newlines_v2(e['raw_analysis'])
        for highquality_idx, solve_step in enumerate(highquality_train_solvesteps):
            question_diff_num = 10000
            flag2 = 0
            flag1 = (question == highquality_train_questions[highquality_idx])
            if flag1==0:
                question_diff_num = abs(len(question)-len(highquality_train_questions[highquality_idx])) #剪枝
                if question_diff_num > 5: continue
                diff_a_b, diff_b_a = counter_diff(question, highquality_train_questions[highquality_idx])  
                question_diff_num = sum([x[1] for x in diff_a_b.items()]) + sum([x[1] for x in diff_b_a.items()]) #real question_diff_num
                if question_diff_num <= 5:
                    flag2 = is_subsequence(solve_step, answer)
            if flag1 ==1 or (flag2 ==1 and question_diff_num<=5):
                pairwise_keys.append({"highquality_idx":highquality_idx, "web57w_idx":e['web57w_idx']})
                break
    return pairwise_keys

def get_web_highquality_pair_idxs():
    highquality_train_examples = read_jsonl("../data/high_precision_train_data.jsonl")
    web_train_examples = read_jsonl("../data/filtered_web_train_data.jsonl")
    for i,e in enumerate(web_train_examples):
        e['web57w_idx'] = i

    pool = Pool(processes=115)
    batch = 5000
    tmp_examples = pool.map(get_web_highquality_pair_idxs_single, [[web_train_examples[i*batch:(i+1)*batch],highquality_train_examples] for i in range(len(web_train_examples)//batch+1)])
    pool.close()
    pairwise_keys = []
    for x in tmp_examples:
        pairwise_keys += x
    import pdb 
    pdb.set_trace()
    save_to_jsonl(pairwise_keys,"../data/web_highquality_pairwise_index.jsonl")


def prepare_pairwise_sft_data():
    pairwise_map = read_jsonl("../data/web_highquality_pairwise_index.jsonl")
    highquality_train_examples = read_jsonl("../data/high_precision_train_data.jsonl")
    web_train_examples = read_jsonl("../data/filtered_web_train_data.jsonl") 
    converter_examples = []
    prompt = "假设你是一个小学数学老师，下面给你一道可能存在语言不规范的题目和对应的答案，请将题目和答案转换成规范格式。\n注意答案只需要保留具体解答步骤，且不要改变原答案的解题思路。\n如果题目非中文数学题，请指出“这不是一道中文数学题。”。如果存在严重的语法错误导致理解困难，请输出“存在语法错误。”。\n\n[题目]\n{}\n\n[答案]\n{}"
    for pair in pairwise_map:
        highquality_idx = pair['highquality_idx']
        web57w_idx = pair['web57w_idx']
        highquality_question = highquality_train_examples[highquality_idx]['input'][0].replace("\u3000"," ")
        highquality_answer = highquality_train_examples[highquality_idx]['solve_step'][0].replace("\u3000"," ")
        web_question = web_train_examples[web57w_idx]['question']
        web_answer = web_train_examples[web57w_idx]['raw_analysis']
        converter_examples.append({"instruction": prompt.format(web_question, web_answer), "output": "[问题]\n{}\n\n[答案]\n{}".format(highquality_question, highquality_answer)})
    save_to_jsonl(converter_examples, "../data/clean_train_data.jsonl")


if __name__ == "__main__":
    get_web_highquality_pair_idxs()
    prepare_pairwise_sft_data()

# python process_pairwise_data.py
