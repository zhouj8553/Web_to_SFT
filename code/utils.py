import numpy as np
from scipy.stats import spearmanr

import os
import json
import pickle
import pandas as pd

################################ read files ################################
def read_jsonl(file_path):
    json_list = []
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            e = json.loads(line.strip())
            # e = eval(line.strip())
            json_list.append(e)
    return json_list

def read_json(file_path):
    return json.load(open(file_path,'r'),encoding = "utf8")

def read_csv(file_path):
    data = pd.read_csv(file_path,index_col=0,encoding='utf-8')
    json_list = []
    for row_name,item in data.iterrows():
        e = {key: item[key] for key in data.columns}
        json_list.append(e)
    return json_list

def read_list(list_path):
    with open(list_path, 'rb') as f:  
        loaded_list = pickle.load(f) 
    return loaded_list

def read_singleline_set(file_path, is_fix_string=False):
    out_set = set()
    with open(file_path, 'r') as fin:
        for line in fin:
            try:
                if is_fix_string:
                    out_set.add(eval(line.strip()))
                else:
                    out_set.add(line.strip())
            except:
                continue
    return out_set



READ_FUNC = {
    "jsonl":read_jsonl,
    "json":read_json,
    "csv":read_csv,
    "list": read_list
}

################################ save files ################################

def save_to_jsonl(json_list,new_file_path):
    with open(new_file_path, 'w') as f:  
        for item in json_list:  
            f.write(json.dumps(item,ensure_ascii=False) + '\n')

def save_to_json(json_list,new_file_path):
    json.dump(json_list,open(new_file_path,'w'),ensure_ascii=False)

def save_to_csv(json_list,new_file_path):
    key_names = [x for x in json_list[0]]
    new_dict = {}
    for key_name in key_names:
        new_dict[key_name] = [x.get(key_name,"UNK") for x in json_list]
    data = pd.DataFrame(new_dict)
    data.to_csv(new_file_path,encoding='utf_8_sig')

def save_list(my_list, list_path):
    with open(list_path, 'wb') as f:
        pickle.dump(my_list, f)

WRITE_FUNC = {
    'jsonl': save_to_jsonl,
    'json': save_to_json,
    'csv': save_to_csv,
    "list": read_list,
}

################################ utils ################################
def transform_key_name(json_list, in_col_list, out_col_list):
    for item in json_list:
        for (in_key, out_key) in zip(in_col_list,out_col_list):
            item[out_key] = item.pop(in_key)
    return json_list


################################ mathematics ################################
# show the distribution of a list
def show_list_distribution(nums):
    print([min(nums)]+[np.percentile(nums,i*10) for i in range(1,10)]+[max(nums)])

def calculate_spearmanr_correlation(x,y):
    corr, p_value = spearmanr(x, y)
    # print("Spearman correlation coefficient:", corr)
    # print("p-value:", p_value)
    return corr, p_value

def get_last_number(sentence):
    digit = ''
    in_digit = False
    for ch in sentence[::-1]:
        if ch >= '0' and ch <='9':
            digit += ch
            in_digit = True
        else:
            if in_digit == True:
                if ch == '.': 
                    if '.' not in digit: digit += ch
                    else: break
                elif ch == ',': continue
                else: break
            else: 
                if len(digit)==0: continue
    if len(digit)==0: return None
    if '.' not in digit and len(digit)!=1 and digit.endswith('0'): 
        digit = digit.rstrip('0')
    if len(digit)==0: return None
    # print(sentence, digit)
    return eval(digit[::-1])

def merge_predictions(examples_list, model_name_list, register_cols = ["output","is_correct"]):
    new_examples = []
    for i in range(len(examples_list)):
        examples = examples_list[i]
        model_name = model_name_list[i]
        for (e_idx,e) in enumerate(examples):
            if i==0:
                new_examples.append({x:y for (x,y) in e.items() if x not in register_cols})
            for col in register_cols:
                new_examples[e_idx]["{}_{}".format(model_name,col)] = e[col]
    return new_examples

def is_chinese(char):
	if '\u4e00' <= char <= '\u9fff' or (char >= '\u3000' and char <= '\u303F'):
		return True
	else:
		return False

if __name__ == "__main__":
    sentence = "sdafsdalf 2341,234"
    print(get_last_number(sentence))

# python utils.py
