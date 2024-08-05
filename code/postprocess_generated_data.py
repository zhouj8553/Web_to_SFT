import math
import os
from utils import read_jsonl, save_to_jsonl

def convert_generated_data_to_sft_data(input_file_path, output_file_path=None, output_column="chatglm2_cot_output", num_limit = -1):
    examples = read_jsonl(input_file_path)
    if num_limit != -1:
        examples = examples[:num_limit]

    ret_examples = []
    for e in examples:
        if "存在语法错误。" in e[output_column] or "这不是一道中文数学题" in e[output_column]:
            continue
        if "[问题]" not in e[output_column] or "\n\n[答案]\n" not in e[output_column]:
            continue
        processed_generated_question = e[output_column].split("[问题]")[1].split("\n\n[答案]\n")[0]
        processed_generated_answer = e[output_column].split("\n\n[答案]\n")[1]
        ret_examples.append({"question": e['question'], "raw_analysis": e['raw_analysis'], "rule_processed_analysis": e["rule_processed_analysis"], "rule_processed_solvestep": e["rule_processed_solvestep"], "dasou57w_idx":e['dasou57w_idx'], "instruction": processed_generated_question, "output": processed_generated_answer})
    if output_file_path is None:
        output_file_path = input_file_path.replace(".jsonl","_{}_processed.jsonl".format(num_limit))
    save_to_jsonl(ret_examples, output_file_path)

def merge_and_convert_generated_data_to_sft_data(input_file_path, output_file_path=None, output_column="chatglm2_cot_output", num_limit = -1):
    assert num_limit>0
    total_examples = []
    for i in range(math.ceil(num_limit/20000)):
        tmp_examples = read_jsonl(os.path.join(input_file_path, "eval_filtered_dasou_cleaned_analysis_train_data_{}-{}.jsonl".format(i*20000,(i+1)*20000)))
        if i==28:
            assert len(tmp_examples)==13960
        else:
            assert len(tmp_examples)==20000
        total_examples += tmp_examples
    total_examples = total_examples[:num_limit]
    if output_file_path is None:
        output_file_path = os.path.join(input_file_path,"eval_filtered_dasou_cleaned_analysis_train_data_0-{}_processed.jsonl".format(num_limit))
    ret_examples = []
    for e in total_examples:
        if "存在语法错误。" in e[output_column] or "这不是一道中文数学题" in e[output_column]:
            continue
        if "[问题]" not in e[output_column] or "\n\n[答案]\n" not in e[output_column]:
            continue
        processed_generated_question = e[output_column].split("[问题]")[1].split("\n\n[答案]\n")[0]
        processed_generated_answer = e[output_column].split("\n\n[答案]\n")[1]
        ret_examples.append({"question": e['question'], "raw_analysis": e['raw_analysis'], "rule_processed_analysis": e["rule_processed_analysis"], "rule_processed_solvestep": e["rule_processed_solvestep"], "dasou57w_idx":e['dasou57w_idx'], "instruction": processed_generated_question.strip(), "output": processed_generated_answer.strip()})
    print(len(ret_examples))
    for e in ret_examples[-3:]:
        print(e)
    save_to_jsonl(ret_examples, output_file_path)
    
                                
