import os
import argparse
from cmath_utils import *


def evaluate(respoinse_dir):
    """
    读取一个jsonl文件中的模型生成结果和标准答案，计算模型准确率。

    Args:
        response_file: str, 包含模型输出内容的jsonl文件路径，每行的json必须包含golden, response字段
    """
    response_file = os.path.join(respoinse_dir, 'eval_cmath_dev_instruction.jsonl')
    if "chatglm" in respoinse_dir:
        output_key = "chatglm2_cot_output"
    else: output_key = "output"
    d = read_jsonl_keys(response_file, [output_key, "golden", "reasoning_step"])

    hit = 0    # 计数：正确
    err = 0    # 计数：模型回复异常
    warn = 0   # 计数：未能从模型回复中提取数字
    # import pdb 
    # pdb.set_trace()
    grade_results = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    idx = 0
    for g, r in zip(d["golden"], d[output_key]):
        predict = extract_digits_prediction(r)
        if len(predict) == 0:
            warn += 1

        if "ERROR" in predict:
            err += 1

        if match_digit_response(g, predict):
            hit += 1
            grade_results[idx//100+1] += 1
            # grade_results[d['reasoning_step'][idx]] += 1
        idx += 1
    # import pdb 
    # pdb.set_trace()
    sample_size = len(d["golden"])
    valid = sample_size - err
    acc = hit / valid
    print("{}/{}".format(hit, valid))
    print("acc={0:.1%}".format(acc))
    print(grade_results)
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_dir")
    args = parser.parse_args()
    evaluate(args.response_dir)

