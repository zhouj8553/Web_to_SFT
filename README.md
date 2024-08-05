# Leaveraging Web-Crawled Data for High-Quality Fine-Tuning

This repository is the official code for the paper "Leveraging Web-Crawled Data for High-Quality Fine-Tuning".


## Introduction

We argue that although the web-crawled data often has formatting errors causing semantic inaccuracies, it can still serve as a valuable source for high-quality supervised fine-tuning in specific domains without relying on advanced models like GPT-4.

To this end, we propose to create a paired training dataset automatically by aligning web-crawled data with a smaller set of high-quality data. 
By training a language model on this dataset, we can convert web data with irregular formats into high-quality ones. Our experiments demonstrate the effectiveness of our method.

Our method involves the following four steps as shown in Figure below:
- Constructing format converter training data by pairing web-crawled data with high-quality data using fuzzy matching.
- Train an LLM with the constructed data to enable it to transform raw web-crawled examples into high-quality examples. 
- Use the trained LLM to convert the web-crawled data into high-quality format.
- Train another LLM (same initialization as that of step 2) to solve mathematical problems using both the high-quality data and the converted web-crawled data.

![image](https://github.com/zhouj8553/Web_to_SFT/blob/main/imgs/model_structure.png)


&nbsp;
## Quickstart
To use our code, you could install the requirements by running:
```bash
pip install -r requirements.txt
```

[Note]: For Qwen1.5-7B-Chat, the "transformers" version is 4.41.0, and for ChatGLM2-6B, the "transformers" version is 4.33.1.

All our fine-tuning process are conducted on 8 NVIDIA A800 GPU with 80GB memory.

&nbsp;
## Method

### Step1: generate paired data
Our code for generating paired data is in `process_pairwise_data.py`. In this file, function `get_web_highquality_pair_idxs` generates paired indexs by matching web-crawled data with high-quality data and the fuction `prepare_pairwise_sft_data` converts paired data into SFT format.

### Step2: finetune the language model using paired data

All our scripts for model training are in the folder `model_train`.

You can finetune the model by running the following scripts:
```bash
cd model_train/chatglm2_model

bash train_chatglm2.sh <dataset_file_name> <input_key> <output_key> ../../../checkpoint chatglm2-6b > myout.file 2>&1 &

```

```bash
cd model_train/qwen15_model

bash finetune.sh <dataset_file_name> <input_key> <output_key> ../../../checkpoint Qwen1.5-7B-Chat
```


In the above instance, the <dataset_file_name> denotes the model, the <input_key> denotes the input key of the data, and <output_key> denotes the output key of the data.


### Step3: generated cleaned web-crawled data
We first generate model output using the finetuned model with the code in the folder `model_eval`. Our inference codes run on a single GPU with batch size 1. A method to speed this up is to divide the data into equivalent parts and process them in parallel.

Suppose the model checkpoint is in <model_ckpt> and the file to be generated is <web_crawled_file>.

If you want to inference with ChatGLM2, run the following scripts:

```bash
cd model_eval

CUDA_VISIBLE_DEVICES=0 python evaluate_chatglm2_singlegpu.py --model_path <model_ckpt> --batch_size 1 --test_file <web_crawled_file> --output_dir ../../data/chatglm2_generated 
```

If you want to inference with Qwen1.5-7B-Chat, run the following scripts:

```bash
cd model_eval

CUDA_VISIBLE_DEVICES=0 python evaluate_qwen15_singlegpu.py --model_path <model_ckpt> --batch_size 1 --test_file <web_crawled_file> --output_dir ../../data/qwen_generated/
```


Afterwards, we extract the output and convert it into the SFT format using ``postprocessing_generated_data.py''. In this file, we provide the function to convert the output into SFT format.


### step4: finetune the math language model and evaluate the model performance
The script of finetuning is the same as that in step 2, while the evaluation script is the same as that in step 3. 


After we get the predicted results, we could run the following scripts to evaluate the model performance.

Evaluation of APE210k
```bash
python evaluate_ape210k.py --base_output_path <base_dir> --ckpt_name <model_name> --eval_file eval_test.ape.jsonl
```

Evaluation of CMATH
```bash
python cmath_eval.py --response_dir <your file path>

```

&nbsp;
## Rule based baseline

We alse provide a rule based baseline for cleaning the web-crawled data, which is really a strong baseline.

```
python rule_cleaning.py 
```

<!-- ## Citation
Please cite our paper if you use the code in your work.
```bibtex


``` -->
