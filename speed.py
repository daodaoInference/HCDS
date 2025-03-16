import json
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import heapq

tokenizer=AutoTokenizer.from_pretrained("/home/zlb/下载/model/llama-2-7B-chat-hf")
jsonl_file = "/home/zlb/project/HCSD/mt_bench_1/home/zlb/project/HCSD/a800_layer_1_w1_silu_norm/ess-llama-2-chat-7b-fp32-baseline_mt_bench_1_write_quantize_eagle_humaneval_6_7_4_10_chat_last-temperature-0.0.jsonl"
# jsonl_file_base = "/home/zlb/project/EAGLE-train/humaneval_1/home/zlb/project/EAGLE-train/a800_layer_1_norm_0/ess-llama-2-chat-7b-fp32-baseline_humaneval_only_cpu_6__write_quantize_new-temperature-0.0.jsonl"
jsonl_file_base = "/home/zlb/project/HCSD/mt_bench_1/home/zlb/project/HCSD/a800_new/ess-llama-2-chat-7b-fp32-baseline_mt_bench_1_only_cpu_6__write_quantize_new_last-temperature-0.0.jsonl"
print("/home/zlb/project/HCSD/mt_bench_1/home/zlb/project/HCSD/a800_layer_1_w1_silu_norm/ess-llama-2-chat-7b-fp32-baseline_mt_bench_1_write_quantize_eagle_humaneval_6_7_4_10_chat_last-temperature-0.0.jsonl")
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



ids_total=0
ids_token_total=0


speeds=[]
speeds1=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    ids=sum(datapoint["choices"][0]['idxs'])
    ids_total += ids
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    ids_token_total += tokens
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)
    speeds1.append(datapoint["choices"][0]['new_tokens'][0]/datapoint["choices"][0]['wall_time'][0])
print('accept',ids_token_total/ids_total)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)




total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



print('speeds1',np.array(speeds1).mean())
print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())




