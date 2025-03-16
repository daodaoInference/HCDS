import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("/gf3/home/lzq/model/Llama-2-7b-chat-hf/")
jsonl_file = "/gf3/home/zlb/HCSD+gpu_cpu_offload+last/humaneval/gf3/home/zlb/HCSD+gpu_cpu_offload+last/partial-offload+SD/output_16-7-10-humaneval_last_10_last_2-temperature-0.0.jsonl"
# jsonl_file ="/gf3/home/lzq/EAGLE-2080/EAGLE/mt_bench/ess-llama-2-chat-7b-fp16-cpu-temperature-0.0.jsonl"
# jsonl_file_base = "/gf3/home/zlb/HCSD+gpu_cpu_offload+last/humaneval/gf3/home/zlb/HCSD+gpu_cpu_offload+last/hf_offload/output_16-7-10-humaneval_last_1-temperature-0.0.jsonl"
jsonl_file_base = "/gf3/home/zlb/HCSD+gpu_cpu_offload+last/humaneval/gf3/home/zlb/HCSD+gpu_cpu_offload+last/CPU-only/output_16-7-10-humaneval_last-temperature-0.0.jsonl"

jsonl_file_baseddd = "/gf3/home/zlb/EAGLE-train/humaneval/humaneval_gn8_onlycpu-24-qint8-temperature-0.0.jsonl,/gf3/home/zlb/EAGLE-train/humaneval/a800_layer_1_norm_0_output_16-7-10_humaneval_gn8-cpu-int8-gpu-bfloat16-gn8-2-temperature-0.0.jsonl"
print(jsonl_file)
print(jsonl_file_base)
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

