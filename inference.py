import torch
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

tqdm.pandas()

###### LOAD YOUR MODEL HERE AND TOKENIZER 
model_name = "models/alpaca_rmu_alpha_0" # need to change

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("model loaded!")

####### CHANGE PATH TO WHERE YOUR VIOLATION TEST DATA IS
df_violation_test = pd.read_csv("data/SafeRLHF-corpora/Privacy Violation_test.csv")

print("violation test data loaded!")

df_safe_test = pd.read_csv("data/SafeRLHF-corpora/test_safe.csv", index_col=0)

print("safe test data loaded!")

###### DEFINE PROMPT AND FORMAT FUNCTION
def prompt(question, preprompt=True):
    if preprompt:
        prompt = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'
        input = prompt.format(input=question)
    else:
        input = question

    input_ids = tokenizer.encode(input, return_tensors='pt').cuda()
    output_ids = model.generate(input_ids, max_new_tokens=512)[0]

    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

def format(response):
    match = re.search(r'ASSISTANT:\s*(.*)', response)
    formatted_response = match.group(1).strip() if match else ''
    return formatted_response

###### RUN INFERENCE
df_violation_save = df_violation_test[['prompt']]
df_violation_save['response'] = df_violation_save.progress_apply(lambda row: format(prompt(row['prompt'])), axis=1,result_type='expand')

df_violation_save.to_csv('inference-response-violations-rmu-alpha-0.csv') # need to change

print("violation test inference saved!")


df_safe_save = df_safe_test[['prompt']]
df_safe_save['response'] = df_safe_save.progress_apply(lambda row: format(prompt(row['prompt'])), axis=1,result_type='expand')

df_safe_save.to_csv('inference-response-safe-rmu-alpha-0.csv') # need to change

print("safe test inference saved!")