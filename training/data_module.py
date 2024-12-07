import torch
from torch import nn
from torch.utils.data import Dataset
import yaml
import pandas as pd

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.pad_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(min(num_question_tokens,max_length)): 
        label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


def get_model_identifiers_from_yaml(path="model_config.yaml"):
    model_configs  = {}
    with open(path, "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs['alpaca-7b']


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512 ):
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.forget_df = pd.read_csv(data_path+'violations_train.csv', index_col=0)
        self.retain_df = pd.read_csv(data_path+'safe_train.csv', index_col=0)       

        self.model_configs = get_model_identifiers_from_yaml()
        

    def __len__(self):
        return len(self.forget_df)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["forget", "alternate", "retain"]:
            
            data = self.forget_df if data_type != "retain" else self.retain_df
            
            question = data.iloc[idx]['prompt']

            if data_type == "alternate":
                for i in range(3):
                    column_name = 'alternate_response'+str(i+1)
                    answer = data.iloc[idx][column_name]
                    converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                    rets.append(converted_data)

            elif data_type == "forget":
                answer = data.iloc[idx]['response']
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)

            else:          
                answer = data.iloc[idx]['response']
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)

        return rets