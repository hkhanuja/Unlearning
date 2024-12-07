import os
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer)

from config import METHODS_DATA_DIR, FINAL_DATA_DIR
from SafeRLHF.config import CLS_MODELS_DIR, TRANSFORMATION_MODELS_DIR
from SafeRLHF.training.config import CLS_MAX_LENGTH, TRANSFORMATION_MAX_LENGTH
from SafeRLHF.training.classification import LSTMClassifier
from SafeRLHF.training.transformation import TransformerSeq2Seq


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASIC_RESPONSE: str = 'I\'m sorry, but I cannot provide a response to this query. I apologize for any inconvenience.'

CAT_NAMES: str = ['Privacy Violation']
CLS_MODELS = {}
TRANSFORMATION_MODELS = {}

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
for cat_name in CAT_NAMES:
    CLS_MODELS[cat_name] = LSTMClassifier(TOKENIZER.vocab_size)
    CLS_MODELS[cat_name].load_state_dict(torch.load(
        os.path.join(CLS_MODELS_DIR, f'{cat_name}.pth'),
        map_location=DEVICE)
    )
    CLS_MODELS[cat_name].to(DEVICE)
    CLS_MODELS[cat_name].eval()

    TRANSFORMATION_MODELS[cat_name] = TransformerSeq2Seq(TOKENIZER.vocab_size)
    TRANSFORMATION_MODELS[cat_name].load_state_dict(torch.load(
        os.path.join(TRANSFORMATION_MODELS_DIR, f'{cat_name}.pth'),
        map_location=DEVICE)
    )
    TRANSFORMATION_MODELS[cat_name].to(DEVICE)
    TRANSFORMATION_MODELS[cat_name].eval()

# PERPLEXITY_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
# PERPLEXITY_MODEL = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
# PERPLEXITY_MODEL.eval()

# PERPLEXITY_MODEL = AutoModelForCausalLM.from_pretrained(
#     'PKU-Alignment/alpaca-7b-reproduced',
#     torch_dtype=torch.bfloat16, device_map='auto').to(DEVICE)
# PERPLEXITY_MODEL.eval()
# PERPLEXITY_TOKENIZER = AutoTokenizer.from_pretrained(
#     'PKU-Alignment/alpaca-7b-reproduced')

COHESION_TOKENIZER = AutoTokenizer.from_pretrained(
    'textattack/bert-base-uncased-CoLA')
COHESION_MODEL = AutoModelForSequenceClassification.from_pretrained(
    'textattack/bert-base-uncased-CoLA')
COHESION_MODEL.eval()


def get_cls_response(model, prompt: str, response: str):
    encoding = TOKENIZER(
        prompt + " " + response,
        truncation=True,
        padding='max_length',
        max_length=CLS_MAX_LENGTH,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    return output


def get_transformation_response(model, unsafe_response: str):
    # Tokenize the input text
    src_encoding = TOKENIZER(
        unsafe_response,
        truncation=True,
        padding='max_length',
        max_length=TRANSFORMATION_MAX_LENGTH,
        return_tensors='pt'
    )
    src_input_ids = src_encoding['input_ids'].to(DEVICE)  # Shape: (1, seq_len)
    src_attention_mask = src_encoding['attention_mask'].to(
        DEVICE).bool()  # Shape: (1, seq_len)

    # Transpose src_attention_mask to match (seq_len, batch_size)
    src_attention_mask = src_attention_mask.transpose(
        0, 1)  # Shape: (seq_len, 1)

    # Embedding with positional encoding
    with torch.no_grad():
        src_embedded = model.embedding(
            src_input_ids) + model.positional_encoding[:, :src_input_ids.size(1), :]
        # Transpose embeddings to match (seq_len, batch_size, embedding_dim)
        # Shape: (seq_len, 1, embedding_dim)
        src_embedded = src_embedded.transpose(0, 1)

        # Encode the source sequence
        memory = model.encoder(
            src_embedded, src_key_padding_mask=src_attention_mask.T)

    # Initialize target sequence with start token ([CLS])
    start_token_id = TOKENIZER.cls_token_id  # Typically 101
    end_token_id = TOKENIZER.sep_token_id    # Typically 102

    tgt_input_ids = torch.tensor(
        [[start_token_id]], device=DEVICE)  # Shape: (1, 1)

    for _ in range(TRANSFORMATION_MAX_LENGTH):
        with torch.no_grad():
            tgt_embedded = model.embedding(
                tgt_input_ids) + model.positional_encoding[:, :tgt_input_ids.size(1), :]
            # Transpose embeddings to match (seq_len, batch_size, embedding_dim)
            # Shape: (tgt_seq_len, 1, embedding_dim)
            tgt_embedded = tgt_embedded.transpose(0, 1)

            # No need for tgt_key_padding_mask during inference in this context
            output = model.decoder(tgt_embedded, memory)
            logits = model.fc_out(output)
            # Get the logits for the last token in the sequence
            next_token_logits = logits[-1, 0, :]  # Shape: (vocab_size)
            # Greedy decoding: pick the token with the highest probability
            next_token_id = torch.argmax(
                next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
            # Append the predicted token to the target sequence
            tgt_input_ids = torch.cat([tgt_input_ids, next_token_id], dim=1)
            if next_token_id.item() == end_token_id:
                break

    # Decode the generated tokens into text
    output_tokens = tgt_input_ids.squeeze().tolist()
    output_tokens = output_tokens[1:]  # Remove the start token
    if end_token_id in output_tokens:
        end_index = output_tokens.index(end_token_id)
        output_tokens = output_tokens[:end_index]

    output_text = TOKENIZER.decode(output_tokens, skip_special_tokens=True)
    return output_text


# def calculate_perplexity(response: str) -> float:
#     tokens = PERPLEXITY_TOKENIZER.encode(
#         response, return_tensors='pt').to(DEVICE)
#     with torch.no_grad():
#         outputs = PERPLEXITY_MODEL(tokens, labels=tokens)
#         loss = outputs.loss
#         perplexity = torch.exp(loss).item()
#     return perplexity


def get_acceptability_score(sentence):
    inputs = COHESION_TOKENIZER(sentence, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = COHESION_MODEL(**inputs).logits
    probabilities = F.softmax(logits, dim=1)
    # Probability that the sentence is acceptable
    acceptability_score = probabilities[0][1].item()
    return acceptability_score


def main():
    basic_cohesion = get_acceptability_score(BASIC_RESPONSE)

    methods = [d for d in os.listdir(METHODS_DATA_DIR)
               if os.path.isdir(os.path.join(METHODS_DATA_DIR, d))]

    for method in methods:
        method_data_dir = os.path.join(METHODS_DATA_DIR, method)
        method_data_file_names = [file for file in os.listdir(method_data_dir)
                                  if file.endswith('.csv')]

        for method_data_file_name in method_data_file_names:
            method_data_file = os.path.join(method_data_dir,
                                            method_data_file_name)
            input_data_df = pd.read_csv(method_data_file, index_col=0)
            input_data_df = input_data_df[['prompt', 'response']]
            for cat_name in CAT_NAMES:
                data_df = input_data_df.copy()
                data_df['score'] = 0.0
                data_df['cohesion score'] = 0.0
                data_df['safer response'] = data_df['response']
                data_df['safer score'] = 0.0
                data_df['safer cohesion score'] = 0.0
                data_df['safe response'] = BASIC_RESPONSE
                data_df['safe cohesion score'] = basic_cohesion

                i = 1
                for index, row in data_df.iterrows():
                    print(f'{i} / {len(data_df)}')
                    i += 1
                    score = get_cls_response(CLS_MODELS[cat_name],
                                             row['prompt'],
                                             row['response']).item()
                    data_df.at[index, 'score'] = score
                    print(len(row['response']))
                    cohesion = get_acceptability_score(row['response'])
                    data_df.at[index, 'cohesion score'] = cohesion

                    if score <= 0.5:
                        safer_response = get_transformation_response(
                            TRANSFORMATION_MODELS[cat_name],
                            row['response'])
                        updated_score = get_cls_response(CLS_MODELS[cat_name],
                                                         row['prompt'],
                                                         safer_response).item()
                        cohesion = get_acceptability_score(safer_response)

                        data_df.at[index, 'safer response'] = safer_response
                        data_df.at[index, 'safer score'] = updated_score
                        data_df.at[index,
                                   'safer cohesion score'] = cohesion

                        if updated_score > 0.5 and cohesion > 0.5:
                            data_df.at[index, 'safe response'] = safer_response
                            data_df.at[index,
                                       'safe cohesion score'] = cohesion

                    else:
                        data_df.at[index, 'safer score'] = score
                        data_df.at[index,
                                   'safer cohesion score'] = cohesion
                        data_df.at[index, 'safe response'] = row['response']
                        data_df.at[index,
                                   'safe cohesion score'] = cohesion

                os.makedirs(os.path.join(FINAL_DATA_DIR, method),
                            exist_ok=True)
                data_df.to_csv(os.path.join(FINAL_DATA_DIR,
                                            method,
                                            method_data_file_name),
                               index=False)


if __name__ == '__main__':
    main()
