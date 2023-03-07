import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 50
beam_num = 200
input_max_length = 358
output_max_length = 20
base_tokenizer = 'facebook/bart-base'
device = 'cuda'
model_dir = 'models/SOTitle-bart'
input_file = 'datasets/SOTitle/test.csv'
output_dir = 'predictions/SOTitle'
output_file = output_dir + '/predictions-bs-200.csv'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == '__main__':
    # Load the dataset
    print("start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    input_texts = pd.read_csv(input_file)
    input_texts = input_texts['src']
    total_len = len(input_texts)

    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    torch.manual_seed(0)
    predictions = []

    with torch.no_grad():
        cache = []

        for index, input_text in input_texts.items():
            if index > 0 and index % batch_size == 0:
                print("Progress: {}/{}".format(index, total_len), len(predictions), predictions[-1][:5])

            cache.append(input_text)
            if (index + 1) % batch_size != 0 and index + 1 != len(input_texts):
                continue

            tokenized_text = tokenizer(cache, truncation=True, padding='max_length', max_length=input_max_length,
                                       return_tensors='pt')

            source_ids = tokenized_text['input_ids'].to(device, dtype=torch.long)
            source_mask = tokenized_text['attention_mask'].to(device, dtype=torch.long)

            # beam search decoding 
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=output_max_length,
                num_beams=beam_num,
                num_return_sequences=beam_num,
                early_stopping=True
            )
            # top-p sampling decoding
            # generated_ids = model.generate(
            #     input_ids=source_ids,
            #     attention_mask=source_mask,
            #     do_sample=True,
            #     max_length=output_max_length,
            #     top_p=0.8,
            #     num_return_sequences=beam_num
            # )
            generated_texts = []
            for i, generated_id in enumerate(generated_ids):
                generated_texts.append(tokenizer.decode(generated_id, skip_special_tokens=True))
                if (i + 1) % beam_num == 0:
                    predictions.append(generated_texts)
                    generated_texts = []

            cache = []

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(output_file)
    print(predictions)
    print("finish time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
