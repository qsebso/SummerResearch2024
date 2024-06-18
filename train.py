from datasets import load_dataset, Dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
import pathlib
import pickle
import json
import pdb
import sys
import os
import datetime
import pandas as pd

model_checkpoint = "t5-small"
dataset = 'docred'
linearization = 'boring'

now = datetime.datetime.now()
timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")

DATA_DIR = f'data/{dataset}/{linearization}'
OUTPUT_DIR = f'outputs/{dataset}/{linearization}/{model_checkpoint}/{timestamp}'

sys.path.append(f'processing/{dataset}')
delinearize = None
exec(f'from process_{dataset} import delinearize_{linearization} as delinearize')

config = json.load(open(f'{DATA_DIR}/config.json', 'r'))
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# update the tokenizer with the special tokens we need for our linearization scheme
new_tokens = json.load(open(f'{DATA_DIR}/tokens.json', 'r'))
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# parse the .json data (outputs from process_{dataset}.py) into huggingface Datasets
json_data = {}
for split in ['train', 'eval']:
	split_json = json.load(open(f'{DATA_DIR}/{split}.json', 'r'))
	split_df = pd.DataFrame(split_json)
	split_df = split_df.rename(columns={'text':'document', 'linearized':'summary'})
	json_data[split] = Dataset.from_dict(split_df)
dataset = DatasetDict(json_data)

# tokenization!
def preprocess_data(examples):
	model_inputs = tokenizer(examples['document'],
			max_length = config['input_ids_max_len'],
			truncation = True,
			padding = True)
	targets = tokenizer(examples['summary'],
			max_length = config['labels_max_len'],
			truncation = True,
			padding = True)
	model_inputs['labels'] = targets['input_ids']
	return model_inputs
tokenized_datasets = dataset.map(preprocess_data, batched = True, remove_columns=['document','summary'])

# just a quick peek to make sure everything looks sane
print(tokenized_datasets['eval'][0]['labels'])
print(tokenizer.convert_ids_to_tokens(tokenized_datasets['eval'][0]['labels']))
print(len(tokenized_datasets['eval'][0]['labels']))

batch_size = 12
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,return_tensors='pt')
train_data_loader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = batch_size, collate_fn = data_collator)
eval_data_loader  = DataLoader(tokenized_datasets["eval"],  shuffle = True, batch_size = batch_size, collate_fn = data_collator)

CUR_EPOCH = 0
def compute_accuracy(eval_pred):
	predictions, labels = eval_pred
	pred_tokens = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in predictions]
	true_tokens = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in labels]
	# convert from model targets/outputs (sequences of tokens) into sets of relations
	pred_relations = delinearize(pred_tokens)
	true_relations = delinearize(true_tokens)

	# TODO: score the predictions using confusion matrix stats from eval.py!

	global CUR_EPOCH
	with open(f'{OUTPUT_DIR}/outputs_{CUR_EPOCH}.json', 'w') as fout:
		outputs = []
		for p_toks, t_toks, p_rels, t_rels in zip(pred_tokens, true_tokens, pred_relations, true_relations):
			outputs.append({
				'pred_tokens': ''.join(p_toks).strip('\u2581').replace('\u2581', ' '),
				'true_tokens': ''.join(t_toks).strip('\u2581').replace('\u2581', ' '),
				'pred_relations': p_rels,
				'true_relations': t_rels
			})
		json.dump(outputs, fout, indent=2)
		CUR_EPOCH += 1
	return { 'accuracy': 0 }

args = Seq2SeqTrainingArguments(
		output_dir=OUTPUT_DIR,
		eval_strategy="epoch",
		learning_rate=1e-3,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		weight_decay=0.01,
		save_total_limit=3,
		num_train_epochs=20,
		predict_with_generate=True,
		generation_max_length=500,
)

trainer = Seq2SeqTrainer(
		model,
		args,
		train_dataset=tokenized_datasets["train"],
		eval_dataset=tokenized_datasets["eval"],
		data_collator=data_collator,
		tokenizer=tokenizer,
		compute_metrics=compute_accuracy)

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(OUTPUT_DIR)
trainer.train()