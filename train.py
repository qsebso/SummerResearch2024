from datasets import load_dataset, Dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import pathlib
import json
import pdb
import sys
import datetime
import pandas as pd
import evaluate


model_checkpoint = "facebook/bart-large"
dataset = 'docred'
linearization = 'boring'

now = datetime.datetime.now()
timestamp = now.strftime("%d-%m-%H-%M-%S")

DATA_DIR = f'data/{dataset}/{linearization}'
OUTPUT_DIR = f'outputs/{dataset}/{linearization}_{model_checkpoint.replace("/", "-")}_{timestamp}'

sys.path.append(f'processing')
exec(f'from process_{dataset} import delinearize_{linearization} as delinearize')

config = json.load(open(f'{DATA_DIR}/config.json', 'r'))
possible_labels = json.load(open(f'data/{dataset}/rel_types.json'))
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# update the tokenizer with the special tokens we need for our linearization scheme
new_tokens = json.load(open(f'{DATA_DIR}/tokens.json', 'r'))
print(new_tokens)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# parse the .json data (outputs from process_{dataset}.py) into huggingface Datasets
json_data = {}
for split in ['train', 'eval']:
	split_json = json.load(open(f'{DATA_DIR}/{split}.json', 'r'))
	split_df = pd.DataFrame(split_json).iloc[:10]
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

batch_size = 8
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,return_tensors='pt',label_pad_token_id=0)
train_data_loader = DataLoader(tokenized_datasets["train"], shuffle = False, batch_size = batch_size, collate_fn = data_collator)
eval_data_loader  = DataLoader(tokenized_datasets["eval"],  shuffle = False, batch_size = batch_size, collate_fn = data_collator)

CUR_EPOCH = 0
def compute_accuracy(eval_pred):
	predictions, labels = eval_pred
	# TODO: figure out where the hell -100 token_ids are coming from
	# tokenizer.pad_token_id and model.config.pad_token_id are both 0. what do?
	predictions[predictions==-100] = 0
	pred_relations = delinearize(predictions, tokenizer)
	true_relations = delinearize(labels, tokenizer)

	# TODO: score the predictions using confusion matrix stats from eval.py!
	global CUR_EPOCH
	outputs = []
	for p_toks, t_toks, p_rels, t_rels in zip(predictions, labels, pred_relations, true_relations):
		outputs.append({
			'pred_tokens': tokenizer.decode(p_toks),
			'true_tokens': tokenizer.decode(t_toks),
			'pred_relations': p_rels,
			'true_relations': t_rels
		})
	json.dump(outputs, open(f'{OUTPUT_DIR}/outputs_{CUR_EPOCH}.json', 'w'), indent=2)
	CUR_EPOCH += 1
	return evaluate.compute_score(true_relations, pred_relations, possible_labels)

args = Seq2SeqTrainingArguments(
		output_dir=OUTPUT_DIR,
		eval_strategy="epoch",
		learning_rate=1e-3,
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