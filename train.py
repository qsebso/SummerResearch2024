from datasets import load_dataset, Dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import pathlib
import json
import pdb
import datetime
import evaluate
import linearization
from typing import cast


CUR_EPOCH = 0
def run_training_loop(MODEL_CKPT, DATASET, ENCODING):
	delinearize = getattr(linearization, f'delinearize_{ENCODING}')

	now = datetime.datetime.now()
	timestamp = now.strftime("%d-%m-%H-%M-%S")

	DATA_DIR = f'data/{DATASET}/{ENCODING}'
	OUTPUT_DIR = f'outputs/{DATASET}/{ENCODING}_{MODEL_CKPT.replace("/", "-")}_{timestamp}'

	config = json.load(open(f'{DATA_DIR}/config.json', 'r'))
	possible_labels = json.load(open(f'data/{DATASET}/rel_types.json'))
	model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPT)

	# update the tokenizer with the special tokens we need for our ENCODING scheme
	new_tokens = json.load(open(f'{DATA_DIR}/tokens.json', 'r'))
	print(new_tokens)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
	tokenizer.add_tokens(new_tokens)
	model.resize_token_embeddings(len(tokenizer))

	# parse the .json data (outputs from process.{dataset}.py) into huggingface Datasets
	split2filename = {
		'train': f'{DATA_DIR}/train.json',
		'eval': f'{DATA_DIR}/eval.json'
	}
	dataset: Dataset = cast(Dataset, load_dataset('json', data_files=split2filename))
	# tokenization!
	def preprocess_data(examples):
		model_inputs = tokenizer(examples['text'],   max_length = 512, truncation = True)
		targets      = tokenizer(examples['target'], max_length = 512, truncation = True)
		model_inputs['labels'] = targets['input_ids']
		return model_inputs

	tokenized_dataset: Dataset = dataset.map(preprocess_data, batched=True)
	# just a quick peek to make sure everything looks sane
	print(tokenized_dataset['eval'][0]['labels'])
	print(tokenizer.convert_ids_to_tokens(tokenized_dataset['eval'][0]['labels']))
	print(len(tokenized_dataset['eval'][0]['labels']))

	def compute_accuracy(eval_pred):
		predictions, labels = eval_pred
		# TODO: figure out where the hell -100 token_ids are coming from
		# tokenizer.pad_token_id and model.config.json.pad_token_id are both 0. what do?
		labels[labels==-100] = 0
		predictions[predictions==-100] = 0
		pred_relations = delinearize(predictions, tokenizer, DATASET)
		true_relations = delinearize(labels, tokenizer, DATASET)

		# TODO: score the predictions using confusion matrix stats from eval.py!
		global CUR_EPOCH
		outputs = []
		for text, p_toks, t_toks, p_rels, t_rels in zip(dataset['eval']['text'], predictions, labels, pred_relations, true_relations):
			outputs.append({
				'text': text,
				'pred_target': tokenizer.decode(p_toks, skip_special_tokens=True),
				'true_target': tokenizer.decode(t_toks, skip_special_tokens=True),
				'pred_relations': [rel.to_dict() for rel in p_rels],
				'true_relations': [rel.to_dict() for rel in t_rels]
			})
		path = pathlib.Path(OUTPUT_DIR)
		if not path.exists():
			path.mkdir(parents=True, exist_ok=True)
		json.dump(outputs, open(f'{OUTPUT_DIR}/outputs_{CUR_EPOCH}.json', 'w'), indent=2)
		CUR_EPOCH += 1
		return evaluate.compute_score(true_relations, pred_relations, possible_labels)

	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
	args = Seq2SeqTrainingArguments(
			output_dir=OUTPUT_DIR,
			evaluation_strategy="epoch",
			learning_rate=1e-3,
			weight_decay=0.01,
			save_total_limit=3,
			num_train_epochs=20,
			per_device_train_batch_size=10,
			per_device_eval_batch_size=8,
			predict_with_generate=True,
			generation_max_length=500,
	)

	trainer = Seq2SeqTrainer(
			model,
			args,
			train_dataset=tokenized_dataset["train"],
			eval_dataset=tokenized_dataset["eval"],
			data_collator=data_collator,
			tokenizer=tokenizer,
			compute_metrics=compute_accuracy)

	print(OUTPUT_DIR)
	trainer.train()

if __name__ == '__main__':
	MODEL_CKPT = "facebook/bart-large"
	DATASET = 'docred'
	ENCODING = 'boring'
	run_training_loop(MODEL_CKPT, DATASET, ENCODING)