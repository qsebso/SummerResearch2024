from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
import pickle
import json
import pdb
import sys
import pandas as pd

model_checkpoint = "t5-small"
dataset = 'docred'
linearization = 'boring'

sys.path.append(f'processing/{dataset}')
exec(f'from process_{dataset} import delinearize_{linearization} as delinearize')

DATA_DIR = f'data/{dataset}/{linearization}'

new_words = json.load(open(f'{DATA_DIR}/tokens.json', 'r'))
data_dump = json.load(open(f'{DATA_DIR}/train.json', 'r'))
config    = json.load(open(f'{DATA_DIR}/config.json', 'r'))

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(new_words)
model.resize_token_embeddings(len(tokenizer))

# get the datasetcd ..
data = pd.DataFrame(data_dump)
data = data.rename(columns={'text': 'document', 'linearized': 'summary'})
split_index = int(len(data) * 0.8)
train_datadict = data.iloc[:split_index]
validate_datadict = data.iloc[split_index:]
train_dataset = Dataset.from_dict(train_datadict)
validate_dataset = Dataset.from_dict(validate_datadict)
dataset = DatasetDict({'train': train_dataset, 'validation': validate_dataset})

pdb.set_trace()

def preprocess_data(examples):
		model_inputs = tokenizer(examples['document'],
				max_length = config['input_ids_max_len'],
				truncation = True,
				padding = True)
		with tokenizer.as_target_tokenizer():
				targets = tokenizer(examples['summary'],
                        max_length = config['labels_max_len'],
						truncation = True,
						padding = True)
		model_inputs['labels'] = targets['input_ids']
		return model_inputs
tokenized_datasets = dataset.map(preprocess_data, batched = True, remove_columns=['document','summary'])
print(tokenized_datasets['validation'][0]['labels'])
print(tokenizer.convert_ids_to_tokens(tokenized_datasets['validation'][0]['labels']))
print(len(tokenized_datasets['validation'][0]['labels']))

pdb.set_trace()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,return_tensors='pt')
batch_size = 12
train_data_loader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = batch_size, collate_fn = data_collator)
eval_data_loader = DataLoader(tokenized_datasets["validation"], shuffle = True, batch_size = batch_size, collate_fn = data_collator)


# confusion matrix generator
def generate_confusion_matrix(pred_labels, true_labels, possible_labels):
	matrix = [[0 for x in range(len(possible_labels))] for y in range(len(possible_labels))]
	matched_full = []
	print()
	for x, article in enumerate(pred_labels):
		for relation in article:
			print(relation)
			matches = _match(relation, true_labels[x])
			pred_idx = possible_labels.index(relation['r'])
			if matches is None:
				# generated an invalid relation
				matrix[pred_idx][-1] += 1
				matched_full.append(relation)
			elif matches is list:
				# we generated the right pair worng relation ID
				actual_idx = [possible_labels.index(matches[0]['r'])]
				matrix[pred_idx][actual_idx] += 1
				matched_full.append(relation[0])
			else:
				actual_idx = [possible_labels.index(matches['r'])]
				matrix[pred_idx][actual_idx] += 1
				matched_full.append(relation)
		if not matched_full == true_labels[x]:
			for x, label in enumerate(true_labels[x]):
				if label not in matched_full:
					matrix[-1][possible_labels.index(label['r'].replace(' ', ''))] += 1

	return matrix

# get the valid labels
def get_labels(filepath):
	with open(filepath, 'r') as file:
		table = pandas.read_json(file, typ='series')
	keys = [key for key in table.keys()]
	return keys

CUR_EPOCH = 0

def compute_accuracy(eval_pred):
	predictions, labels = eval_pred
	token_preds = tokenizer.batch_decode(predictions)
	token_labs = tokenizer.batch_decode(labels)


	#need to delinearize before confusion matrix
	# need dictionaries for preds and labs of each data_point
	pred_dicts = [{'linearized': pred_lin} for pred_lin in token_preds]
	lab_dicts = [{'linearized': lab_lin} for lab_lin in token_labs]

	global CUR_EPOCH
	with open(f'outputs_{CUR_EPOCH}.pkl', 'w') as fout:
		json.dump({'pred': pred_dicts, 'labeled': lab_dicts}, fout, indent=2)
		CUR_EPOCH += 1
	try:
		delinearize(token_preds, pred_dicts)
		delinearize(token_labs, lab_dicts)

		pred_rels = [article['relations'] for article in pred_dicts]
		lab_rels = [article['relations'] for article in lab_dicts]


		label_list = get_labels('rel_info.json')
		confusion_matrix = generate_confusion_matrix(pred_rels, lab_rels, label_list)


		num_correct = 0
		for x in range(len(label_list)):
			num_correct += confusion_matrix[x][x]

		print('about to return the accuracy')
		return {'accuracy': num_correct/len(predictions)}
	except:
		return {'accuracy': 0 }

	
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
		output_dir=f"{model_checkpoint}-DocRED",
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
		eval_dataset=tokenized_datasets["validation"],
		data_collator=data_collator,
		tokenizer=tokenizer,
		compute_metrics=compute_accuracy)

trainer.train()
