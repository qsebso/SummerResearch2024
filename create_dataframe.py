# Goal/Table/Why/What
"""
The overall goal of this analysis is to determine if there are trends/patterns in documents or relations that the models make mistakes on. 
Performing this analysis can provide benefit in several ways:
If there are common failure modes, this may inform ways to improve the model or metrics 
(e.g. The alignment criteria is clearly unreasonable for Evidence Inference, but also our model seems to always predice "No change" so there's something to fix there too)
If can determine which inputs are likely to be hard for the model to predict, we can make sure that users are alerted that they should verify model outputs
We could determine which domains/datasets are a good match for this type of model, and where we're just doomed from the beginning
If there are inputs that are easy for some models but not others, it would be possible to do things like train multiple models and use the one that best matches the type of document
To answer these questions, we just need to capture (a) quantitative properties of different inputs, and (b) quantitative measures of model performance on those inputs. 
I think the most useful piece of information to construct would be a table that looks like the following (numbers made up):


length, n_sents ,n_rels ,n_entities, ...whatever else we think of, acc, precision, recall, f1
Doc 0 743 10 7 17 ... 0.43 0.322 0.521 0.398
Doc 1 590 6 8 14 ... 0.22 0.117 0.682 0.451
... ... ... ... ... ... ... ... ... ...

For analysis (statistics and data vis), this would probably be easiest to work with as a pandas DataFrame. 
This way you can directly use stats functions in pandas, or hand the DataFrame to seaborn to make plots. 
For example, DataFrames have a .corr() method that will give you pairwise correlations between all of the columns. 
The real trophy here is to find patterns/relationships between the primary scoring metric (f1, by default) and any of the columns with features about the input. 
If there are overall trends we should see them in correlation statistics, 
but if there are other non-linear relationships (e.g. clusters) we may not notice them unless we visualize the data.
"""

# Here are the concrete steps that need to be completed in order to carry out the analysis:
"""
Write code that loads an output file, and generates the above DataFrame for it
    Compute input-specific stats for each document (you've mostly done this already)
    Compute output-specific metrics for each document (we have code for the confusion matrix, but not for deriving precision/recall/f1)
    Assemble all of the per-document metrics into a single DataFrame
Write code that takes one of the above DataFrames, and automatically generates correlation scores between f1 and the input metrics
Write code that takes one of the above DataFrames, and automatically makes scatterplots of x=f1, y=length (or n_sents, or n_rels, or whatever)
Get output files for a variety of datasets and models, and run the above code to look for any trends or patterns
Spot-check specific examples as needed to help develop an understanding of what's working and what isn't

Things that are being worked on by me and/or Tanner include:
Confusion matrix -> metric scoring functions
Adding the "text" field to the output files (surprisingly not easy!)
Experimenting with different models/setupts to produce better output files
"""

from get_eval_features import get_input_specific_stats
from evaluate import compute_score
from classes import Entity, Relation
import json
import pandas as pd

# File paths
output_file_first = 'outputs_2.json'
output_file_newest = 'REBEL_outputs.json'
rel_types_file = 'data/docred/rel_types.json'

# DF file
dataframe_file_fist = 'analysis_output.csv'
dataframe_file_newest = 'analysis_output_REBEL.csv'
# Compute input-specific stats for each document
input_specific_stats = get_input_specific_stats(output_file_newest)

# Load the original output data
with open(output_file_newest, 'r') as file:
    data = json.load(file)

# Prepare data for DataFrame
doc_ids = [f"Doc {i}" for i in range(len(data))]
df = pd.DataFrame(input_specific_stats, index=doc_ids)

# Load relation types
with open(rel_types_file, 'r') as file:
    rel_types = json.load(file)

# Convert relations from JSON
def relation_from_json(json_data):
    rtype = json_data['rtype']
    entities = [Entity('', entity['span']) for entity in json_data['entities'].values()]
    slots = list(json_data['entities'].keys())
    evidence = json_data.get('evidence', [])
    return Relation(rtype, entities, slots, evidence)

# Compute per-document metrics
per_doc_metrics = []
for doc in data:
    true_relations = [relation_from_json(rel) for rel in doc['true_relations']]
    pred_relations = [relation_from_json(rel) for rel in doc['pred_relations']]
    scores = compute_score([true_relations], [pred_relations], rel_types)
    per_doc_metrics.append(scores)

# Extract metrics from per_doc_metrics and add to DataFrame
precision = [metrics['weighted avg']['precision'] for metrics in per_doc_metrics]
recall = [metrics['weighted avg']['recall'] for metrics in per_doc_metrics]
f1 = [metrics['weighted avg']['f1-score'] for metrics in per_doc_metrics]
support = [metrics['weighted avg']['support'] for metrics in per_doc_metrics]

df['precision'] = precision
df['recall'] = recall
df['f1'] = f1
df['support'] = support

# Save the DataFrame to a CSV file for further analysis
df.to_csv(dataframe_file_newest, index=False)

# Print the DataFrame
print(df.head())
