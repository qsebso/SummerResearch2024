import pandas as pd
import json
import os
import pickle
import utils
from classes import Entity, Article, Relation

# this function takes the directory where the abstracts folder along with csv files live
# it returns an array of dictionaries representing data points(keys values: PMCID int, abstract string, relations array)

DATASET = 'evidence_inference'

def load_evidence_inference() -> list[Article]:
    docs = json.load(open(f'data/{DATASET}/{DATASET}.json'))
    articles = []
    for doc in docs:
        relations = []
        for rel in doc['relations']:
            entities = [Entity('i', rel[0]),
                        Entity('c', rel[1]),
                        Entity('o', rel[2])]
            relations.append(Relation(rel[3], entities, ['i', 'c', 'o'], rel[4]))
        article = Article(doc['abstract'], relations)
        articles.append(article)
    return articles


def linearize_vertex_ref(dataset):
    name = 'vertex_ref'
    new_words = ['<rel>', '<i>', '<c>', '<o>', '<vertex>'] + [f'<{k}>' for k in RELATION_TYPES.keys()] + [f'<{i}>' for i in range(100)]
    json.dump(new_words, open(f'data/{DATASET}/{name}/tokens.json', 'w'), indent=2)

    config_data = {
        # this may need to change
        'input_ids_max_len': 600,
        'labels_max_len': 500,
    }
    json.dump(config_data, open(f'data/{DATASET}/{name}/config.json', 'w'), indent=2)

    docs = json.load(open(f'data/{DATASET}/{DATASET}.json'))
    train, eval = utils.partition_seq(docs, 0.8)
    for split_name, split_docs in {'train': train, 'eval': eval}.items():
        for article in split_docs:
            relation_strs = []
            vertex_strs = []
            for relation in article['relations']:
                i_str, c_str, o_str, type_str = relation
                if i_str not in vertex_strs:
                    vertex_strs.append(i_str)
                if c_str not in vertex_strs:
                    vertex_strs.append(c_str)
                if o_str not in vertex_strs:
                    vertex_strs.append(o_str)


                relation_strs.append(f'<rel> <{type_str}> <i> {vertex_strs.index(i_str)} <c> {vertex_strs.index(i_str)} <o> {vertex_strs.index(i_str)}')