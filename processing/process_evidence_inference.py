import pandas as pd
import json
import os
import pickle
import utils

# this function takes the directory where the abstracts folder along with csv files live
# it returns an array of dictionaries representing data points(keys values: PMCID int, abstract string, relations array)

DATASET = 'evidence_inference'
RELATION_TYPES = json.load(open(f'data/{DATASET}/rel_types.json'))

def linearize_boring():
    # i guess we should have the DATASET by this point
    name = 'boring'
    new_words = ['<rel>', '<i>', '<c>', '<o>'] + [f'<{k}>' for k in RELATION_TYPES.keys()]
    json.dump(new_words, open(f'data/{DATASET}/{name}/tokens.json', 'w'), indent=2)

    config_data = {
        'input_ids_max_len': 600,
        'labels_max_len': 500,
    }
    json.dump(config_data, open(f'data/{DATASET}/{name}/config.json', 'w'), indent=2)

    docs = json.load(open(f'data/{DATASET}/{DATASET}.json'))
    train, eval = utils.partition_seq(docs, 0.8)
    for split_name, split_docs in {'train': train, 'eval': eval}.items():
        article_strs = []
        for article in split_docs:
            relation_strs = []
            for rel in article['relations']:
                i_str, c_str, o_str, type_str = rel
                relation_strs.append(f'<rel> <{type_str}> <i> {i_str} <c> {c_str} <o> {o_str}')
            article['linearized'] = ' '.join(relation_strs)
            article['text'] = article['abstract']
            del article['abstract']
        json.dump(split_docs, open(f'data/{DATASET}/{name}/{split_name}.json', 'w'), indent=2)


def delinearize_boring(linearized_tokens):
    per_doc_relations = []
    for token_seq in linearized_tokens:
        relations = set()
        rel_token_seqs = utils.split_seq(token_seq, '<rel>')
        for rel_token_seq in rel_token_seqs:
            # Can't have fewer than the required tokens
            if len(rel_token_seq) < 3:
                continue
            rel_type_token = rel_token_seq[0].strip('<>')
            # The first token should be the relation type
            if rel_type_token not in RELATION_TYPES:
                continue
            # we need one head entity
            if rel_token_seq.count('<i>') != 1:
                continue
            # and one tail entity
            if rel_token_seq.count('<c>') != 1:
                continue
            if rel_token_seq.count('<o>') != 1:
                continue
            # the tail can't come before the head
            i_idx = rel_token_seq.index('<i>')
            c_idx = rel_token_seq.index('<c>')
            o_idx = rel_token_seq.index('<o>')
            if not (i_idx < c_idx < o_idx):
                continue
            # everything seems in order! let's build the relation tuple
            rel_type = RELATION_TYPES[rel_type_token]
            # join the token strings into a single string
            i_toks = tuple(rel_token_seq[i_idx+1:c_idx])
            c_toks = tuple(rel_token_seq[c_idx+1:o_idx])
            o_toks = tuple(rel_token_seq[o_idx+1:])
            relations.add((rel_type, i_toks, c_toks, o_toks))
        per_doc_relations.append(list(relations))
    return per_doc_relations


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



if __name__ == '__main__':
    linearize_boring()