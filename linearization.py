import json

import utils
from classes import Article, Relation

def linearize_boring(docs: list[Article], dataset: str):
    RELATION_TYPES = json.load(open(f'data/{dataset}/rel_types.json'))
    RELATION_SLOTS = json.load(open(f'data/{dataset}/rel_slots.json'))
    # i guess we should have the DATASET by this point
    name = 'boring'
    new_words = ['<rel>'] + [f'<{k}>' for k in RELATION_TYPES.keys()] + ['<{k}>' for k in RELATION_SLOTS.keys()]
    json.dump(new_words, open(f'data/{dataset}/{name}/tokens.json', 'w'), indent=2)

    config_data = {
        'input_ids_max_len': 600,
        'labels_max_len': 500,
    }
    json.dump(config_data, open(f'data/{dataset}/{name}/config.json', 'w'), indent=2)

    train, eval = utils.partition_seq(docs, 0.8)
    for split_name, split_docs in {'train': train, 'eval': eval}.items():
        article_strs = []
        for article in split_docs:
            relation_strs = []
            for rel in article.relations:
                rel_pieces= []
                rel_type = rel.type
                rel_pieces += ['<rel>', f'<{rel_type}>']
                for entity, slot in zip(rel.entities, rel.slots):
                    rel_pieces += [f'<{slot}>', entity.span]
                relation_strs.append(''.join(rel_pieces))
            article.linearized = ' '.join(relation_strs)
        json.dump(split_docs, open(f'data/{dataset}/{name}/{split_name}.json', 'w'), indent=2)


def delinearize_boring(linearized_tokens: list[list[int]], tokenizer, dataset: str):
    RELATION_TYPES = json.load(open(f'data/{dataset}/rel_types.json'))
    RELATION_SLOTS = json.load(open(f'data/{dataset}/entity_types.json'))
    per_doc_relations = []
    for token_seq in linearized_tokens:
        relations = set()
        rel_token_seqs = utils.split_seq(token_seq, '<rel>')
        for rel_token_seq in rel_token_seqs:
            # Can't have fewer than the required tokens
            if len(rel_token_seq) < len(RELATION_SLOTS) + 1:
                continue
            rel_type_token = rel_token_seq[0].strip('<>')
            # The first token should be the relation type
            if rel_type_token not in RELATION_TYPES:
                continue
            # we need one head entity
            entity_token_counts = [rel_token_seq.count(f'<{entity_type}>') for entity_type in RELATION_SLOTS]
            if not all([count == 1 for count in entity_token_counts]):
                continue
            # the tail can't come before the head
            entity_token_idxs = [rel_token_seq.index(f'<{entity_type}>') for entity_type in RELATION_SLOTS]
            valid_idxs = [entity_token_idxs[i] < entity_token_idxs[i+1] for i in range(len(entity_token_idxs)-1)]
            if not all(valid_idxs):
                continue

            # everything seems in order! let's build the relation tuple
            rel_type = rel_type_token
            # the start and stop token indices for adjacent entities
            slice_idxs = entity_token_idxs + [len(rel_token_seq)]
            entities = []
            for start, stop in zip(slice_idxs[:-1], slice_idxs[1:]):
                entities.append(Entity('[UNK]', tokenizer.decode(rel_token_seq[start+1:stop])))
            relations.add(Relation(rel_type, entities, RELATION_SLOTS))
        per_doc_relations.append(list(relations))
    return per_doc_relations