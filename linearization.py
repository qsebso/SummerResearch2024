import pdb
import json

import utils
from classes import Article, Relation, Entity

def linearize_boring(docs: list[Article], dataset: str, filename):
    RELATION_TYPES = json.load(open(f'data/{dataset}/rel_types.json'))
    RELATION_SLOTS = json.load(open(f'data/{dataset}/rel_slots.json'))
    # i guess we should have the DATASET by this point
    name = 'boring'
    new_words = ['<rel>'] + [f'<{k}>' for k in RELATION_TYPES.keys()] + [f'<{k}>' for k in RELATION_SLOTS]
    json.dump(new_words, open(f'data/{dataset}/{name}/tokens.json', 'w'), indent=2)

    config_data = {
        'input_ids_max_len': 600,
        'labels_max_len': 500,
    }
    json.dump(config_data, open(f'data/{dataset}/{name}/config.json', 'w'), indent=2)

    outputs = []
    for article in docs:
        relation_strs = []
        for rel in article.relations:
            rel_pieces = ['<rel>', f'<{rel.rtype}>']
            for entity, slot in zip(rel.entities, rel.slots):
                rel_pieces += [f'<{slot}>', entity.span]
            relation_strs.append(''.join(rel_pieces))
        target = ' '.join(relation_strs)
        article_output = article.to_dict()
        article_output['target'] = target
        outputs.append(article_output)

    json.dump(outputs, open(f'data/{dataset}/{name}/{filename}.json', 'w'), indent=2)


def delinearize_boring(linearized_tokens: list[list[int]], tokenizer, dataset: str):
    RELATION_TYPES = json.load(open(f'data/{dataset}/rel_types.json'))
    RELATION_SLOTS = json.load(open(f'data/{dataset}/rel_slots.json'))
    str2token = tokenizer.get_added_vocab()
    rel_token = str2token['<rel>']
    per_doc_relations = []
    for token_seq in linearized_tokens:
        relations = set()
        rel_token_seqs = utils.split_seq(token_seq, rel_token)
        for rel_token_seq in rel_token_seqs:
            # Can't have fewer than the required tokens
            if len(rel_token_seq) < len(RELATION_SLOTS) + 1:
                continue
            rel_type_str = tokenizer.convert_ids_to_tokens(rel_token_seq[0]).strip('<>')
            # The first token should be the relation type
            if rel_type_str not in RELATION_TYPES:
                continue
            # we need one head entity
            slot_token_strs = [f'<{slot_name}>' for slot_name in RELATION_SLOTS]
            slot_tokens = [str2token[slot_token] for slot_token in slot_token_strs]
            slot_token_counts = [rel_token_seq.count(slot_token) for slot_token in slot_tokens]
            if not all([count == 1 for count in slot_token_counts]):
                continue
            # the tail can't come before the head
            slot_token_idxs = [rel_token_seq.index(slot_token) for slot_token in slot_tokens]
            valid_idxs = [slot_token_idxs[i] < slot_token_idxs[i+1] for i in range(len(slot_token_idxs)-1)]
            if not all(valid_idxs):
                continue

            # everything seems in order! let's build the relation tuple
            # the start and stop token indices for adjacent entities
            slice_idxs = slot_token_idxs + [len(rel_token_seq)]
            entities = []
            for start, stop in zip(slice_idxs[:-1], slice_idxs[1:]):
                entities.append(Entity('[UNK]', tokenizer.decode(rel_token_seq[start+1:stop])))
            relations.add(Relation(rel_type_str, entities, RELATION_SLOTS))
        per_doc_relations.append(list(relations))
    return per_doc_relations


def linearize_docred():
    from processing.process_docred import get_docred
    for fname_in, fname_out in [('train_data', 'train'), ('dev', 'eval')]:
        docs = get_docred(f'data/docred/{fname_in}.json')
        linearize_boring(docs, 'docred', fname_out)