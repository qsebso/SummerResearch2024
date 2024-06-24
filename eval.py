from typing import Tuple
import pdb

# This is where our human-readable evaluation metrics will go!
# e.g. precision, recall, f1, etc.

def compute_score(true_rels, pred_rels):
    return { 'f1': 0, 'precision': 0, 'recall': 0 }

def get_relation_label_entities(rel: list) -> Tuple[str, list]:
    return rel[0], rel[1:]

def match_entity(e1: list[str], e2: list[str]) -> bool:
    return e1 == e2

def match_entities(true_entities: list[list], pred_entities: list[list]) -> bool:
    return all([match_entity(t, p) for t, p in zip(true_entities, pred_entities)])

def tupleize_relation(relation):
    # JSON-serialized relations are lists, but we need them to be hashable
    # input:
    # [type_str, [entity0_token0, entity0_token1, ...], [entity1_token0, entity1_token1, ...], ...]
    # output:
    # (type_str, (entity0_token0, entity0_token1, ...), (entity1_token0, entity1_token1, ...), ...)
    return (relation[0],) + tuple(map(tuple, relation[1:]))

# takes a given predicted relation and a list of expected true relations
# returns the tuple if it exactly matches a true label
# returns a list of tuples with the same entities involved if no exact matches
# returns none if there was no expected relation between those entites
def match_relation(true_rel, pred_rels) -> list:
    matched_pred_rels = set()
    true_label, true_entities = get_relation_label_entities(true_rel)
    for pred_rel in pred_rels:
        pred_label, pred_entities = get_relation_label_entities(pred_rel)
        if match_entities(true_entities, pred_entities):
            matched_pred_rels.add(pred_rel)
    return list(matched_pred_rels)


def generate_confusion_matrix(all_true_rels: list[list[list]],
                              all_pred_rels: list[list[list]],
                              possible_labels: dict) -> list[list[int]]:
    """
    arguments:
    all_true_rels: a list of lists of lists representing the true labels for each
    """
    ordered_labels = {}
    MISSING_LABEL = '<NONE>'
    possible_labels[MISSING_LABEL] = MISSING_LABEL
    for i, label in enumerate(possible_labels.values()):
        ordered_labels[label] = i
    print(ordered_labels)
    matrix = [[0 for x in range(len(ordered_labels))] for y in range(len(ordered_labels))]
    for doc_true_rels, doc_pred_rels in zip(all_true_rels, all_pred_rels):
        # list of things that we've found matches for
        doc_true_rels = [tupleize_relation(rel) for rel in doc_true_rels]
        doc_pred_rels = {tupleize_relation(rel) for rel in doc_pred_rels}
        true_rel_matches = {}
        for true_rel in doc_true_rels:
            matched_pred_rels = match_relation(true_rel, doc_pred_rels)
            true_rel_matches[true_rel] = matched_pred_rels
            doc_pred_rels = doc_pred_rels.difference(matched_pred_rels)
        for true_rel, pred_rel_matches in true_rel_matches.items():
            if len(pred_rel_matches) == 0:
                pred_labels = [MISSING_LABEL]
            else:
                pred_labels = [label for label, entities in map(get_relation_label_entities, pred_rel_matches)]
            true_label, true_entities = get_relation_label_entities(true_rel)
            for pred_label in pred_labels:
                matrix[ordered_labels[true_label]][ordered_labels[pred_label]] += 1
        unmatched_pred_labels = [label for label, entities in map(get_relation_label_entities, doc_pred_rels)]
        for pred_label in unmatched_pred_labels:
            matrix[ordered_labels[MISSING_LABEL]][ordered_labels[pred_label]] += 1

    return matrix
