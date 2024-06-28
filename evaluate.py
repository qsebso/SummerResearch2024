from typing import Tuple, cast, Iterable
import pdb
from classes import Entity, Relation, Article
from sklearn.metrics import precision_recall_fscore_support, classification_report

# This is where our human-readable evaluation metrics will go!
# e.g. precision, recall, f1, etc.

def match_entity(e1: Entity, e2: Entity) -> bool:
    return e1.span == e2.span

def match_entities(true_entities: list[Entity], pred_entities: list[Entity]) -> bool:
    return all([match_entity(t, p) for t, p in zip(true_entities, pred_entities)])

# takes a given predicted relation and a list of expected true relations
# returns the tuple if it exactly matches a true label
# returns a list of tuples with the same entities involved if no exact matches
# returns none if there was no expected relation between those entites
def match_relation(true_rel: Relation, pred_rels: Iterable[Relation]) -> list[Relation]:
    matched_pred_rels = set()
    for pred_rel in pred_rels:
        if match_entities(true_rel.entities, pred_rel.entities):
            matched_pred_rels.add(pred_rel)
    return list(matched_pred_rels)


def compute_score(all_true_rels: list[list[Relation]],
                  all_pred_rels: list[list[Relation]],
                  possible_labels: dict) -> dict:
    MISSING_LABEL = '<NONE>'
    eval_labels = [MISSING_LABEL] + list(possible_labels.keys())
    all_true_labels = []
    all_pred_labels = []

    for doc_true_rels, doc_pred_rels in zip(all_true_rels, all_pred_rels):
        # list of things that we've found matches for
        doc_pred_rels = set(doc_pred_rels)

        # the first task is to align all of the predicted relations to the true
        # relations with matching entities
        # this is potentially a one-to-many alignment, since it's possible to generate
        # multiple copies of the same entities as predictions
        true_rel_matches: dict[Relation, list[Relation]]= {}
        for true_rel in doc_true_rels:
            # determine which (if any) of the predictions go with this relation
            matched_pred_rels = match_relation(true_rel, doc_pred_rels)
            true_rel_matches[true_rel] = matched_pred_rels
            # each prediction should only get aligned to a single true relation
            doc_pred_rels = doc_pred_rels.difference(matched_pred_rels)

        # we can tap in to sklearn's functions for all this stuff by flattening out
        # the labels into parallel lists
        doc_true_labels = []
        doc_pred_labels = []

        # for each true relation, consider all of the aligned predicted relations
        for true_rel, pred_rel_matches in true_rel_matches.items():
            # maybe we just didn't predict any relation for the same entities; we have a special label for this case
            if len(pred_rel_matches) == 0:
                pred_labels = [MISSING_LABEL]
            # if we did find prediction(s), we just need the labels from them
            else:
                pred_labels = [rel.rtype for rel in pred_rel_matches]
            # and what the true label was supposed to be
            # once we have all the (1+) predicted labels, we update our bookkeeping
            for pred_label in pred_labels:
                doc_true_labels.append(true_rel.rtype)
                doc_pred_labels.append(pred_label)
        
        # finally, we have to consider the predictions that didn't get aligned to anything
        # this means we predicted entities that just didn't match anything, which is a mistake
        unmatched_pred_labels = [rel.rtype for rel in doc_pred_rels]
        for pred_label in unmatched_pred_labels:
            # the correct thing would have been not to predict a relation for these entities at all! oops
            doc_true_labels.append(MISSING_LABEL)
            doc_pred_labels.append(pred_label)

        all_true_labels.extend(doc_true_labels)
        all_pred_labels.extend(doc_pred_labels)

    print(classification_report(
        all_true_labels, all_pred_labels, labels=eval_labels, output_dict=False))

    return cast(dict, classification_report(
        all_true_labels, all_pred_labels, labels=eval_labels, output_dict=True))
