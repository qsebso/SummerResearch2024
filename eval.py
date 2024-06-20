# This is where our human-readable evaluation metrics will go!
# e.g. precision, recall, f1, etc.

def compute_score(true_rels, pred_rels):
    return { 'f1': 0, 'precision': 0, 'recall': 0 }

# takes a given predicted relation and a list of expected true relations
# returns the tuple if it exactly matches a true label
# returns a list of tuples with the same entities involved if no exact matches
# returns none if there was no expected relation between those entites
def match(relation, true_rels):
    output = []
    for comparison in true_rels:
        if comparison[1:] == relation[1:]:
            if comparison[0] == relation[0]:
                return comparison
            else:
                output.append(comparison)
    if output == []:
        return None
    else:
        return output


def generate_confusion_matrix(true_rels: list[list[list]],
                              pred_rels: list[list[list]],
                              possible_labels: dict) -> list[list[int]]:
    ordered_labels = {}
    for x, label in enumerate(possible_labels.values()):
        ordered_labels[label] = x
    matrix = [[0 for x in range(len(ordered_labels) + 1)] for y in range(len(ordered_labels) + 1)]
    for x, article in enumerate(pred_rels):
        # list of things that we've found matches for
        matched_full = []
        for relation in article:
            matches = match(relation, true_rels[x])
            pred_idx = ordered_labels[relation[0]]
            if matches is None:
                # no matching entity combination
                actual_idx = -1
            elif matches is list:
                # we have multiple possible things this could be guessing
                actual_idx = ordered_labels[matches[0][0]]
                matched_full.append(matches[0])
            else:
                # we generated a relationship that we expected to see
                actual_idx = ordered_labels[matches[0]]
                matched_full.append(matches)
            matrix[pred_idx][actual_idx] += 1
        if matched_full != true_rels[x]:
            for y, label in enumerate(true_rels[x]):
                if label not in matched_full:
                    matrix[-1][ordered_labels[label[0]]] += 1

    return matrix
