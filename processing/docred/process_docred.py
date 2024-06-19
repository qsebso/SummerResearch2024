import pandas
import json
import pickle
import pdb
import os

RELATION_TYPES = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}

# take a filepath for json containing data
# return a dictionary containing data of interest
def get_docred(fp):
    i = 0
    file = pandas.read_json(fp)
    articles = []
    for index, row in file.iterrows():
        document = ""
        sentence_lengths = []
        for sentence in row['sents']:
            sentence_lengths.append(len(sentence))
            for word in sentence:
                document += word + ' '

        vertices = []
        for vertex_set in row['vertexSet']:
            vertex = dict()
            for usage in vertex_set:
                vertex['span'] = usage['name']
                sent_idx = sum(sentence_lengths[:usage['sent_id']])
                vertex['start_idx'] = sent_idx + usage['pos'][0]
                vertex['end_idx'] = sent_idx + usage['pos'][1]
            vertices.append(vertex)

        relations = row['labels']
        for relation in relations:
            del relation['evidence']
        article = dict()
        article['text'] = document
        article['vertexList'] = vertices
        article['relations'] = relations
        articles.append(article)

        i += 1
    return articles

# take a dataset from json import format
# return a linear string including vertices and relations
def linearize_vertex_ref(dataset):
    # relations
    output = []
    for article in dataset:
        linear = ""
        for x, vertex in enumerate(article['vertexList']):
            linear += "<vertex>" + vertex['span'] + '[['+ str(x) + ']]'

        for relation in article['relations']:
            linear += "<r>" + relation['r'] + "<h>" + str(relation['h']) + "<t>" + str(relation['t'])
        linear += '<end>'

        # linear is the full, complete string. It may be useful to have this come as a tuple with the input as well
        output.append((article['text'], linear))
    return output


def _delinearize_relations(strings):
    output = []
    for string in strings:
        ht = string.split('<h>')
        relation_type = ht.pop(0).replace(' ', '')
        final_split = ht[0].split('<t>')
        head = int(final_split.pop(0))
        if '<end>' in final_split[0]:
            # final_split[0] = final_split[0].replace('<pad>', '')
            tail = int(final_split.pop()[:-11])
        else:
            tail = int(final_split.pop())
        output.append({'r': relation_type, 'h': head, 't': tail})
    return output


def delinearize_vertex_ref(linearized_strings):
    for x, linearized_string in enumerate(linearized_strings):
        linearized_string = linearized_string.replace('<pad>', '')
        split = linearized_string.split('<r>')
        vertices = linearized_string.pop(0)
        relations = _delinearize_relations(split)
        vertices = vertices.split('<vertex>')
        vertices.pop(0)
        vertices_dict = {}
        for vertex in vertices:
            split = vertex.split('[[')
            if '</s>' in split[1]:
                split[1] = split[1].replace('</s>', '')
                split[1] = split[1].replace('<end> ', '')
            vertices_dict[int(split[1][:-3])] = split[0][1:-1]
        output = {}
        output[x]['relations'] = relations
        output[x]['vertexList'] = vertices_dict
        output[x]['linearized'] = linearized_string

        return output

def linearize_boring():
    name = 'boring'
    relation_ids = [f'<{i}>' for i in range(100)]
    new_words = ['<rel>', '<t>', '<h>'] + [f'<{k}>' for k in RELATION_TYPES.keys()] + relation_ids

    json.dump(new_words, open(f'../../data/docred/{name}/tokens.json', 'w'), indent=2)

    config_data = {
        'input_ids_max_len': 600,
        'labels_max_len': 500,
    }
    json.dump(config_data, open(f'../../data/docred/{name}/config.json', 'w'), indent=2)

    splits = {'train': 'train_data', 'eval': 'dev'}
    for output_split, input_split in splits.items():
        data = get_docred(f'../../data/docred/{input_split}.json')
        article_strs = []
        for article in data:
            relation_strs = []
            for rel in article['relations']:
                type_str = rel['r']
                head_str = article['vertexList'][rel['h']]['span']
                tail_str = article['vertexList'][rel['t']]['span']
                relation_strs.append(f'<rel> <{type_str}> <h> {head_str} <t> {tail_str}')
            article['linearized'] = ' '.join(relation_strs)

        json.dump(data, open(f'../../data/docred/{name}/{output_split}.json', 'w'), indent=2)

def split_seq(seq, val):
    result = []
    temp_list = []
    for x in seq:
        if x == val:
            if temp_list:
                result.append(temp_list)
                temp_list = []
        else:
            temp_list.append(x)
    result.append(temp_list)
    return result

def delinearize_boring(linearized_tokens):
    per_doc_relations = []
    for token_seq in linearized_tokens:
        relations = set()
        rel_token_seqs = split_seq(token_seq, '<rel>')
        for rel_token_seq in rel_token_seqs:
            # Can't have fewer than the required tokens
            if len(rel_token_seq) < 3:
                continue
            rel_type_token = rel_token_seq[0].strip('<>')
            # The first token should be the relation type
            if rel_type_token not in RELATION_TYPES:
                continue
            # we need one head entity
            if rel_token_seq.count('<h>') != 1:
                continue
            # and one tail entity
            if rel_token_seq.count('<t>') != 1:
                continue
            # the tail can't come before the head
            h_idx = rel_token_seq.index('<h>')
            t_idx = rel_token_seq.index('<t>')
            if t_idx < h_idx:
                continue
            # everything seems in order! let's build the relation tuple
            rel_type = RELATION_TYPES[rel_type_token]
            # join the token strings into a single string
            head = ''.join(rel_token_seq[h_idx+1:t_idx]).replace('\u2581', ' ')
            tail = ''.join(rel_token_seq[t_idx+1:]).replace('\u2581', ' ')
            relations.add((rel_type, head, tail))
        per_doc_relations.append(list(relations))
    return per_doc_relations

if __name__ == '__main__':
    linearize_boring()