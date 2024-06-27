import pandas
import json
import pickle
import pdb
import os
import utils
import sys
sys.path.append("..")
import config

DATA_DIR = f'{config.TOP}/data/docred'
RELATION_TYPES = json.load(open(f'{DATA_DIR}/rel_types.json'))

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
                vertex['etype'] = usage['type']
                sent_idx = sum(sentence_lengths[:usage['sent_id']])
                vertex['start_idx'] = sent_idx + usage['pos'][0]
                vertex['end_idx'] = sent_idx + usage['pos'][1]
            vertices.append(vertex)

        #relations = row['labels']
        #for relation in relations:
        #    del relation['evidence']
        relations = []
        for relation in row['labels']:
            h_vertex = vertex_set[relation['h']]
            entities = [Entity(h_vertex['type'], h_vertex['span']),
                        Entity(h_vertex['type'], h_vertex['span'])]
            relations.append(Relation(row['r'], entities, ['h', 't']))

        article = Article(document, relations)
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

if __name__ == '__main__':
    linearize_boring()