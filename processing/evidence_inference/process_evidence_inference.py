import pandas as pd
import json
import os
import pickle

# this function takes the directory where the abstracts folder along with csv files live
# it returns an array of dictionaries representing data points(keys values: PMCID int, abstract string, relations array)


def linearize_standard(dataset):
    # i guess we should have the dataset by this point
    # dataset = pickle.load(open('../../data/evidence_inference/EI.pkl', 'rb'))
    for data_point in dataset:
        linear = ''
        for relation in data_point['relations']:
            linear += "<rel><i>" + relation[0] + "<c>" + relation[1] + "<o>" + relation[2] + "<type>" + str(relation[3])
        linear += "<end>"
        data_point['linearization'] = linear


def delinearize_standard(strings):
    data = [{} for i in range(len(strings))]
    for x, string in enumerate(strings):
        data[x]['linearization'] = string
        data_points = []
        relations = string.split("<rel>")
        # remove the first element as its an empty string
        relations.pop(0)
        for relation in relations:
            # we know the first element is our intervention so lose the 3 char tag
            relation = relation[3:]
            cot = relation.split("<c>")
            # pop the intervention off
            intervention = cot.pop(0)
            ot = cot[0].split("<o>")
            comparator = ot.pop(0)
            final_split = ot[0].split("<type>")
            outcome = final_split[0]
            if '<end>' in final_split[1]:
                relation_type = int(final_split[1][:-5])
            else:
                relation_type = int(final_split[1])
            relation_data = (intervention, comparator, outcome, relation_type)
            data_points.append(relation_data)
        data[x]['relations'] = data_points