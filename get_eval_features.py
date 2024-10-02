# get_eval_features.py
import json
import re

# Function to get the length of the article 
def getArticleLength(doc):
    return len(doc['text'])

# Function to get number of sentences 
def getNumSentence(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences)

# Function to get the avg word length
def getAvgWordLength(doc):
    text = re.sub(r'[^\w\s]', '', doc['text'])
    words = text.split()
    return sum(len(word) for word in words) / len(words)

# Function to get unique entries (assuming entities are in the 'true_relations' key)
def getUniqueEntries(doc):
    unique_entries = set()
    for relation in doc['true_relations']:
        for entity in relation['entities'].values():
            unique_entries.add(entity['span'])
    return len(unique_entries)

# Function to get number of relations
def getNumRelations(doc):
    return len(doc['true_relations'])

# Function to get relation density over the whole article
def getRelationDensity(doc):
    article_length = getArticleLength(doc)
    number_of_relations = getNumRelations(doc)
    return number_of_relations / article_length if article_length > 0 else 0

# Function to get relation density per sentence
def getRelationDensityPerSentence(doc):
    sentences = re.split(r'[.!?]+', doc['text'])
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    num_sentences = len(sentences)
    num_relations = getNumRelations(doc)
    return num_relations / num_sentences if num_sentences > 0 else 0

# Main Function
def get_input_specific_stats(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    stats = []
    for doc in data:
        article_length = getArticleLength(doc)
        num_sentences = getNumSentence(doc['text'])
        avg_word_length = getAvgWordLength(doc)
        unique_entries = getUniqueEntries(doc)
        num_relations = getNumRelations(doc)
        relation_density = getRelationDensity(doc)
        relation_density_per_sentence = getRelationDensityPerSentence(doc)

        # Collect metrics for the document
        doc_stats = {
            'article_length': article_length,
            'num_sentences': num_sentences,
            'avg_word_length': avg_word_length,
            'unique_entries': unique_entries,
            'num_relations': num_relations,
            'relation_density': relation_density,
            'relation_density_per_sentence': relation_density_per_sentence
        }
        stats.append(doc_stats)
    
    return stats
