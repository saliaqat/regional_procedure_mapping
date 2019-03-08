from gensim.models import KeyedVectors
import re
from data_reader import DataReader
from data_manipulator import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# clean for BioASQ
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

tokens = bioclean('This is a sentence w/o you!')
print(tokens)

data_reader = DataReader()
df = data_reader.get_all_data()
df = df[['RIS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION', 'ON WG IDENTIFIER']]

# drop missing rows
df= df.dropna()
df['text'] = df[['RIS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION']].apply(lambda x: ' '.join(x), axis=1)
df = df.drop(['RIS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION'], axis=1)
df = df.rename(columns={'ON WG IDENTIFIER': 'target'}).values

targets =  df[:, 0]
words = df[:, 1]
vectorizer = CountVectorizer(tokenizer=bioclean)
v = vectorizer.fit_transform(words)
additional_vocab = vectorizer.vocabulary_

# load PubMed embedding and the vocabulary
word_vectors = KeyedVectors.load_word2vec_format('../pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)

# create new vocabulary so that we can add more words later
new_vocab = {}
for word in word_vectors.vocab:
    new_vocab[word] = word_vectors.vocab[word].index

# remove the overlapping words
for word in new_vocab:
    if word in additional_vocab:
        additional_vocab.pop(word, None)
        
# add the additional words to the existing vocabulary
i = len(new_vocab)
for word in additional_vocab:
    new_vocab[word] = i
    i += 1
    
#labels = {}
#for i in range(len(df)):
#    tokenized = bioclean(df[i][1])
#    if df[i][0] in labels: 
#        labels[df[i][0]].append(tokenized)       
#    else:
#        labels[df[i][0]] = [tokenized]
#   
#final_labels = {}    
#for label in labels:
#    if len(labels[label]) >= 20:
#        final_labels[label] = labels[label]
#        
#
#for label in final_labels:
#    for s in final_labels[label]:
#        vector = np.zeros(200)
#        for word in s:
#            if word != 'renalbil' and word != 'or-embolectomy':
#                vector += word_vectors[word]
#    
#    
##print(df['ON WG IDENTIFIER'].nunique())
#print('Avg no. of samples for a class: {}'.format(round(np.mean(counts), 2)))
#