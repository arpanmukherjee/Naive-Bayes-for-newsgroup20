import numpy as np

path = '/Users/arpn/Google Drive/Semester2/Information Retrieval/IR-Assignment3/preProcessedDataset/'
doc_len = 5000
tf_idf = {}
for id in range(doc_len):
    word_list = [line.rstrip('\n') for line in open(str(id)+'.txt')]
    for word in word_list:
        if word in tf_idf.keys():
            tf_idf[word][id] += 1
        else:
            tf_idf[word] = np.zeros(doc_len)
            tf_idf[word][id] += 1
