import json


def train_naive(train_X, train_Y, path):
    model = {}
    total_doc = len(train_X)
    j = 0
    for c in range(5):
        text_c = {}
        total_sum = 0
        while j < total_doc and c == train_Y[j]:
            data_j = [line.rstrip('\n') for line in open('preProcessedDataset/'+str(train_X[j])+'.txt')]
            total_sum += len(data_j)
            for token in data_j:
                if token not in text_c.keys():
                    text_c[token] = 0
                text_c[token] += 1
            j += 1
        total_keys = len(text_c.keys())
        for token in text_c.keys():
            text_c[token] = float(text_c[token]) / float(total_sum + total_keys)
        text_c['my_unk'] = float(1) / float(total_sum + total_keys)
        model[str(c)] = text_c
        print ('class '+str(c) + ' done')
    with open(path+'/model.json', 'w') as fp:
        json.dump(model, fp)
