import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def draw_cm(actual, predict, path):
    f = open(path+"/Result.txt", 'w')
    cm = confusion_matrix(actual, predict)
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig(path+'/confusion_matrix.png')

    f.write('Accuracy = '+str(accuracy_score(actual, predict))+'%\n')
    f.close()


def test_naive(test_X, test_Y, path):
    model = json.load(open(path + '/model.json'))
    predict = []
    for i in range(len(test_X)):
        print(i)
        score = []
        data_j = [line.rstrip('\n') for line in open('preProcessedDataset/' + str(test_X[i])+'.txt')]
        for c in range(5):
            score.append(0.0)
            for token in data_j:
                if token in model[str(c)].keys():
                    score[c] += np.log(float(model[str(c)][token]))
                else:
                    score[c] += np.log(float(model[str(c)]['my_unk']))
        predict.append(np.argmax(score))

    draw_cm(test_Y, predict, path)
