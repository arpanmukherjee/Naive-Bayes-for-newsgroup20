import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

path = '/Users/arpn/Google Drive/Semester2/Information Retrieval/IR-Assignment3/dataset'
ps = PorterStemmer()
stop_words = stopwords.words('english') + list(string.punctuation)


def file_write(file_name, ar, path=''):
    f = open(path+str(file_name), 'w')
    for i in ar:
        try:
            f.write(i+'\n')
        except:
            print('Ignore val')
    f.close()


dirs = os.listdir(path)
if '.DS_Store' in dirs:
    dirs.remove('.DS_Store')
docIdList = []
groundTruth = []
docId = 0
class_label = 0
for folder in dirs:
    files = os.listdir(path+'/'+folder)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    for id in files:
        docIdList.append(folder+'/'+id)
        groundTruth.append(str(docId)+' '+str(class_label))
        fileData = [line.rstrip('\n') for line in open(path + '/' + folder + '/' + id)]
        lenDoc = 0
        for line in fileData:
            if line[:7] == "Lines: ":
                lenDoc = int(line[7:])
                break
        fileData = fileData[-lenDoc:]
        cleaned_data = []
        tokens = []
        for sentence in fileData:
            try:
                temp_tokens = [i for i in nltk.word_tokenize(sentence.lower().strip()) if
                               i not in stop_words]
                for word in temp_tokens:
                    if all(char in stop_words for char in word):
                        continue
                    else:
                        tokens.append(word)
            except:
                print("Error")
            for w in tokens:
                try:
                    cleaned_data.append(ps.stem(w))
                except:
                    print w
        file_write(str(docId)+'.txt', cleaned_data, 'preProcessedDataset/')
        docId += 1
    class_label += 1

file_write('docId.txt', docIdList)
file_write('groundTruth.txt', groundTruth)
