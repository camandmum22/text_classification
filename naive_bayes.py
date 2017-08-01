import math
import os
import numpy as np
"""
Naive Bayes with Laplace add-one Smoothing
"""

def train(labels, data, alpha):
    map_doc_lab = {}  # documents per label
    map_doc_lab_word = {}  # documents per label/word

    for x, lab in enumerate(labels):
        # we count number of documents in label l
        map_doc_lab[lab] = map_doc_lab.get(lab, 0.0) + 1
        for word in data[x].keys():
            # we count number of documents with word
            if word in map_doc_lab_word:
                map_doc_lab_word[word][lab] = map_doc_lab_word[word].get(lab, 0.0) + 1.0
            else:
                map_doc_lab_word[word] = {lab: 1.0}

    docs = len(labels)
    num_labels = len(map_doc_lab.keys())
    # Model building
    prior_prop = {}  # prior probability of labels
    for lab, val in map_doc_lab.items():
        prior_prop[lab] = math.log(val + alpha) - math.log(docs + num_labels * alpha)
    # We'll use Laplace add-one smoothing
    set_cpt = {}  # Container for CPTs of words
    for word in map_doc_lab_word.keys():
        set_cpt[word] = {}
        #set_cpt[word]['word'] = list_words[word]
        for lab in map_doc_lab.keys():
            set_cpt[word][lab] = math.log(map_doc_lab_word[word].get(lab, 0.0) + alpha) - \
                                 math.log(map_doc_lab.get(lab, 0.0) + 2 * alpha)

    return prior_prop, set_cpt


def predict(labels, data, prior_prop, set_cpt):
    num_correct = 0  # Number of correct predictions
    predicted_labels = []
    predicted_log_likelihood = []

    for x, expected_lab in enumerate(labels):
        predicted_val = []
        max_val = -float('Inf')  # -1 * sys.maxint
        for lab, val in sorted(prior_prop.items()):
            for word, num in data[x].items():
                if word in set_cpt:
                    # If the word exists, we add it to the prior probability of the label
                    val += num * set_cpt[word][lab]
            if val > max_val:
                max_val = val
                max_lab = lab
            predicted_val.append(val)
        predicted_labels.append(max_lab)
        predicted_log_likelihood.append(predicted_val)
        if expected_lab == max_lab:
            num_correct += 1

    # We calculate accuracy
    acc = 1.0 * num_correct / len(labels)
    return predicted_labels, acc, predicted_log_likelihood

#--------------------------testing----------------------------------
def main():
    os.chdir(r'C:\Users\Camilo Andres\Desktop\Variety\[waterloo]\a_3_python2')
    FILE_WORDS = "data/base/words.txt"
    FILE_TRAIN_DATA = "data/base/trainData.txt"
    FILE_TRAIN_LAB = "data/base/trainLabel.txt"
    FILE_TEST_DATA = "data/base/testData.txt"
    FILE_TEST_LAB = "data/base/testLabel.txt"

    target = 'CLASS_EXPECTED'
    words = (list(line.rstrip('\n') for line in open(FILE_WORDS, 'r')))
    test_label = np.array(list(int(line.rstrip('\n')) for line in open(FILE_TEST_LAB, 'r')), np.int64)
    train_label = np.array(list(int(line.rstrip('\n')) for line in open(FILE_TRAIN_LAB, 'r')), np.int64)

    test_data_nb = []
    old_key = 0
    vec = {}
    with open(FILE_TEST_DATA) as f:
        for line in f:
            (key, val) = line.split()
            key = int(key) - 1
            val = int(val) - 1
            if old_key != key:
                # print "%d-%d" % (old_key, key)
                old_key = key
                test_data_nb.append(vec)
                vec = {}
            if val in vec:
                vec[val] += 1.0
            else:
                vec[val] = 1.0
    test_data_nb.append(vec)

    train_data_nb = []
    old_key = 0
    vec = {}
    with open(FILE_TRAIN_DATA) as f:
        for line in f:
            (key, val) = line.split()
            key = int(key) - 1
            val = int(val) - 1
            # train_data[int(key) - 1, len(words)] = train_label[int(key) - 1]
            if old_key != key:
                old_key = key
                train_data_nb.append(vec)
                vec = {}
            if val in vec:
                vec[val] += 1.0
            else:
                vec[val] = 1.0
    train_data_nb.append(vec)
    # Document ID 1016 does not contain any word
    train_data_nb.insert(1016, {})

    prior_prop, set_cpt = train(train_label, train_data_nb, 0.02)
    predicted_labels_train, train_acc, predicted_log_train = predict(train_label, train_data_nb, prior_prop, set_cpt)
    predicted_labels_test, test_acc, predicted_log_test = predict(test_label, test_data_nb, prior_prop, set_cpt)

    '''
    f = open("data/test/nb_results.txt", 'wb')
    f.write(str(train_acc) +'\t'+ str(test_acc)+'\t'+str(prior_prop)+'\t'+str(len(predicted_labels_train))+'\t'+
            str(len(predicted_labels_test))+'\t'+str(len(predicted_log_train))+'\t'+str(len(predicted_log_test))+'\n')
    f.write(str(predicted_log_train) + '\n\n')
    f.write(str(predicted_log_test) + '\n\n')
    f.close()
    
    #print "%s %s %s\n%s\n%s\n%s\n%s\n%s"% (prior_prop,train_acc,test_acc,set_cpt,
    #                                      predicted_labels_train,predicted_labels_test,
    #                                     predicted_log_train,predicted_log_test)
    '''

def training(train_labels, train_data, alpha):
    # train 0.02      test 0.0011
    train_set = train(train_labels, train_data, 0.02)
    test_set = train(train_labels, train_data, 0.0011)
    return train_set, test_set
#main()