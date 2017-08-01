from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy
import numpy as np
import math
import os
from tree import *
from naive_bayes import *

def get_accuracy(expected, predicted):
    num = 0.0
    for x in range(len(expected)):
        if(expected[x]==predicted[x]):
            num += 1
    return num/float(len(expected))

def main():
    # -----------------------------------------------------------------------------
    # We load the datasets
    # -----------------------------------------------------------------------------
    os.chdir(r'C:\Users\Camilo Andres\Desktop\Variety\[waterloo]\text_classification')
    FILE_WORDS = "data/base/words.txt"
    FILE_TRAIN_DATA = "data/base/trainData.txt"
    FILE_TRAIN_LAB = "data/base/trainLabel.txt"
    FILE_TEST_DATA = "data/base/testData.txt"
    FILE_TEST_LAB = "data/base/testLabel.txt"

    target = 'CLASS_EXPECTED'
    words = (list(line.rstrip('\n') for line in open(FILE_WORDS, 'r')))
    test_label = np.array(list(int(line.rstrip('\n')) for line in open(FILE_TEST_LAB, 'r')), np.int64)
    train_label = np.array(list(int(line.rstrip('\n')) for line in open(FILE_TRAIN_LAB, 'r')), np.int64)

    test_data = np.zeros((len(test_label), len(words)), dtype=np.int64)
    test_data_nb = []
    old_key = 0
    vec = {}
    with open(FILE_TEST_DATA) as f:
        for line in f:
            (key, val) = line.split()
            key = int(key) - 1
            val = int(val) - 1
            test_data[key, val] = test_data[key, val]+1.0
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

    train_data = np.zeros((len(train_label), len(words)), dtype=np.int64)
    train_data_nb = []
    old_key = 0
    vec = {}
    with open(FILE_TRAIN_DATA) as f:
        for line in f:
            (key, val) = line.split()
            key = int(key) - 1
            val = int(val) - 1
            train_data[key, val] = train_data[key, val] + 1.0
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

    # -----------------------------------------------------------------------------
    # We use Decision Tree Learning and Naive Bayes algorithms from sklearn
    # -----------------------------------------------------------------------------
    depth = 4
    metrics_skt_nb = []
    metrics_skt_dt = []

    # Decision Tree Learning
    while (True):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        model.fit(train_data, train_label)
        predicted_train = model.predict(train_data)
        predicted_test = model.predict(test_data)
        m_train = get_accuracy(train_label, predicted_train)
        m_test = get_accuracy(test_label, predicted_test)
        
        metrics_skt_dt.append([m_train, m_test])
        if (m_train == 1):
            break
        depth += 1
        break

    # Naive Bayes algorithm
    model = BernoulliNB(alpha=1, binarize=None)
    model.fit(train_data, train_label)
    predicted_train = model.predict(train_data)
    predicted_test = model.predict(test_data)
    m_train = get_accuracy(train_label, predicted_train)
    m_test = get_accuracy(test_label, predicted_test)
    metrics_skt_nb.append([m_train, m_test])
    
    # -----------------------------------------------------------------------------
    # We use the Decision Tree Learning algorithm implemented from scratch
    # -----------------------------------------------------------------------------
    depth = 4
    metrics_dt = []
    f = open("data/test/tree.txt", 'ab')
    while True: #for depth in range(1):
        print ('tree depth = ' + str(depth))
        tree = decision_tree(depth,words)
        tree.train(train_data, train_label, is_root=True)
        m_train = tree.guess_class(train_data, train_label)
        m_test = tree.guess_class(test_data, test_label)
        metrics_dt.append([m_train, m_test])
        #f.write("\n\n" + tree.print_tree())  # "\n"
        if (m_train == 1):
            break
        depth += 1
        break
    f.close()
    tree.compute_info_gain()

    # -----------------------------------------------------------------------------
    # We use the Naive Bayes algorithm implemented from scratch
    # -----------------------------------------------------------------------------
    metrics_nb = []
    train_set, test_set = training(train_label, train_data_nb, alpha=1.0)
    predicted_labels_train, train_acc , _ = predict(train_label, train_data_nb, train_set[0],train_set[1])
    predicted_labels_test, test_acc, _ = predict(test_label, test_data_nb, test_set[0],test_set[1])
    metrics_nb.append([train_acc, test_acc])

    print "naive bayes train_acc = %s, test_acc = %s" % (train_acc, test_acc)
    set_cpt = test_set[1]
    disc_words = {}
    for cpt in set_cpt.keys():
        word = get_word(cpt, words)
        value = abs(math.log(abs(set_cpt[cpt][1])) - math.log(abs(set_cpt[cpt][2])))
        if value in disc_words:
            disc_words[value].append(word)
        else:
            disc_words[value] = [word]
    sorted_words = sorted(disc_words)
    sorted_words.reverse()
    print("rating\tdiscriminative by\tword(s)")
    for s in range(10):
        print("%d\t%s\t%s" % ((s + 1), sorted_words[s], disc_words[sorted_words[s]]))

    # -----------------------------------------------------------------------------
    # We compare sklearn and the scratch implementations
    # -----------------------------------------------------------------------------
    print('metrics_skt_dt '+str(metrics_skt_dt))
    print('metrics_dt '+str(metrics_dt))
    print('metrics_skt_nb ' + str(metrics_skt_nb))
    print('metrics_nb '+str(metrics_nb))

def get_word(id, words):
    if id == 24:
        id = 570
    if id == 152:
        id = 211
    return '{%s] %s'%(id,words[id])

if __name__ == "__main__":
    main()
