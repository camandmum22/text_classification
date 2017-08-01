__author__ = 'camandmum22'

# Implementation of a Classification Decision Tree Learning Algorithm
# ===========================================================================
import numpy as np
import os


# This class represents a node in the decision tree
class tree_node:
    def __init__(self, tree, word=None, parent=None, level=0, entropy=0., leaf=False):
        self.parent = parent
        self.entropy = entropy
        self.level = level
        self.info_gain = None
        self.leaf = leaf
        self.word = word
        self.tree = tree
        if leaf:
            self.child = [0]
        else:
            self.child = [0, 0]

    def convert_to_leaf(self, class_label):
        self.child = [class_label]
        self.leaf = True

    def classify(self, value):
        if self.leaf:
            return self.child[0]
        if value <= 0:
            return self.child[0]
        else:
            return self.child[1]


    def __repr__(self):
        end_of_line = "\n%s"
        if len(self.child) < 2:
            end_of_line = "%s\n"
        feature = ''
        if self.word is not None:
            feature = 'www'
        rep = (
            ("%s Node level = %s, word = %s, entropy = %s, info_gain = %s\n" ) % (#+ end_of_line
                '\t' * (self.level + 1), self.level, feature, self.entropy, self.info_gain)) #,self.child
        return rep


class decision_tree:
    def __init__(self, depth, words):
        self.depth_limit = depth
        self.nodes = [[]]
        self.words = words

    # class_distribution() returns the proportion of each label in the classes vector
    def class_distribution(self, classes):
        labels, times = np.unique(classes, return_counts=True)
        proportions = ((times * 1.) / len(classes)).reshape((len(labels), 1))
        return labels, proportions

    # select_class() returns the most probable class in the classes vector
    def select_class(self, classes):
        labels, proportions = self.class_distribution(classes)
        max_index = np.argmax(proportions)
        return labels[max_index]

    # get_entropy() computes the entropy of the classes vector
    def get_entropy(self, classes):
        if len(classes) == 0:
            return 0.
        labels, proportions = self.class_distribution(classes)
        entropy = -np.dot(proportions.T, np.log2(proportions + pow(10,-5)))
        return entropy[0][0]

    def compute_info_gain(self):
        for level in self.nodes:
            for node in level:
                if (node.level == (self.depth_limit-1)):
                    node.info_gain = None
                elif(node.level == 0):
                    #print 'entro'
                    node.info_gain = abs(node.entropy - (node.child[0].entropy + node.child[1].entropy))

    def print_tree(self):
        map = []
        rep = self.desciption()+'\n'
        for level in self.nodes:
            for node in level:
                absent = ""
                if node.level >0:
                    if node.parent.word in map:
                        absent = "present"
                        map.remove(node.parent.word)
                    else:
                        absent = "absent"
                        map.append(node.parent.word)
                str_gain =''
                str_word = ''
                par_w =''
                if node.parent is not None:
                    par_w = node.parent.word
                if node.info_gain is not None:
                    str_gain = "info_gain = %s"%node.info_gain
                if node.word is not None:
                    str_word = "word %s = %s"%(node.word, self.words[node.word])
                rep += (("%s (%s) Node level = %s, "+str_word+", entropy = %s, "+str_gain+" parent = %s\n") % (  # + end_of_line
                        '\t' * (node.level), absent, node.level, node.entropy, par_w))
        return rep

    # find_split() finds the split with lowest entropy
    def find_split(self, words, classes):
        unique_words = np.unique(words)
        min_entropy = np.inf
        split = 0
        for u in unique_words:
            left_side = (words <= u)
            right_side = (words > u)
            entropy_left = self.get_entropy(classes[left_side])
            entropy_right = self.get_entropy(classes[right_side])
            # weighted mean of entropies
            w_entropy = (np.sum(left_side) * entropy_left + np.sum(right_side) * entropy_right) / len(classes)
            if w_entropy < min_entropy:
                min_entropy = w_entropy
                split = u
        return split, min_entropy

    # train() executes the following procedure:
    # 1) At the root node we calculate the entropy, we'll call it root.entropy
    # 2) if root.entropy == 0, we create a leaf node and stop
    # 3) else, we'll start looking for possible splits
    # 4) for each word, we calculate the smallest entropy when the tree is split using this feature
    # 5) we select the word with smallest entropy and split the tree
    # 6) we verify if we have to make the split node a leaf
    # 7) generate the 2 tree branches and go back to 1.
    def train(self, data, classes, current_depth=0, parent_node=None, is_leaf=False, is_root=False):
        if is_root:
            self.nodes = [[]]
        init_entropy = self.get_entropy(classes)
        if init_entropy <= 0. or current_depth >= self.depth_limit or is_leaf:
            leaf_node = tree_node(tree = self, parent=parent_node, level=current_depth, entropy=init_entropy, leaf=True)
            leaf_node.child[0] = self.select_class(classes)
            self.nodes[current_depth].extend([leaf_node])
            if parent_node is not None:
                leaf_node.info_gain = abs(parent_node.entropy - init_entropy)
            return leaf_node
        else:
            # iterate over all the features
            words = range(data.shape[1])
            min_entropy = np.inf
            branching = [0, 0]
            for w in words:
                # try all possible partitions along this word and return the lowest entropy(le)
                # if le is smaller that the current minimum, modify
                data_word = data[:, w]
                split, entropy = self.find_split(data_word, classes)
                if entropy < min_entropy:
                    min_entropy = entropy
                    branching = [w, split]

            new_node = tree_node(tree = self, word=branching[0], parent=parent_node, level=current_depth, entropy=min_entropy)
            self.nodes[current_depth].extend([new_node])
            if parent_node is not None:
                new_node.info_gain = abs(parent_node.entropy - min_entropy)

            if len(data.shape) == 1:  # data only contains one-word
                data = data.reshape((len(data), 1))

            left_branch = data[:, branching[0]] <= branching[1]
            right_branch = data[:, branching[0]] > branching[1]
            words.remove(branching[0])

            left_classes = classes[left_branch]
            right_classes = classes[right_branch]

            # verify if partitioning puts all samples into only one branch
            if left_branch.all():
                new_node.convert_to_leaf(self.select_class(left_classes))
                return new_node
            if right_branch.all():
                new_node.convert_to_leaf(self.select_class(right_classes))
                return new_node

            if len(words) == 0:
                left_data = data[left_branch, :]
                right_data = data[right_branch, :]
                is_leaf = True
            else:
                left_data = data[left_branch, :][:, words]
                right_data = data[right_branch, :][:, words]

            if len(left_data) == 0 or left_data.shape[1] == 0:
                new_node.convert_to_leaf(self.select_class(right_classes))
                return new_node
            if len(right_data) == 0 or right_data.shape[1] == 0:
                new_node.convert_to_leaf(self.select_class(left_classes))
                return new_node

            # validate if there is already a level below
            try:
                self.nodes[current_depth + 1]
            except IndexError:
                self.nodes.append([])

            # recursively call self again on the two children nodes
            new_node.child[0] = self.train(left_data, left_classes, current_depth=current_depth + 1,
                                           parent_node=new_node, is_leaf=is_leaf)
            new_node.child[1] = self.train(right_data, right_classes, current_depth=current_depth + 1,
                                           parent_node=new_node, is_leaf=is_leaf)
            return new_node

    # classify_sample() takes one sample and classifies it with a class
    def classify_sample(self, sample):
        node = self.nodes[0][0]
        while isinstance(node.child[0], tree_node):
            s = sample[node.word]
            sample = np.delete(sample, node.word)
            node = node.classify(s)
        return node.child[0]

    # make_prediction calculates predicted classes for the supplied dataset
    def make_prediction(self, data):
        classes = np.zeros(len(data))
        for y, x in enumerate(data):
            classes[y] = self.classify_sample(x)
        return classes

    def guess_class(self, data, expected):
        predicted = self.make_prediction(data)
        same_values = (expected == predicted)
        return (1. * np.sum(same_values)) / len(predicted)

    def desciption(self):
        n = 0
        for level in self.nodes:
            n += len(level)
        return ("Tree with %d nodes and depth = %s\n" % (n, self.depth_limit))


# --------------------------testing----------------------------------
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

    test_data = np.zeros((len(test_label), len(words)), dtype=np.int64)
    with open(FILE_TEST_DATA) as f:
        for line in f:
            (key, val) = line.split()
            key = int(key) - 1
            val = int(val) - 1
            test_data[key, val] = test_data[key, val] + 1.0

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

    depth = 4
    metrics_dt = []
    f = open("data/test/tree.txt", 'wb')
    while True:  # for depth in range(1):
        print ('dt ' + str(depth))
        tree = decision_tree(depth)
        tree.train(train_data, train_label, is_root=True)
        m_train = tree.guess_class(train_data, train_label)
        m_test = tree.guess_class(test_data, test_label)
        metrics_dt.append([m_train, m_test])
        f.write("\n\n" + str(tree))  # "\n"
        if (m_train == 1):
            break
        depth += 1
        break
    f.close()
    print(metrics_dt)

    # main()
