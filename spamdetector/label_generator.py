import collections
import json
import os
import pickle
import re
from string import punctuation

import numpy as np
from spamdetector.lexicator import process_message
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def number_of_chars(s):
    return len(s)


def unique_chars(s):
    s2 = ''.join(set(s))
    return len(s2)


def weighted_unique_chars(s):
    return unique_chars(s) / number_of_chars(s)


def words_count(message):
    regex = re.compile(r'[{}]'.format(punctuation))
    res_list = regex.sub(' ', message)
    return len(res_list.split())


def words_counter_object(message):
    cnt = collections.Counter()
    words = message.split()
    for w in words:
        cnt[w] += 1
    return cnt


def total_words(cnt):
    sum = 0
    for k in dict(cnt).keys():
        sum += int(cnt[k])
    return sum


def is_repeated(cnt):
    for k, v in cnt.most_common(1):
        freq = v / total_words(cnt)
        if freq > 0.5:
            return 1
    return 0


def repeated_count_top_3(cnt):
    freq = 0
    for k, v in cnt.most_common(3):
        freq += v
    return freq / 3


def longest(s):
    mylist = s.split()
    if len(mylist) == 0:
        return 0
    return len(max(mylist, key=len))


def construct_feature_vec(mail_dir, parsed_dict):
    mails = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    x_size = len(mails)
    y_size = 3000
    features_vec = np.zeros((x_size, y_size))
    doc_id = 0
    for mail in mails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    for word in words:
                        for word_id, d in enumerate(parsed_dict):
                            if d[0] == word:
                                features_vec[doc_id, word_id] = words.count(
                                    word)
            doc_id += 1
    return features_vec


def generate_testing_vector(beautified_string, dictionary):
    words = beautified_string.split()
    features_vec = np.zeros(3000)
    for word in words:
        for word_id, d in enumerate(dictionary):
            if d[0] == word:
                features_vec[word_id] = words.count(word)

    return features_vec


def read_lingspam_corpus(directory_of_dataset):
    emails = [os.path.join(directory_of_dataset, file) for file in
              os.listdir(directory_of_dataset)]
    words = list()
    for email in emails:
        with open(email) as m:
            for i, line in enumerate(m):
                if i == 2:
                    content = line.split()
                    words += content
    counter = collections.Counter(words)
    to_delete = list()
    for word in counter.keys():
        if not word.isalpha() or len(word) == 1:
            to_delete.append(word)

    for i in to_delete:
        del counter[i]

    return counter.most_common(3000)


def jsonify(dictionary, path):
    with open(path, 'w') as p:
        json.dump(dictionary, p)


def train_models(directory, spam_amount, all_amount, save_classifier=True):
    train_dir = directory
    dictionary = read_lingspam_corpus(train_dir)
    print(dictionary)
    train_labels = np.zeros(all_amount)  # 702
    train_labels[spam_amount:all_amount - 1] = 1  # 351 : 701
    print("Labeling")
    train_matrix = construct_feature_vec(train_dir, dictionary)
    model1 = MultinomialNB()
    model2 = LinearSVC()
    print("Trenuje MNB")
    model1.fit(train_matrix, train_labels)
    print("Trenuje SVC")
    model2.fit(train_matrix, train_labels)
    # model_testing(model1, model2)
    if save_classifier:
        with open('spam_dictionary.dat', 'w+b') as d:
            pickle.dump(dictionary, d)
        joblib.dump(model1, 'spam_naive_bayes.pkl')
        joblib.dump(model2, 'spam_supported_vec_mach.pkl')
    return model1, model2, dictionary


def __model_testing(model1, model2, dictionary):
    test_dir = 'test-mails'
    test_matrix = construct_feature_vec(test_dir, dictionary)
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    result1 = model1.predict(test_matrix)
    print(result1)
    result2 = model2.predict(test_matrix)
    print(result2)
    print(confusion_matrix(test_labels, result1))
    print(confusion_matrix(test_labels, result2))


