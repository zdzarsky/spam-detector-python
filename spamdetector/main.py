import os

import spamdetector.lexicator as lexicator
from spamdetector.label_generator import train_models, generate_testing_vector
from sklearn.externals import joblib
import pickle
import requests
import tempfile


def get_bayes_from_web():
    req = requests.get(
        "https://github.com/zdzarsky/spam-detector-python/blob/master"
        "/spamdetector/spam_naive_bayes.pkl?raw=true")

    return joblib.load(req.text)


def get_svm_from_web():
    req = requests.get(
        "https://github.com/zdzarsky/spam-detector-python/blob/master"
        "/spamdetector/spam_supported_vec_mach.pkl?raw=true")
    return joblib.load(req.text)


def get_dictionary_from_web():
    req = requests.get(
        "https://github.com/zdzarsky/spam-detector-python/blob/master"
        "/spamdetector/spam_dictionary.dat?raw=true")
    return pickle.load(req.text)


def train_on_own_corpus(directory, false, all_set, save_classifs):
    try:
        train_models(directory, false, all_set, save_classifier=save_classifs)
    except FileNotFoundError:
        print("Unable to find dataset. Visit https://github.com/zdzarsky/"
              "spam-detector-python and read README.md carefully, if not "
              "working still please notice Issue")


def train_bayes_on_default():
    try:
        train_models('train-mails', 351, 702, save_classifier=True)
    except FileNotFoundError:
        print("Unable to find directory with training dataset. Visit:"
              "https://github.com/zdzarsky/spam-detector-python"
              "and download missing directories.")


def is_spam(subject, message, dictionary_path, bayes_path, svm_path):
    parsed_message = lexicator.process_message(subject, message)
    bayes = joblib.load(bayes_path)
    svm = joblib.load(svm_path)
    with open(dictionary_path, 'rb') as dp:
        dictionary = pickle.load(dp)
    vector = generate_testing_vector(parsed_message, dictionary)
    vector = vector.reshape(1, -1)
    result_b = bayes.predict(vector)
    result_b = result_b[0]
    result_svm = svm.predict(vector)
    result_svm = result_svm[0]
    est = (result_b + result_svm) / 2
    if 0 < est < 1:
        print("It's hard to estimate spam factor")
    elif est == 0:
        print("Message is reliable")
    elif est == 1:
        print("Message is spam")


def is_spam_online(subject, message):
    parsed_message = lexicator.process_message(subject, message)
    svm = get_svm_from_web()
    bayes = get_bayes_from_web()
    dictionary = get_dictionary_from_web()
    vector = generate_testing_vector(parsed_message, dictionary)
    vector = vector.reshape(1, -1)
    result_b = bayes.predict(vector)
    result_b = result_b[0]
    result_svm = svm.predict(vector)
    result_svm = result_svm[0]
    est = (result_b + result_svm) / 2
    if 0 < est < 1:
        print("It's hard to estimate spam factor")
    elif est == 0:
        print("Message is reliable")
    elif est == 1:
        print("Message is spam")


if __name__ == '__main__':
    # is_spam("Hello", "Hello", 'spam_dictionary.dat', 'spam_naive_bayes.pkl',
    #       'spam_supported_vec_mach.pkl')
    #  is_spam_online("Hello", "Hello")
    get_bayes_from_web()
