import os, random
import nltk
from nltk import word_tokenize, WordNetLemmatizer from nltk.corpus
import stopwords
import spam_filter as sf

def run_online(classifier, setting):
    while True:
    features = sf.get_features(raw_input('Your new email: '), setting)
    if (len(features) == 0):
        break
    print (classifier.classify(features))