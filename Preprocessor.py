import os
from collections import Counter

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stoplist = stopwords.words('english')
stoplist = stoplist + ['Subject','subject','SUBJECT',':','To','From','enron','\n']

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list


def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    return [lemmatizer.lemmatize(word.lower()) for word in toker.tokenize(unicode(sentence, errors='ignore'))]


def get_features(text, setting):
    if setting == 'bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def get_frequent(all_features, spam_support_count, ham_support_count):
    spam_word_count = {}
    ham_word_count = {}
    for (words, label) in all_features:
        # Collecting unique words from email to set
        words_in_mail = words
        wordsset_in_email = set()
        for word in words_in_mail:
            wordsset_in_email.add(word)
        label_of_email = label
        # Collecting count of each word in Spam and Ham dictionary
        if label_of_email == 'spam':
            for word in wordsset_in_email:
                spam_word_count[word] = spam_word_count.setdefault(word, 0) + 1
        else:
            for word in wordsset_in_email:
                ham_word_count[word] = ham_word_count.setdefault(word, 0) + 1
    # Taking words having count greater than support counts
    spam_frequent = {word: count for (word, count) in spam_word_count.items() if count > spam_support_count}
    ham_frequent = {word: count for (word, count) in ham_word_count.items() if count > ham_support_count}
    print(spam_frequent, ham_frequent)
    return spam_frequent, ham_frequent


