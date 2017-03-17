from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import own_naivebayes as owncl
#import own_naivebayes.classifier as owncl
#Driver program to run own naive bayes with apriory
stoplist = stopwords.words('english')

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
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def get_frequent(all_features, spam_support_count, ham_support_count):
    spam_word_count = {}
    ham_word_count = {}
    for (words, label) in all_features:
        #Collecting unique words from email to set
        words_in_mail = words
        wordsset_in_email=set()
        for word in words_in_mail:
            wordsset_in_email.add(word)
        label_of_email = label
        #Collecting count of each word in Spam and Ham dictionary
        if label_of_email == 'spam':    
            for word in wordsset_in_email:
                spam_word_count[word] = spam_word_count.setdefault(word,0)+1
        else:
            for word in wordsset_in_email:
                ham_word_count[word] = ham_word_count.setdefault(word,0)+1
    #Taking words having count greater than support counts
    spam_frequent = {word:count for (word,count) in spam_word_count.items() if count > spam_support_count}             
    ham_frequent = {word:count for (word,count) in ham_word_count.items() if count > ham_support_count}
    return spam_frequent, ham_frequent

def evaluate(train_set, test_set, classifier):
    # check how the classifier performs on the training and test sets
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    # check which words are most informative for the classifier
    classifier.show_most_informative_features(20)

if __name__ == "__main__" :
    # initialise the data
    spam = init_lists('enron1/spam/')
    ham = init_lists('enron1/ham/')
    spam_size = len(spam) 
    ham_size = len(ham)
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    random.shuffle(all_emails)
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')

    # extract the features
    all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
    print ('Collected ' + str(len(all_features)) + ' feature sets')

    #define Support value in %
    support = 10 
    spam_support_count = (spam_size * 10)/ 100;
    ham_support_count = (ham_size * 10)/ 100;
    print('Spam support count:'+str(spam_support_count))
    print('Ham support count:'+str(ham_support_count))

    #get the spam frequent itemset and ham frequent itemset
    spam_frequent, ham_frequent = get_frequent(all_features, spam_support_count, ham_support_count)
    #print(spam_frequent)
    #print("\n")
    #print(ham_frequent)


    # train the our own naivebayes classifier and collect dictionary of raw probabilities of words
    train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab= owncl.train(all_features, 0.8)

    #Replacing raw probabilities of frequent words
    #Using following function
    #Papri (word|spam/ ham) =nf/(napr + vocabulary)
    #nf number of occurrences of a frequent word in a coming email, napr=number of occurrences of all frequent word in a coming email.
    #vocabulary = number of words in the spam/ham final frequent item set.
    
    for (spam_frequent_word, count) in spam_frequent
        raw_spam_prob[spam_frequent_word] = count/(spam_total+spam_vocab)

    for (ham_frequent_word, count) in ham_frequent
        raw_ham_prob[ham_frequent_word] = count/(ham_total+ham_vocab)

    # evaluate its performance
    #evaluate(train_set, test_set, classifier)