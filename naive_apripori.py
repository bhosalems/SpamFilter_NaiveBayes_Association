from __future__ import print_function, division
import numpy
import nltk
import math
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
        return [word for word in preprocess(text) if not word in stoplist]


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

def evaluate(train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior, ham_prior):
    # check how the classifier performs on the training and test sets
    train_accuracy = classify(train_set,raw_spam_prob,raw_ham_prob, spam_total, ham_total,spam_vocab, ham_vocab, spam_prior, ham_prior)
    test_accuracy = classify(test_set,raw_spam_prob,raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior, ham_prior)
    print ('Accuracy on the training set = '+str(train_accuracy))
    print ('Accuracy of the test set = '+str(test_accuracy))
    # check which words are most informative for the classifier
    
def classify(data_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior, ham_prior):
    all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
    total_mail=0
    cnt1 = 0
    cnt2 = 0
    correct_count = 0.0000000
    for(features,label) in all_features:
        spam_prob = 1.000000
        ham_prob = 1.000000
        is_spam = False 
        for word in features: 
            #Handling probability of non occuring words with laplaces
                
            #try:
            raw_spam_prob[word]
            cnt1+=1
            spam_prob = (numpy.log(spam_prior) + numpy.log(raw_spam_prob[word]))*spam_prob
            #except KeyError:
            #   cnt2+=1
            #    raw_spam_prob[word]= (1/(spam_total+spam_vocab+1))
            #    spam_prob = (numpy.log(spam_prior) + numpy.log(raw_spam_prob[word]))*spam_prob
            #try:
            raw_ham_prob[word]
            ham_prob = (numpy.log(ham_prior) + numpy.log(raw_ham_prob[word]))*ham_prob
            #except KeyError:
            #    raw_ham_prob[word]= (1/(ham_total+ham_vocab+1))
            #    ham_prob = (numpy.log(ham_prior) + numpy.log(raw_ham_prob[word]))*ham_prob
            
            print(word,raw_spam_prob[word],raw_ham_prob[word])
        
        if(spam_prob>ham_prob):
            is_spam = True
        else:
            is_spam = False
        if ((label==spam) and is_spam):
            correct_count+=1;
        if ((label==ham) and not(is_spam)):
            correct_count+=1; 
        total_mail+=1
    print('correct count' +str(correct_count), 'cnt1'+ str(cnt1), 'cnt2'+ str(cnt2))
    return (correct_count/total_mail)

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

    # train the our own naivebayes classifier and collect dictionary of raw probabilities of words
    t = owncl.train(all_features, 0.8)
    train_set = t[0]
    test_set = t[1] 
    raw_spam_prob = t[2]
    raw_ham_prob = t[3]
    spam_total = t[4]
    ham_total = t[5]
    spam_vocab = t[6]
    ham_vocab = t[7]
    spam_prior = t[8]
    ham_prior = t[9]
    #for words, label in train_set:
    #    print(label)
    #print(raw_spam_prob)
    #Replacing raw probabilities of frequent words
    #Using following function
    #Papri (word|spam/ ham) =nf/(napr + vocabulary)
    #nf number of occurrences of a frequent word in a coming email, napr=number of occurrences of all frequent word in a coming email.
    #vocabulary = number of words in the spam/ham final frequent item set.
    
    for spam_frequent_word, count in spam_frequent.iteritems():
        raw_spam_prob[spam_frequent_word] = count/(spam_total+spam_vocab)

    for ham_frequent_word, count in ham_frequent.iteritems():
        raw_ham_prob[ham_frequent_word] = count/(ham_total+ham_vocab)
    

    # evaluate its performance
    #evaluate(train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior, ham_prior)