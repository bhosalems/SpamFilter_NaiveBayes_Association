import os
from collections import Counter

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    
    #Initialize spam and ham mail counts
    spam_count = 0.000000
    ham_count = 0.000000
    
    #Initialize spam words count and ham words count
    spam_word_count = {}
    ham_word_count = {}

    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    
    # train the classifier

    #claculating prior probabilities of class
    for (words, label) in train_set:
        if(label == 'spam'):
            spam_count+=1
        else:
            ham_count+=1
    spam_prior = spam_count/len(train_set)
    ham_prior = ham_count/len(train_set)

    #Calculating likelihood of words in train_set
    
    #Calculating word count in each class
    for (words, label) in train_set:
        if label_of_email == 'spam':
            for word in words:
                spam_word_count[word] = spam_word_count.setdefault(word,0)+1
        else:
            for word in words:
                ham_word_count[word] = ham_word_count.setdefault(word,0)+1

