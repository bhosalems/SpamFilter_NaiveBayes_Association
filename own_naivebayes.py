import os
from collections import Counter

#Own naive bayes implementation
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    
    #Initialize spam and ham mail counts
    spam_count = 0.000000
    ham_count = 0.000000
    
    #Initialize spam words count and ham words count and total words 
    spam_word_count = {}
    ham_word_count = {}
    spam_total = 0
    ham_total = 0

    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    
    # train the classifier

    #Claculating prior probabilities of class
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
        if label == 'spam':
            for word in words:
                spam_word_count[word] = spam_word_count.setdefault(word,0.000000)+1
                spam_total+=1
        else:
            for word in words:
                ham_word_count[word] = ham_word_count.setdefault(word,0.000000)+1
                ham_total+=1

        #Vocabulary of classes
    spam_vocab = len(spam_word_count) 
    ham_vocab = len(ham_word_count)

    print('Spam mail count in train set:'+str(spam_count))
    print('Ham mail count in train set:'+str(ham_count))
    
    print(train_set)

    #Calculating raw probabilities
    #Initializing spam and ham raw probabilities
    raw_spam_prob = {}
    raw_ham_prob = {}
    i=0;
    for (words,label) in train_set:
        if label == 'spam':
            for word in words:
                #Applying Laplace's solution
                raw_spam_prob[word] = (float)((spam_word_count.setdefault(word,0.000000)+1)/(spam_total+spam_vocab))
        else:
            for word in words:
                #Applying Laplace's solution
                raw_ham_prob[word] = (float)((ham_word_count.setdefault(word,0.000000)+1)/(ham_total+ham_vocab))
    
    return [train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior, ham_prior]