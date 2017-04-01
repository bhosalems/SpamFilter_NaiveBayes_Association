import random

import Preprocessor

spam_word_count = {}
ham_word_count = {}

words_in_spam = 0
words_in_ham = 0

raw_spam_prob = {}
raw_ham_prob = {}


def train(samples_proportion=0.8):
    global words_in_ham, ham_word_count, words_in_spam, spam_word_count, raw_ham_prob, raw_spam_prob

    ham, spam = read_spam_ham()

    print ("Spam size: " + str(len(spam)) + " Ham size: " + str(len(ham)))

    all_emails = append_ham_and_spam(ham, spam)

    random.shuffle(all_emails)

    print('Corpus size = ' + str(len(all_emails)) + ' emails')

    features = [(Preprocessor.get_features(email, 'bow'), label) for (email, label) in all_emails]

    print('Collected ' + str(len(features)) + ' feature sets')

    '''
    # define Support value in %
    support = 10
    spam_support_count = (spam_size * 10) / 100;
    ham_support_count = (ham_size * 10) / 100;
    print('Spam support count:' + str(spam_support_count))
    print('Ham support count:' + str(ham_support_count))
    # get the spam frequent itemset and ham frequent itemset
    # spam_frequent, ham_frequent = get_frequent(all_features, spam_support_count, ham_support_count)
    # train the our own naivebayes classifier and collect dictionary of raw probabilities of words
    '''

    train_size = int(len(features) * samples_proportion)

    train_set, test_set = features[:train_size], features[train_size:]

    ham_mail_count, spam_mail_count = mails_in_ham_spam(train_set)

    spam_prior = 1.0 * spam_mail_count / len(train_set)
    ham_prior = 1.0 * ham_mail_count / len(train_set)

    words_in_ham, words_in_spam = frequency_in_ham_spam(train_set)

    spam_vocab = len(spam_word_count)
    ham_vocab = len(ham_word_count)

    t = get_probabilities_in_each_class(ham_prior, words_in_ham, ham_vocab, ham_word_count, raw_ham_prob, raw_spam_prob,
                                        spam_prior, words_in_spam, spam_vocab, spam_word_count, test_set, train_set)

    ham_prior, words_in_ham, ham_vocab, raw_ham_prob, raw_spam_prob, spam_prior, words_in_spam, spam_vocab, test_set, train_set = get_parameters(
        t)

    evaluate(train_set, test_set, raw_spam_prob, raw_ham_prob, words_in_spam, words_in_ham, spam_vocab, ham_vocab,
             spam_prior,
             ham_prior)


def get_probabilities_in_each_class(ham_prior, ham_total, ham_vocab, ham_word_count, raw_ham_prob, raw_spam_prob,
                                    spam_prior, spam_total, spam_vocab, spam_word_count, test_set, train_set):
    for (words, label) in train_set:
        if label == 'spam':
            for word in words:
                raw_spam_prob[word] = (float)((spam_word_count.setdefault(word, 0)) / (spam_total))
        else:
            for word in words:
                raw_ham_prob[word] = (float)((ham_word_count.setdefault(word, 0)) / (ham_total))
    return [train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior,
            ham_prior]


def frequency_in_ham_spam(train_set):
    global spam_word_count, ham_word_count

    spam_total = 0
    ham_total = 0

    for (words, label) in train_set:
        if label == 'spam':
            for word in words:
                spam_word_count[word] = spam_word_count.setdefault(word, 0.0) + 1
                spam_total += 1
        else:
            for word in words:
                ham_word_count[word] = ham_word_count.setdefault(word, 0.0) + 1
                ham_total += 1
    return ham_total, spam_total


def mails_in_ham_spam(train_set):
    spam_count = 0
    ham_count = 0
    for (words, label) in train_set:
        if (label == 'spam'):
            spam_count += 1
        else:
            ham_count += 1
    return ham_count, spam_count


def classify(data_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior,
             ham_prior):
    total_mail = 0
    correct_count = 0.0000000
    for (features, label) in data_set:
        spam_prob = 1.000000
        ham_prob = 1.000000

        is_spam = check_spam(features, ham_prior, ham_prob, ham_total, ham_vocab, raw_ham_prob, raw_spam_prob,
                             spam_prior, spam_prob, spam_total, spam_vocab)

        if (label == 'spam') and is_spam:
            correct_count += 1;

        if (label == 'ham') and not (is_spam):
            correct_count += 1;
        total_mail += 1

    print('correct count' + str(correct_count))
    return correct_count / total_mail


def check_spam(features, ham_prior, ham_prob, ham_total, ham_vocab, raw_ham_prob, raw_spam_prob, spam_prior, spam_prob,
               spam_total, spam_vocab):
    for word in features:
        # Handling probability of non occuring words with laplaces

        try:
            spam_prob = raw_spam_prob[word] * spam_prob
        except KeyError:
            raw_spam_prob[word] = (1 / (spam_total + spam_vocab + 1))
            spam_prob = raw_spam_prob[word] * spam_prob
        try:
            ham_prob = raw_ham_prob[word] * ham_prob
        except KeyError:
            raw_ham_prob[word] = (1 / (ham_total + ham_vocab + 1))
            ham_prob = raw_ham_prob[word] * ham_prob
    spam_prob *= spam_prior
    ham_prob *= ham_prior
    if spam_prob > ham_prob:
        is_spam = True
    else:
        is_spam = False
    return is_spam


def evaluate(train_set, test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab, spam_prior,
             ham_prior):

    train_accuracy = classify(train_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab,
                              spam_prior, ham_prior)

    test_accuracy = classify(test_set, raw_spam_prob, raw_ham_prob, spam_total, ham_total, spam_vocab, ham_vocab,
                             spam_prior, ham_prior)
    print('Accuracy on the training set = ' + str(train_accuracy))
    print('Accuracy of the test set = ' + str(test_accuracy))

    #TODO: check which words are most informative for the classifier


def read_spam_ham():
    spam = Preprocessor.init_lists('enron_full/spam/')
    ham = Preprocessor.init_lists('enron_full/ham/')
    return ham, spam


def get_parameters(t):
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
    return ham_prior, ham_total, ham_vocab, raw_ham_prob, raw_spam_prob, spam_prior, spam_total, spam_vocab, test_set, train_set


def append_ham_and_spam(ham, spam):
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    return all_emails


def get_spam_ham_features(ham_emails, spam_emails):
    spam_features = [(Preprocessor.get_features(email, 'bow'), label) for (email, label) in spam_emails]
    ham_features = [(Preprocessor.get_features(email, 'bow'), label) for (email, label) in ham_emails]
    return ham_features, spam_features
