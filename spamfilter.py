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
#Only for validating some message in text
if __name__ == "__main__":
    soam =sf.init_lists('enrn1/sapam/')
    ham = sf.init_lists('enron1/ham/')
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    random.shuffle(all_emails)
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')

    all_features = [(sf.get_features(email, ''), label) for (email, label) in all_emails]
    train_set, test_set, classifier = sf.train(all_features, 1.0)

    #classify your new email
    run_online(classifier, "")
#For detecting spam in whole mail
def detect_spam(folder, classifier, setting):
        output = {}
        file_list = os.listdir(folder)
        for a_file in file_list:
            f = open(folder + a_file, 'r')
            features = sf.get_features(f.read(), setting)
            f.close()
            output[a_file] = classifier.classify(features)
            for item in output.keys():
                print(item + '\t' + output.get(item))
                
