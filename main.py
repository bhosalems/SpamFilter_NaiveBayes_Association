from ModifiedNaiveBayes import customized_train
from NLTK_NaiveBayes import train as NLTK_train
if __name__ == "__main__":
    print('Do you want to improve train with apriori? Y or N')
    choice = raw_input()
    if(choice == 'N'):
        NLTK_train()
    else:
        customized_train()