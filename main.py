from NaiveBayes import train
from ModifiedNaiveBayes import customized_train
if __name__ == "__main__":
    print('Do you want to improve train with apriori? Y or N')
    choice = raw_input()
    if(choice == 'N'):
        train()
    else:
        customized_train()