from sklearn.metrics import precision_recall_fscore_support
import json 
from tai_util import *
from tai_ending import train_complexity_classifier, grade_ending, extact_training_data, grade_ending
import pandas as pd



# for now Im testing with one document, this file will eventually loop through every 
# essay and grade the lead, ending, and craft
def main():
    documents = get_docs("tai-documents-v3.json")

    # make train test splits
    train_test_partition = int(.7 * len(documents))

    # train conclusion classifier
    training_data = extact_training_data(documents[:train_test_partition])
    test_data = documents[train_test_partition:]
    concl_classifier = train_complexity_classifier(training_data)

    # get actual grades reported from teachers
    true_grades = get_true_grades(test_data)

    predicted_grades = []
    for doc in test_data:
        ending_grade = grade_ending(doc["plaintext"], concl_classifier)
        predicted_grades.append(ending_grade)






def output_accuracy(true_grades, predicted_grades):
    num_correct = 0
    for i in range(len(predicted_grades)):
        if(true_grades[i] == predicted_grades[i]):
            num_correct += 1

    print(float(num_correct / len(predicted_grades)))
 



def get_true_grades(documents):
    true_grades = []
    for doc in documents:
        grade = doc["grades"][1]["score"]["criteria"]["ending"]
        true_grades.append(grade)

    return true_grades




if(__name__ == "__main__"):
    main()


