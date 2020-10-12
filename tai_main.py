from sklearn.metrics import confusion_matrix
import json 
from tai_util import *
from tai_ending import train_complexity_classifier, grade_ending, extact_training_data, grade_ending
import math
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

    confusion_matrix(true_grades, predicted_grades)

    # change confusion matrix to data frame and output to csv
    cm_as_df=cm2df(confusion_matrix,list(set(true_grades)))
    cm_as_df.to_csv("ending_confusion_matrix")
    



def get_true_grades(documents):
    true_grades = []
    for doc in documents:
        grade = doc["grades"][1]["score"]["criteria"]["ending"]
        grade = math.floor(grade)
        true_grades.append(grade)

    return true_grades



def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]



if(__name__ == "__main__"):
    main()


