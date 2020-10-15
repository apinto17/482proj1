import nltk
import re
from nltk import RegexpParser
from nltk.tree import Tree
from tai_util import get_docs, clean_document, tokenize_sentences
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json 
from tai_util import *
import pandas as pd
import numpy as np


# average features for grade 5 essay (obtained from predict.py)
predict5th_feature_dict = {}
predict5th_feature_dict["imperative"] = True 
predict5th_feature_dict["num_chars"] = 708.86646948
predict5th_feature_dict["num_words"] = 206.37622155
predict5th_feature_dict["num_sents"] = 9.16172628

predict5th_feature_dict["avg_chars_per_word"] = 5.12625721
predict5th_feature_dict["avg_words_per_sent"] = 45.54967897


# average features for grade 6 essay (obtained from predict.py)
predict6th_feature_dict = {}
predict6th_feature_dict["imperative"] = True 
predict6th_feature_dict["num_chars"] = 846.1004423
predict6th_feature_dict["num_words"] = 247.46135645
predict6th_feature_dict["num_sents"] = 10.75622246

predict6th_feature_dict["avg_chars_per_word"] = 5.02876279
predict6th_feature_dict["avg_words_per_sent"] = 51.43039091


# list of emotional words
emotional_words = ["content", "bother", "uncomfortable", "shy",
                    "glad", "blah", "annoy", "startle", "curious",
                    "please", "blue", "irritate", "uneasy", "sass",
                    "playful", "gloomy", "mean", "tense", "weird",
                    "cheerful", "rotten", "crabby", "anxious", "confuse",
                    "giddy", "sad", "cranky", "worry", "moody",
                    "calm", "unhappy", "grumpy", "concern", "small",
                    "comfortable", "empty", "grouchy", "timid", "quiet",
                    "cozy", "jealous", "safe", "embarrass",
                    "relax", "guilty", "cold",
                    "confident", "responsible", "strong", "concern",
                    "peaceful", "asham", "caring", "bored",
                    "delight", "disappoint", "disgust", "alarm",
                    "jolly", "hurt", "scare",
                    "bubbly", "lost", "mad", "afraid",
                    "tickle", "sorry", "angry", "frighten",
                    "silly", "ashame", "smoldering", "fearful",
                    "frisky", "lonely", "hot", "threaten",
                    "happy", "down", "frustrate", "trembly",
                    "proud", "hopeless", "impatient", "shaken",
                    "joyful", "discourag", "disturb",
                    "excite", "awful", "thankful", "great", 
                    "love", "blissful", "grateful",
                    "satisfy", "alive", "miserable", "dread",
                    "sparkle", "crush", "fuming", "panic",
                    "wonderful", "helpless", "infuriate", "terrify",
                    "ecstatic", "depress", "destructive", "horrible",
                    "terrific", "withdrawn", "explosive", "petrify",
                    "jubilant", "heartbroken", "violent"]





def grade_ending(doc, classifier):
    # get the conlcusion of the essay
    concl = get_conclusion(doc)

    # get basic metrics of conclusion
    num_words = len(nltk.word_tokenize(concl))

    # classify general complexity of essay
    complexity = classify_complexity(concl, classifier)

    # classify conclusions using these metrics
    return str(complexity)
    


def get_conclusion(doc):
    last_par = doc.split("\n")[-1]
    return last_par.strip()



def classify_complexity(concl, classifier):
    feature_dict = make_feature_dict(concl)
    if(feature_dict is None):
        return 0
    elif(predict5th(feature_dict)):
        return 5
    elif(predict6th(feature_dict)):
        return 6
    else:
        return classifier.classify(feature_dict)



def predict5th(feature_dict):
    is_5th_grade_essay = True
    if(feature_dict["imperative"] == False):
        is_5th_grade_essay = False
    if(abs(feature_dict["num_chars"] - predict5th_feature_dict["num_chars"]) > 50):
        is_5th_grade_essay = False
    if(abs(feature_dict["num_words"] - predict5th_feature_dict["num_words"]) > 10):
        is_5th_grade_essay = False
    if(abs(feature_dict["num_sents"] - predict5th_feature_dict["num_sents"]) > 3):
        is_5th_grade_essay = False
    if(abs(feature_dict["avg_chars_per_word"] - predict5th_feature_dict["avg_chars_per_word"]) > 3):
        is_5th_grade_essay = False
    if(abs(feature_dict["avg_words_per_sent"] - predict5th_feature_dict["avg_words_per_sent"]) > 5):
        is_5th_grade_essay = False

    return is_5th_grade_essay



def predict6th(feature_dict):
    is_6th_grade_essay = True
    if(feature_dict["imperative"] == False):
        is_6th_grade_essay = False
    if(abs(feature_dict["num_chars"] - predict6th_feature_dict["num_chars"]) > 50):
        is_6th_grade_essay = False
    if(abs(feature_dict["num_words"] - predict6th_feature_dict["num_words"]) > 10):
        is_6th_grade_essay = False
    if(abs(feature_dict["num_sents"] - predict6th_feature_dict["num_sents"]) > 3):
        is_6th_grade_essay = False
    if(abs(feature_dict["avg_chars_per_word"] - predict6th_feature_dict["avg_chars_per_word"]) > 3):
        is_6th_grade_essay = False
    if(abs(feature_dict["avg_words_per_sent"] - predict6th_feature_dict["avg_words_per_sent"]) > 5):
        is_6th_grade_essay = False

    return is_6th_grade_essay



# classifier for overall complexity using the following features:
#   1. Number of words
#   2. Number of sentences
#   3. Average characters in words
#   4. Average words per sentence
#   5. Contains imperative sentence or question (indicative of call to action)
# takes a list of tuples of each doc in the form: [(essay, grade) ... ]
# returns classifier
def train_complexity_classifier(train):
    training_set = []
    # for each doc, collect features and create training set
    for concl, grade in train:
        if(grade >= 2 and concl != ""):
            # append tuple in the form (feature_dict, grade)
            feature_dict = make_feature_dict(concl)
            training_set.append((feature_dict, grade))

    return nltk.classify.NaiveBayesClassifier.train(training_set)



# makes a list of features to use for the complexity classifier
def make_feature_dict(concl):
    feature_dict = {}

    # get basic features for model
    num_chars = len(re.findall("[A-Za-z]", concl))
    num_words = len(nltk.word_tokenize(concl))
    num_sents = len(nltk.sent_tokenize(concl))

    if(num_words <= 1 or num_sents <= 1):
        return None

    # see if the conclusion contains a question or call to action
    feature_dict["imperative"] = has_imperative(concl) or "?" in concl

    # add basic features to feature_dict
    feature_dict["num_chars"] = num_chars 
    feature_dict["num_words"] = num_words
    feature_dict["num_sents"] = num_sents

    feature_dict["avg_chars_per_word"] = (num_chars / num_words)
    feature_dict["avg_words_per_sent"] = (num_words / num_sents)
  
    return feature_dict



# gets training data for complexity classifier
# in the form: [(essay, grade) ... ]
def extact_training_data(docs):
    training_data = []
    
    for doc in docs:
        concl = get_conclusion(doc["plaintext"])
        # get basic metrics of conclusion
        num_words = len(nltk.word_tokenize(concl))
        num_sents = len(nltk.sent_tokenize(concl))

        if(num_words <= 1 or num_sents <= 1):
            continue

        # get grade and round down
        grade = doc["grades"][1]["score"]["criteria"]["ending"]
        # appends (conclusion, grade of ending) to training_data
        training_data.append((concl, grade))

    return training_data



def has_imperative(concl):
    cleaned_concl = clean_document(concl)
    sents = tokenize_sentences(cleaned_concl)
    for sent in sents:
        if(is_imperative(nltk.pos_tag(sent))):
            return True 
    
    return False



def is_imperative(tagged_sent):
    # catches simple imperatives
    if(tagged_sent[0][1] == "VB" or tagged_sent[0][1] == "MD"):
        return True

    # catches imperative sentences starting with words like 'please', 'you'
    else:
        chunk = get_chunks(tagged_sent)
        # check if the first chunk of the sentence is a VB-Phrase
        if(type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase"):
            return True

    return False



# chunks the sentence into grammatical phrases based on its POS-tags
def get_chunks(tagged_sent):
    chunkgram = r"""VB-Phrase: {<DT><,>*<VB>}
                    VB-Phrase: {<RB><VB>}
                    VB-Phrase: {<UH><,>*<VB>}
                    VB-Phrase: {<UH><,><VBP>}
                    VB-Phrase: {<PRP><VB>}
                    VB-Phrase: {<NN.?>+<,>*<VB>}
                    Q-Tag: {<,><MD><RB>*<PRP><.>*}"""
    chunkparser = RegexpParser(chunkgram)
    return chunkparser.parse(tagged_sent)



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


    accuracy = get_accuracy(true_grades, predicted_grades)
    precession, recall, fscore, _ = precision_recall_fscore_support(true_grades, predicted_grades, average="micro")

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precession))
    print("Recall: " + str(recall))
    print("F-Score: " + str(fscore))

    cm = confusion_matrix(true_grades, predicted_grades)

    # change confusion matrix to data frame and output to csv
    cm_as_df=cm2df(cm,list(set(true_grades)))
    cm_as_df.to_csv("ending_confusion_matrix.csv")    


def get_accuracy(true_grades, predicted_grades):
    num_correct = 0
    for i in range(len(predicted_grades)):
        if(true_grades[i] == predicted_grades[i]):
            num_correct += 1

    return float(num_correct / len(predicted_grades))
 



def get_true_grades(documents):
    true_grades = []
    for doc in documents:
        grade = str(doc["grades"][1]["score"]["criteria"]["ending"])
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



    
