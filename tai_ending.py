import nltk
import re
from nltk import RegexpParser
from nltk.tree import Tree
from tai_util import get_docs, clean_document, tokenize_sentences
import math







def grade_ending(doc, classifier):
    # get the conlcusion of the essay
    concl = get_conclusion(doc)

    # get basic metrics of conclusion
    num_words = len(nltk.word_tokenize(concl))

    # classify general complexity of essay into 1, 2, 3, 4, 5, 6
    complexity = classify_complexity(concl, classifier)

    # classify conclusions using these metrics
    if(num_words == 1):
        return 0
    else:
        return complexity
    




def get_conclusion(doc):
    last_par = doc.split("\n")[-1]
    return last_par.strip()



def classify_complexity(concl, classifier):
    feature_dict = make_feature_dict(concl)
    if(feature_dict is None):
        return 0
    else:
        return classifier.classify(feature_dict)



# classifier for overall complexity using the following features:
#   1. Number of words
#   2. Number of sentences
#   3. Average characters in words
#   4. Average words per sentence
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
        grade = math.floor(grade)
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




    
