from tai_util import get_docs
from tai_ending import get_conclusion
import re
import nltk
import numpy as np




def main():
    docs = get_docs("tai-documents-v3.json")
    sums = {}

    for doc in docs:
        concl = get_conclusion(doc["plaintext"])
        grade = doc["grades"][1]["score"]["criteria"]["ending"]

        if(concl == ""):
            continue
        
        if(grade in sums.keys()):
            sums[grade] = add_to_sums(grade, concl, sums)
        else:
            concl_metrics = get_concl_metrics(concl)
            if(concl_metrics is not None):
                concl_metrics.append(0)
                sums[grade] = concl_metrics
        

    metrics = avg_metrics_matrix(sums)

    predict_5th = get_prediction(metrics, 5)
    predict_6th = get_prediction(metrics, 6)

    print(predict_5th)
    print(predict_6th)



def get_prediction(metrics, grade_to_predict):
    grades = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    feat_set1 = metrics[:,0]
    feat_set2 = metrics[:,1]
    feat_set3 = metrics[:,2]
    feat_set4 = metrics[:,3]
    feat_set5 = metrics[:,4]

    m1, b1 = np.polyfit(grades, feat_set1, 1)
    m2, b2 = np.polyfit(grades, feat_set2, 1)
    m3, b3 = np.polyfit(grades, feat_set3, 1)
    m4, b4 = np.polyfit(grades, feat_set4, 1)
    m5, b5 = np.polyfit(grades, feat_set5, 1)

    return [m1 * grade_to_predict + b1,
            m2 * grade_to_predict + b2,
            m3 * grade_to_predict + b3,
            m4 * grade_to_predict + b4,
            m5 * grade_to_predict + b5]
    
    


def avg_metrics_matrix(sums):
    metrics = []
    for key in sorted(sums.keys()):
        metrics_sums = sums[key]
        avg_metrics = []
        for i in range(len(metrics_sums) - 1):
            total = metrics_sums[-1]
            avg_metrics.append(metrics_sums[i] / float(total))

        metrics.append(avg_metrics) 

    return np.matrix(metrics)
    


def add_to_sums(grade, concl, sums):
    res = []
    concl_metrics_old = sums[grade]
    concl_metrics_new = get_concl_metrics(concl)
    for i in range(len(concl_metrics_new)):
        res.append(concl_metrics_new[i] + concl_metrics_old[i])

    res.append(concl_metrics_old[-1] + 1)

    return res 
        


def get_concl_metrics(concl):
    num_chars = len(re.findall("[A-Za-z]", concl))
    num_words = len(nltk.word_tokenize(concl))
    num_sents = len(nltk.sent_tokenize(concl))

    avg_chars_per_word = (num_chars / num_words)
    avg_words_per_sent = (num_words / num_sents)

    return [num_chars, num_words, num_sents, avg_chars_per_word, avg_words_per_sent]



if(__name__ == "__main__"):
    main()





