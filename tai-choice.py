import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import sys
stop = stopwords.words('english')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def word_freq_dist(document):
    """Returns a word count frequency distribution"""
    words = nltk.tokenize.word_tokenize(document)
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    return fdist

def extract_subjects(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(10)
                       if nltk.pos_tag([w])[0][1] in NOUNS]
    # print("Most frequent nouns: ", most_freq_nouns)

    return most_freq_nouns

def find_bad_vocab(document):
    bad_words = ['thing', 'it', 'stuff', 'anything', 'lot']
    really_bad_words = ['gonna']
    tokens = nltk.word_tokenize(document)
    
    deductions = 0
    # Only add deductions if they use a 'bad word' too much
    for word in bad_words:
        count = tokens.count(word)
        if count > len(tokens) / 50:
            deductions += 0.5
    # Always add deductions if they use a 'really bad word'
    for word in really_bad_words:
        if tokens.count(word) > 0:
            deductions += 0.5

    return deductions

def evaluate_vocab(subjects, document):
    vocab_score = 0
    tokens = nltk.word_tokenize(document)
    tokens = list(dict.fromkeys(tokens))
    freq_dist = word_freq_dist(document)
    
    for subject in subjects:
        synsets = wn.synsets(subject, pos=wn.NOUN)
        
        for synset in synsets:
            for token in tokens:
                token_syns = wn.synsets(token, pos=wn.NOUN)
                for token_syn in token_syns:
                    sim = wn.path_similarity(synset,token_syn)
                    if sim > 0.5:
                        vocab_score += freq_dist.freq(token)
                        break

            for token in tokens:
                token_syns = wn.synsets(token, pos=wn.NOUN)
                for token_syn in token_syns:
                    for hypo in synset.hyponyms():
                        sim = wn.path_similarity(hypo,token_syn)
                        if sim > 0.5:
                            vocab_score += freq_dist.freq(token)
                            break

            for token in tokens:
                token_syns = wn.synsets(token, pos=wn.NOUN)
                for token_syn in token_syns:
                    for hyper in synset.hypernyms():
                        sim = wn.path_similarity(hyper,token_syn)
                        if sim > 0.5:
                            vocab_score += freq_dist.freq(token)
                            break
    return vocab_score

def evaluate_tone(document):
    teaching_phrases = ['that means', 'that really means', 'explain',
        'because', 'since', 'therefore', 'according to', 'up to', 'in summary']
    score = 0
    # use regular expression to find phrases
    for phrase in teaching_phrases:
        if re.search(phrase, document):
            score += 0.5
    return score

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: tai-choice.py input')

    filename = sys.argv[1]
    file = open(filename, mode='r')
    document = file.read()
    file.close()
    
    document = clean_document(document)
    
    subjects = extract_subjects(document)

    vocab_score = evaluate_vocab(subjects, document) * 0.8
    # print('Vocab score: ', vocab_score)

    tone_score = evaluate_tone(document)
    # print('Tone score: ', tone_score)

    deductions = find_bad_vocab(document)
    # print('Deductions: ', deductions)

    final_score = vocab_score + tone_score - deductions
    # print('Final Score: ', final_score)

    if final_score > 2.5:
        print('Score is 6')
    elif final_score > 2:
        print('Score is 5')
    elif final_score > 1.5:
        print('Score is 4')
    elif final_score > 1:
        print('Score is 3')
    elif final_score > 0.5:
        print('Score is 2')
    else:
        print('Score is 1')
        
