from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from nltk import word_tokenize
import numpy as np
import textblob, pickle, nltk, re

"""
function to recode POS/NEG/NEU to NON/NEU for first classifier
input: string (either POSITIVE, NEGATIVE, or NEUTRAL)
output: string (NON or NEUTRAL)
"""
def recode_neutral(s):
    if s == "NEUTRAL":
        return s
    return "NOT"

"""
function to determine if there are quotes in the sentence
input: string
output: 1 if there is a quote, 0 otherwise
"""
def tag_quote(s):
    if re.search(".*\".*\".*", s):
        return 1
    return 0

"""
function to determine if there is an instance of "not" in the sentence
input: string
return: 1 if there is an occurrence, 0 ow
"""
def tag_not(s):
    if re.search(".*not.*", s) or re.search(".*n't.*", s):
        return 1
    return 0

"""
extract the y features for each observation
input: data array
output: vector of sentiments
"""
def get_y_t2(data):
    return np.array([line[2] for line in data])

"""
if the sentence contains "not" or "n't", flip the polarity
input: row of data
output: row of data
"""
def flip_polarity(el):
    if el[3] == 1:
        return -el[0]
    return el[0]

"""
extract the features for training
input: data file
output: array of features, the sentiment(maybe flipped), log(abs(polarity)),
number of adjectives, and whether or not there are quotes
"""
def get_features_t2(data):
    sents = [line[1] for line in data]
    tbfeats = [textblob.TextBlob(sent).sentiment[0] for sent in sents]
    invpol = [np.log(abs(k + 0.001)) for k in tbfeats]
    quo = [(tag_quote(s), tag_not(s)) for s in sents]
    feats = np.column_stack((np.array(tbfeats), np.array(invpol)))
    feats = np.column_stack((feats, np.array(quo)))
    d_pos = [nltk.pos_tag(word_tokenize(sent.lower())) for sent in sents]
    only_pos = [[el[1] for el in pos] for pos in d_pos]
    num_pos = [len(el) for el in only_pos]
    num_jj = [sum([1 for el in pos if el == "JJ"]) for pos in only_pos]
    prop_jj = np.array(num_jj) / np.array(num_pos)
    feats = np.column_stack((feats, prop_jj))

    # polarity_negate = np.array([flip_polarity(el) for el in feats])
    # feats = np.delete(feats, 0 , 1)
    # feats = np.delete(feats, 2, 1)
    # new_feats = np.column_stack((feats, polarity_negate))

    # return new_feats
    return feats
"""
train both models and save them
input: data file
output: none
"""
def train_model_t2(train_data):
    model1 = DecisionTreeClassifier()
    y_1 = np.array([recode_neutral(s) for s in get_y_t2(train_data)])
    vect = TfidfVectorizer(use_idf = False, norm = "l1")
    X_1 = vect.fit_transform([l[1] for l in train_data])
    pickle.dump(vect, open("vect_t2.sav", "wb"))

    model1.fit(X_1, y_1)
    pickle.dump(model1, open("task2_model1.sav", "wb"))

    y_2 = get_y_t2([l for l in train_data if l[2] != "NEUTRAL"])
    X_2 = get_features_t2([l for l in train_data if l[2] != "NEUTRAL"])

    costs = [1, .5, .1, .05, .01, .005, .001]
    full_scores = []
    skf = StratifiedKFold(n_splits = 5)
    for cost in costs:
        scores = []
        for train_ind, test_ind in skf.split(X_2, y_2):
            model = linear_model.LogisticRegression(penalty = "l2", C = cost)
            model.fit(X_2[train_ind], y_2[train_ind])

            scores.append(model.score(X_2[test_ind], y_2[test_ind]))
        full_scores.append(np.mean(scores))
        print("Cost: %0.4f\tAccuracy:%0.4f (+/- %0.4f)" % (cost, np.mean(scores), np.std(scores) * 2))

    chosen_cost = costs[np.argmax(full_scores)]

    model2 = linear_model.LogisticRegression(penalty = "l2", C = chosen_cost)
    model2.fit(X_2, y_2)
    pickle.dump(model2, open("task2_model2.sav", "wb"))

"""
run trained models on new data
input: data file
output: test predictions
"""
def get_predictions_t2(test_data):
    vect = pickle.load(open("vect_t2.sav", "rb"))
    model1 = pickle.load(open("task2_model1.sav", "rb"))
    model2 = pickle.load(open("task2_model2.sav", "rb"))

    X_1 = vect.transform([l[1] for l in test_data])
    y_1 = model1.predict(X_1)
    actual_y = [recode_neutral(s) for s in get_y_t2(test_data)]
    # print("Model 1 accuracy: %0.4f" % model1.score(X_1, actual_y))

    non_ind = []
    neu_ind = []
    # gather indices of non-neutral predictions
    for i in range(0, len(y_1)):
        if y_1[i] == "NEUTRAL":
            neu_ind.append(i)
        else:
            non_ind.append(i)

    X_2 = get_features_t2(np.array(test_data)[non_ind])
    y_2 = model2.predict(X_2)
    # print("Model 2 accuracy: %0.4f" % model2.score(X_2, get_y_t2(test_data)[non_ind]))

    y_f = []
    j = 0
    for i in range(0, len(y_1)):
        if y_1[i] == "NEUTRAL":
            y_f.append(y_1[i])
        else:
            y_f.append(y_2[j])
            j += 1

    acc = []
    y_test = get_y_t2(test_data)
    for i in range(0, len(y_f)):
        if y_f[i] == y_test[i]:
            acc.append(1)
        else:
            acc.append(0)
    print("Task 2 accuracy: %f" % np.mean(acc))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_f))

    return y_f
