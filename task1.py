from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import pickle

"""
Trains Mulinomial Naive Bayes model using bag of words features
and saves them to a file to avoid retraining every time
"""
def train(train_data):
    #train_data = [line.strip().split("\t") for line in open("PS3_training_data.txt")]
    train_X = [line[1] for line in train_data]
    train_y = [line[4] for line in train_data]
    count_vect = CountVectorizer()
    train_X_counts = count_vect.fit_transform(train_X)
    nb_clf = MultinomialNB()
    nb_clf.fit(train_X_counts, train_y)
    pickle.dump(nb_clf, open("task1_saved_model", "wb"))
    pickle.dump(count_vect, open("task1_saved_features", "wb"))
    
    #tfidf_transformer = TfidfTransformer()
    #train_X_tfidf = tfidf_transformer.fit_transform(train_X_counts)
    #validate_X_counts = tfidf_transformer.transform(validate_X_counts)

"""
Extracts the features from test data and gets the prediction
for it using the previously generated Naive Bayes model.
"""
def extract_test_features(test_data):
    global test_X_counts, test_y, prediction
    test_X = [line[1] for line in test_data]
    test_X_counts = count_vect.transform(test_X)
    test_y = [line[4] for line in test_data]
    prediction = nb_clf.predict(test_X_counts)

"""
Function to write the predictions to file replacing test labels
with newly predicted labels
"""
#def predict_and_write():
#    output_file = open("test_output.txt", "w")
#    for i in range(len(test_data)):
#        output_file.writelines(test_data[i][0] + '\t' + test_data[i][1] + '\t' + test_data[i][2] +
#              '\t' + test_data[i][3] + '\t' + prediction[i] + "\n")

"""
Starting point for task 1 training.
"""
def train_model(train_data):
    train(train_data)

"""
Starting point for task 1 prediction. Returns the predicted results in an array
"""
def get_predictions(test_data):
    global nb_clf, count_vect
    nb_clf = pickle.load(open("task1_saved_model", "rb"))
    count_vect = pickle.load(open("task1_saved_features", "rb"))
        
    extract_test_features(test_data)
    print("Task 1 Model accuracy on test data: ", nb_clf.score(test_X_counts, test_y))
    return prediction
    
#def task_1(test_file):
#    global nb_clf, count_vect
#    saved_model_file = Path("task1_saved_model")
#    saved_features_file = Path("task1_saved_features")
#    
#    if saved_model_file.is_file() and saved_features_file.is_file():
#        nb_clf = pickle.load(open("task1_saved_model", "rb"))
#        count_vect = pickle.load(open("task1_saved_features", "rb"))
#        
#    else:
#        train()
#        
#    extract_test_features(test_file)
#    print("Model accuracy on test data: %f", nb_clf.score(test_X_counts, test_y))
#    return prediction
#    #predict_and_write()