
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import pickle 


def train(train):
    '''
    This function trains the data on a logistic regression model and the model 
    is pickled afterwards.
    Starting point for task 1 training.
    '''
    
    train_X=[]
    train_y=[]
    
    for line in train:
        if line[3]=="NONE":
            pass
        else:
            train_X.append(line[1])
            train_y.append(line[3].strip())
    datasentences = train_X
    vectorizer=CountVectorizer()
    train_features=vectorizer.fit_transform(datasentences)
    pickle.dump(vectorizer,open("task3_CountV.sav","wb"))
    maxBest=0
    best=[1,0.5,0.75,0.8,0.25]
    for i in range(len(best)-1):
        logreg = linear_model.LogisticRegression()
        logreg = linear_model.LogisticRegression(C=best[i],penalty="l1")
        logreg.fit((train_features),(train_y))
        scor=logreg.score(train_features,train_y)
        if (scor >= maxBest):
                maxBestIndex=i
                maxBest=scor
    logreg = linear_model.LogisticRegression()
    logreg = linear_model.LogisticRegression(C=best[maxBestIndex],penalty="l1")
    logreg.fit((train_features),(train_y))
    pickle.dump(logreg,open("task3_model.sav","wb"))



def getPrediction(test):
    '''
    This function extracts the test features of the given test data
    and then tests the saved model them and returns back the predicted data. 
    Starting point for task 1 prediction. Returns the predicted results in an array
    '''
    global validate, validate_features, prediction    
    logreg=pickle.load(open("task3_model.sav","rb"))
    vectorizer=pickle.load(open("task3_CountV.sav","rb"))
    validate = test
    validate_X=[]
    validate_y=[]
    
    for line in validate:
        if line[3]=="NONE":
            pass
        else:
            validate_X.append(line[1])
            validate_y.append(line[3])

    datasentences1 =  validate_X
    validate_features=vectorizer.transform(datasentences1)
    prediction = logreg.predict(validate_features)
    answer=[]
    j=0
    for line in (validate):
        if line[3]=="NONE":
            answer.append("NONE")
        else:
            answer.append(prediction[j].upper())
            j=j+1
    print("Task 3 Accuracy on test data: ", logreg.score(validate_features, validate_y))
    return answer

