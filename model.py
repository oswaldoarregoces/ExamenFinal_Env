import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Obtenir data

df = pd.read_csv('pointure.data')

# pre-tratement des données 



label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

# DEFINIR LES FEATURES

X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]



#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# apprendre  le modele

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_naive_bayes1 = gnb.predict(X_train)


accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Accuracy du modele Naive Bayes predit: " + str(accuracy))


recall_score = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_score))

f1_score = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))


# evaluation du test

y_naive_bayes2 = gnb.predict(X_test)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))

recall_score = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes predit: " + str(recall_score))

f1_score = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))



# prediction 

d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d) 
dfToPredict

yPredict = gnb.predict(dfToPredict)
print('La classe predite est : ', yPredict)

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("recall score:  {0:2.1f} \n".format(recall_score))
        outfile.write("f1 score {0:2.1f}\n".format(f1_score))