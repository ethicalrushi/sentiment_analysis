#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:45:01 2018

@author: rushikesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 01:17:30 2018

@author: rushikesh
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle 

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))
        
random.shuffle(documents)
#print(documents[1])
save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
#all_words.most_common(15)

#print(all_words["stupid"])
#just taking the 3000 most common words for training
word_features = list(all_words.keys())[:3000]

save_word_features = open("word_features.pickle","wb")
pickle.dump(word_features, save_word_features )
save_word_features.close()




#Building the bag of words
#Document here as an argument is actually just review or words
def find_features(document):
    #words is a set of docs it gives single iteration of every word 
    #not the amount of words i.e unique
    words = set(document)
    features = {}
    for w in word_features:
        #setting a boolean w in words i.e true or false
        features[w] = (w in words)
    return features
#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))



#documents are of the format review, category
#category is to be predicted i.e pos or neg
featuresets =[(find_features(rev), category) for (rev,category) in documents]


#Splitting the data
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes' accuracy:",(nltk.classify.accuracy(classifier, testing_set)))

classifier.show_most_informative_features(15)

#accuracy 0.76

#Saving the trained classifier instead of training again


#wb is for writing in bytes
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



#Using the saved classsifier
#read in bytes

classifier_f = open("naivebayes.pickle","rb")
classifier1 = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes' accuracy:",(nltk.classify.accuracy(classifier1, testing_set)))
#0.76
classifier1.show_most_informative_features(15)




#Using Sklearn algorithms with nltk

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#GaussianNB needed a dense matrix so left it

#1]
MNB_classifier = SklearnClassifier(MultinomialNB())
#This converts the MNB_classifier into nltk's classifier so you can use it as
# a simple nltk classifier


MNB_classifier.train(training_set)
print("Multinomial Naive Bayes accuracy:",(nltk.classify.accuracy(MNB_classifier, testing_set)))
#0.79

#SklearnClassiifer has no attribute show_most_informative_features(15)
save_MNB_classifier = open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_MNB_classifier )
save_MNB_classifier.close()


#2]
Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print("Bernoulli Naive Bayes accuracy:",(nltk.classify.accuracy(Bernoulli_classifier, testing_set)))
#0.76

save_Bernoulli_classifier = open("Bernoulli_classifier.pickle","wb")
pickle.dump(Bernoulli_classifier, save_Bernoulli_classifier )
save_Bernoulli_classifier.close()


#Using some other algos from sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC , NuSVC

#3]
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy:",(nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))
#0.85

save_LogisticRegression_classifier = open("LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier )
save_LogisticRegression_classifier.close()




#4]
SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGDClassifieraccuracy:",(nltk.classify.accuracy(SGD_Classifier, testing_set)))
#0.84

save_SGD_Classifier = open("SGD_Classifier.pickle","wb")
pickle.dump(SGD_Classifier, save_SGD_Classifier )
save_SGD_Classifier.close()



#5]
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy:",(nltk.classify.accuracy(SVC_classifier, testing_set)))
#0.75

save_SVC_classifier = open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier, save_SVC_classifier )
save_SVC_classifier.close()


#6]
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy:",(nltk.classify.accuracy(LinearSVC_classifier, testing_set)))
#0.87

save_LinearSVC_classifier = open("LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier )
save_LinearSVC_classifier.close()



#7]
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy:",(nltk.classify.accuracy(NuSVC_classifier, testing_set)))
 #0.87
 
save_NuSVC_classifier = open("NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_NuSVC_classifier )
save_NuSVC_classifier.close()
 
 
 #Custom classifier using all of above classifiers except svc as it has low accuracy
from nltk.classify import ClassifierI
from statistics import mode
 
class VoteClassifier(ClassifierI):
    
     
#*classifiers is list passed as args
    def __init__(self, *classifiers):
         self._classifiers = classifiers
         
    def classify(self, features):
         votes = []
         for c in self._classifiers:
             v = c.classify(features)
             votes.append(v)
         return mode(votes)
     
    def confidence(self, features):
        
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
         
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
    
voted_classifier = VoteClassifier(classifier,MNB_classifier,Bernoulli_classifier,LogisticRegression_classifier,SGD_Classifier,LinearSVC_classifier,NuSVC_classifier )

print("VotedClassifier accuracy:",(nltk.classify.accuracy(voted_classifier, testing_set)))
#0.85

print("Classification:", voted_classifier.classify(testing_set[5][0]),"confidence:", voted_classifier.confidence(testing_set[5][0]))     
print("Classification:", voted_classifier.classify(testing_set[1][0]),"confidence:", voted_classifier.confidence(testing_set[1][0]))       
         
     
#Check the accuracy of voted_classifier on pos and neg review by testing non shuffled data
#first 2000 neg data next 2000 pos       
#To get acc on pos and neg and get the bias
 
 

 
 
 

