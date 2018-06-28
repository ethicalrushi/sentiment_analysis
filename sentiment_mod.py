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

import os.path
import sys
sys.path.append('/path/to/sentiment_analysis')

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)



import nltk

import pickle 

documents_f = open("documents.pickle","rb")
documents= pickle.load(documents_f)
documents_f.close()


word_features_f = open("word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
   
    words = set(document)
    features = {}
    for w in word_features:
  
        features[w] = (w in words)
    return features


featuresets =[(find_features(rev), category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]


classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


MNB_classifier_f = open("MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()


Bernoulli_classifier_f = open("Bernoulli_classifier.pickle","rb")
Bernoulli_classifier = pickle.load(Bernoulli_classifier_f)
Bernoulli_classifier_f.close()


LogisticRegression_classifier_f = open("LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()


SGD_Classifier_f = open("SGD_Classifier.pickle","rb")
SGD_Classifier = pickle.load(SGD_Classifier_f)
SGD_Classifier_f.close()


SVC_Classifier_f = open("SVC_classifier.pickle","rb")
SVC_classifier = pickle.load(SVC_Classifier_f)
SVC_Classifier_f.close()


LinearSVC_Classifier_f = open("LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(LinearSVC_Classifier_f)
LinearSVC_Classifier_f.close()


NuSVC_Classifier_f = open("NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(NuSVC_Classifier_f)
NuSVC_Classifier_f.close()
 

from nltk.classify import ClassifierI
from statistics import mode
 
class VoteClassifier(ClassifierI):
    
     

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


def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
 
 
 
 
 
 
 

