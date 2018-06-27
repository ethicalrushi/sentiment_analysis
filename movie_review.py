#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 01:17:30 2018

@author: rushikesh
"""

import nltk
import random
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))
        
random.shuffle(documents)
#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
#all_words.most_common(15)

#print(all_words["stupid"])
#just taking the 3000 most common words for training
word_features = list(all_words.keys())[:3000]

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

print("Naive Bayes' accuracy:",(nltk.classify.accuracy(classifier, testing_set)))

classifier.show_most_informative_features(15)

#accuracy 0.78

#Saving the trained classifier instead of training again
import pickle

#wb is for writing in bytes
save_classifier = open("NaiveBayes.pickle","wb")
pickle.dumps(classifier, save_classifier)
save_classifier.close()



#Using the saved classsifier
#read in bytes

classsifier_f = open("NaiveBayes.pickle","rb")
classifier1 = pickle.load(classifier_f)
classifer_f.close()

print("Naive Bayes' accuracy:",(nltk.classify.accuracy(classifier1, testing_set)))

classifier1.show_most_informative_features(15)






















