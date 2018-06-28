#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 02:46:06 2018

@author: rushikesh
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import json

import sentiment_mod as s



ckey = ''
csecret = ''
atoken = ''
asecret = ''



class Listener(StreamListener):
    
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data['text']
        sentiment_value, confidence = s.sentiment(tweet)
        
        print(tweet, sentiment_value,confidence)
        return True
    
    def on_error(self, status):
        print(status)
    
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=['car'])




