import tweepy
from textblob import TextBlob

consumer_key = 'FtdxPkwxNycQsoff0YyBT6mhr'
consumer_secret = 'Y7xcl4yvOzHCbUG1lQo6KcuX9CocVFwaO8pEp2QCI70Yh1jxuS'

access_token = '808233723481100288-DreiznXUXvFUJ25CUkbe3pcmixsa4ia'
access_token_secret = 'rD9qXxwt51rdUqgX8FbSCHgGJCMAt3THevNxYpKJd6j9G'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Modi')

for tweet in public_tweets:
	clean_tweet=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text).split())
	print(clean_tweet)
	analysis = TextBlob(clean_tweet)
	print(analysis.sentiment)



 