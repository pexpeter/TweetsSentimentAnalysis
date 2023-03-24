import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class TweetSentimentPredictor():
    """Reads a tweet returns a sentiment label.
        
        Args:
            tweet (list): A list of a tweet to check its sentiment.
            
        Returns:
            str : A sentence with the tweet and its sentiment.
    """
    def __init__(self):
        # reads in data 
        self.df = pd.read_csv("C:/Users/Admin 21/Downloads/tweets.csv",
                    encoding='ISO-8859-1', 
                    header = None, 
                    names= ["target", "tweet_id", "date", "flag", "user", "tweet"]
                    )
        # drop excess columns
        self.df = self.df.drop(columns = ["tweet_id", "date", "flag", "user"])
        # data preprocessing with `0` for negative sentiment and `1` for positive sentiment.
        self.df["target"] = self.df["target"].map({0:0,4:1})
        
        def text_processing(text):
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'http\S+', '',text)
            text = re.sub(r'#\w+', '',text )
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.lower()
            return text.strip()
        # column change to clean text
        self.df["tweet"] = self.df["tweet"].apply(text_processing)
        
        # Model building 
        X = self.df["tweet"]
        y= self.df["target"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature extraction
        self.vector = CountVectorizer(stop_words='english')
        self.X_train_vector = self.vector.fit_transform(self.X_train)
        
        #Model
        self.naves = MultinomialNB()
        self.naves.fit(self.X_train_vector, self.y_train)
        
    def prediction(self, tweet):
        #Model prediction
        tweet_tok = self.vector.transform(tweet)
        pred = self.naves.predict(tweet_tok)
        
        results =[]
        for i in range(len(tweet)):
            if pred[i] == 1:
                results.append(f"{tweet[i]} ' tweet has a positive sentiment label.")
            else:
                results.append(f"{tweet[i]} ' tweet has a negative sentiment label.")
                
        return "\n".join(results)
