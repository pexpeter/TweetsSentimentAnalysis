import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def pred_sentiment(tweet):
    """Reads a tweet returns a sentiment label.

    Args:
        tweet (list): A list of a tweet to check its sentiment.
        
    Returns:
        str : A sentence with the tweet and its sentiment.
    """
    # reads in data 
    df = pd.read_csv("C:/Users/Admin 21/Downloads/tweets.csv",
                encoding='ISO-8859-1', 
                header = None, 
                names= ["target", "tweet_id", "date", "flag", "user", "tweet"]
                )
    # drop excess columns
    df = df.drop(columns = ["tweet_id", "date", "flag", "user"])
    # data preprocessing with `0` for negative sentiment and `1` for positive sentiment.
    df["target"] = df["target"].map({0:0,4:1})
    
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
    df["tweet"] = df["tweet"].apply(text_processing)
    
    # Model building 
    X = df["tweet"]
    y= df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature extraction
    vector = CountVectorizer(stop_words='english')
    X_train_vector = vector.fit_transform(X_train)
    X_test_vector = vector.transform(X_test)
    
    #Model
    naves = MultinomialNB()
    naves.fit(X_train_vector, y_train)
    
    #Model prediction
    tweet_tok = vector.transform(tweet)
    pred = naves.predict(tweet_tok)
    
    for i in range(len(tweet)):
        print(f"'{tweet[i]} ' tweet has a {'positive' if pred[i]==1 else 'negative'} sentiment label.")  
