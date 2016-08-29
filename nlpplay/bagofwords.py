# Word vectorization model: 
# * tfidf with sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# * word2vec: google
# import numpy as np
# B = np.random.randint(0,255,(sizeOfArray,sizeOfArray)
# (A == B).mean()
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Explicacion way: https://github.com/pangolulu/sentiment-analysis

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk

#nltk.download()

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,  delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)

    print('The first review is:')
    print(train["review"][0])
    input("Press Enter to continue...")

    clean_train_reviews = []

    print("Cleaning and parsing the training set movie reviews...\n")
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    print("Creating the bag of words...\n")

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    
    # Random forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["sentiment"])

    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print("Predicting test labels...\n")
    result = forest.predict(test_data_features)
        
    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_Words_model.csv")

