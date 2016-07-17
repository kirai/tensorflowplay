# Word vectorization model: 
# * tfidf with sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# * word2vec: google
# import numpy as np
# B = np.random.randint(0,255,(sizeOfArray,sizeOfArray)
# (A == B).mean()

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,  delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)


    print('The first review is:')
    print(train["review"][0])

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    train_data_features = train_data_features.toarray()

    print "Training the random forest (this may take a while)..."

    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["sentiment"] )
