## [1] Overview
This project is to detect geolocation of twitter users based on TF-IDF.
Multinomial Naive Bayes and Random forest were considered in this experiment.


## [2] Preprocessing
TweetTokenizer was used as this tokenizer was better fitted for this tweet text in that 
1) it reduces repeated characters to a certain length 

    i.e. haaaaaaaa => haaa 

2) it can contain userids, hastags and emoticons that might be excluded by many other tokenizers.

After tokenization, stopwords, special characters, and punctuation were removed and lemmatized word was stored.


## [3] Feature engineering
1. TF-IDF score is calculated for all records.
2. Sort TF-IDF score by each class. i.e Select top 20 features from Georgiax as the position of the blend word characters, 
3. combine the vocabulary that obtained top scores from each class and remove the duplication.
4. Feed the combined vocabulary again to the TF-IDF vectorizer



ex)
Top 5 words that have highest chi square scores for each class:

**California: mor, gw, hella, hahaha, haha**

**Georgia: famusextape, willies, atlanta, thatisall, atl**

**NewYork: lml, lmaooo, lmaoo, inhighschool, haha**


## [4] Meta features
Along with tf-idf scores based on texts, meta features were used, which are taggedusers, # of emoticon used, text length, # of swear words used, # of repeated same characters, ratio of all upper case letters, respectively.


## [5] Sampling for imbalance
An attempt to tackle imbalance between classes using SMOTE, Synthetic Minority Oversampling Technique.
However, it did not lead to improvement in evaluation scores or the better distribution of the score. 
