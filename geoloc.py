# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:41:47 2019
@author: annonymous
"""
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import  chi2
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import hstack
le = preprocessing.LabelEncoder()
from imblearn.over_sampling import SMOTE

def clean_text(text):
    text_nonum = re.sub(r'\d+', '', text)
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace

def chk_stop(text):
    idx = text.index('\'')
    text = text[:idx]
    return text.lower()

def split_token(text):
    returnlist = []
    splitted = text.split('_')
    for x in splitted:
        returnlist.append("".join([char.lower() for char in x if char not in string.punctuation])) 

    return returnlist 

def chk_repeated(string):
    cnt=0
    for i in range(len(string)-1):
        if string[i] == string[i+1]:
            cnt+=1
    if cnt>3:
        return True
    else:
        return False
    
def match_fword(string):
    f_match = ['fuck', 'fucked', 'fucking', 'fucker', 'fuxk', 'f**k', 'f*ck', 'fuuuck', 'facefuck', 'f——k',
           'ass', 'asses', 'dumbass', 'bastard', 'bastards', 'bitch', 'bitches', 'damn', 'damned',
           'darn', 'goddamn', 'hell', 'hellish', 'shit', 'shits', 'shitted', 'shitting', 'shat', 'shite',
           'piss', 'cunt', 'cock', 'sucker', 'cocksucker', 'motherfucker', 'tits', 'christ', 'crap', 'wtf', 'wtff']
    
    if string in f_match:
        return True
    else:
        return False
    
def match_emo(string):
    emo_match = [':)', '(:', ':(', ':):)',':(:(',':((',':(((', ':):(',  ':/',  ':-(', ';-(', ';-)', ':-)', ':-*', ':-/', ':~/',            
            ':D',';D','xD', '=D','XD',  '=/', ':]', ';]', ';[', ':[',  '=]', '=[', '-_-', '-__-',  '-___-','=)', '=('  ]
    if string in emo_match:
        return True
    else:
        return False
    

def proc_texts(output_text, input_text):
    with open(output_text, 'w', encoding="latin-1") as out, open(input_text, encoding="latin-1") as tweets:
          
        tweets = tweets.readlines()  

        stop = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()       
        collist = ['tweet-id', 'user', 'text', 'emo_cnt', 'textlen', 'f_cnt', 'repeated_cnt', 'upper_ratio', 'class']                                                   
        fmt = ''
        for col in collist:
            fmt+= col + '\t'
        fmt = fmt.strip() + '\n'
        out.write(fmt)
        
        glotaggedlist = []
        group = None
        groupcnt=0
        for line in tweets:
            line = line.strip()  
            tweet_id = line.split(',')[0]
            first_user_st_idx = line.index('USER_')           
            taggedlist, f_list = [],[]
            emo_cnt, f_cnt, upper_cnt, upper_ratio, repeated_cnt, user_cnt = 0,0,0,0,0,0
            #emo_list =[]            
            locadict = dict.fromkeys(['ny', 'gg', 'cl'], 0) 

            label_idx = -(line[::-1].index(','))                           
            label = line[label_idx:]
            text = line[first_user_st_idx:label_idx-1]            
            textlen = "{:.4f}".format(float(len(text)/100))
            
            while "USER_" in text:   
                st_idx = text.index('USER_')
                end_idx = st_idx+13
                user = text[st_idx:end_idx]                
                taggedlist.append(user)
                text = text[end_idx:]

                former = text[:st_idx-1] 
                if 'RT' in former:
                    former=former[:former.index('RT')]                    
                latter = text[end_idx+1:]
                text = former + ' ' + latter  

            tt = TweetTokenizer(strip_handles=True,  reduce_len=True)                    
            tokens = tt.tokenize(text)
            removelist = []
            tokenlist= []            
            pronoun = ''            
            
            for token in tokens:
                if len(token)>1 and token.isupper() and token.isalpha():
                    upper_cnt +=1                    
                token = token.lower()                    
                        
                if match_fword(token) or match_fword(lemmatizer.lemmatize(token)):
                    f_cnt +=1
                    f_list.append(lemmatizer.lemmatize(token))
                    tokenlist.append(lemmatizer.lemmatize(token))
                    
                if match_emo(token):
                    emo_cnt +=1
                    tokenlist.append(token)
                    continue
                
                if '_' in token:
                    splitted = split_token(token)
                    tokenlist.extend(splitted)
                    continue
  
                if '\'' in token:    
                    pronoun = chk_stop(token)                    
                    
                if '#' in token:
                    tokenlist.append(token)
                    continue

                if chk_repeated(token):
                    repeated_cnt+=1

                cleaned = clean_text(token)                
                if '#' not in token and cleaned not in stop and cleaned not in string.punctuation and pronoun not in stop  \
                    and len(token)> 1 and not bool(re.search(r'^[0-9]*$', token)) :
                    #lemma = lemmatizer.lemmatize(cleaned)
                    tokenlist.append(cleaned)    
                    pronoun = ''
                else:
                    removelist.append(token)
            
            if len(taggedlist)==0:
                taggedlist = glotaggedlist
                #taggedlist.append('nottagged')
            else:                
                glotaggedlist = taggedlist

            tokens = [e for e in tokens if e not in removelist]

            if upper_cnt>0:
                upper_ratio = "{:.4f}".format(float(upper_cnt/len(tokens)))
                1==1
            if len(tokenlist) == 0:
                tokenlist.append('nomessage')

            fmt = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(tweet_id, ' '.join(map(str, taggedlist)), 
                   ' '.join(map(str, tokenlist)),  emo_cnt, textlen, f_cnt, repeated_cnt, upper_ratio, label)
            
            out.write(fmt)


def proc_users(ny_path, cl_path, gg_path,input_text):
    with open(ny_path, 'w', encoding="latin-1") as out_ny, open(cl_path, 'w', encoding="latin-1") as out_cl,  open(gg_path, 'w', encoding="latin-1") as out_gg, open(input_text, encoding="latin-1") as tweets:
        tweets = tweets.readlines()    
        nyusers=[]
        clusers=[]
        ggusers=[]
        for line in tweets:
            line = line.strip()  
            first_user_st_idx = line.index('USER_')
            label_idx = -(line[::-1].index(','))  
            label = line[label_idx:]
            text = line[first_user_st_idx:label_idx-1]            

            while "USER_" in text:   
                st_idx = text.index('USER_')
                end_idx = st_idx+13
                user = text[st_idx:end_idx]
                text = text[end_idx:]
                if label == 'NewYork':                
                    nyusers.append(user)
                elif label == 'California':
                    clusers.append(user)
                else:
                    ggusers.append(user)

        out_ny.write("{}\n".format(','.join(map(str, set(nyusers)))) ) 
        out_cl.write("{}\n".format(','.join(map(str, set(clusers)))) ) 
        out_gg.write("{}\n".format(','.join(map(str, set(ggusers)))) )     
        

def get_TFIDF(X_train, X_test, MAX_NB_WORDS=1000):    
    param = { "sublinear_tf":True, "analyzer":'word', "min_df":5, "max_df": 0.5,
             "max_features":MAX_NB_WORDS, "stop_words":'english', 'norm':'l2' }    
    vectorizer_x = TfidfVectorizer(**param)
    X_train = vectorizer_x.fit_transform(X_train)
    X_test = vectorizer_x.transform(X_test)
    #print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)

def preprocess():    
    #proc_users('userny.csv', 'usercl.csv', 'usergg.csv', 'tweets/users_tweets.txt')    
    
    ##### development mode => feature engineering
    proc_texts('tfidf.csv', 'tweets/train_tweets.txt')
    proc_texts('tfidf_test.csv', 'tweets/dev_tweets.txt')
    
    ##### test mode => feature engineering
    #proc_texts('userny.csv','usergg.csv','usercl.csv', 'tfidf.csv', 'tweets/traindev_tweets.txt')
    #proc_texts('userny.csv','usergg.csv','usercl.csv', 'tfidf_test.csv', 'tweets/test_tweets.txt')  

def getfeature(col, df):
    param = { "sublinear_tf":True, "analyzer":'word', "min_df":10, "max_df": 0.5, 'norm':'l2' } #sublinear for lagarithm scale
    tfidf = TfidfVectorizer(**param)
    features = tfidf.fit_transform(df[col])
    
    return tfidf, features

## Get TOP N vocabulary that has the highest chi square scores in each class 
def return_topval(list_, col, label_dict, labels, df):
  N = 100
  tfidf, features = getfeature(col, df)
  for label, num in sorted(label_dict.items()):
      features_chi2 = chi2(features, labels == num)
      indices = np.argsort(features_chi2[0])       
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      
      ## display highly related words by class
      print("# '{}':".format(label))
      print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
      # Return N vocabulary to feed TF-IDF vectorizer
      list_.extend(unigrams[-N:])
      
  return list_

def model():      
    ##### development mode
    df =pd.read_csv('x_tfidf.csv', encoding ='latin1', sep = '\t')
    df_test =pd.read_csv('x_tfidf_test.csv', encoding ='latin1', sep = '\t')  
        
    ##### test mode
    #df =pd.read_csv('tfidf.csv', encoding ='latin1', sep = '\t')
    #df_test =pd.read_csv('tfidf_test.csv', encoding ='latin1', sep = '\t')
    
    train_y = df['class']
    test_y = df_test['class']
    df['id'] = df['class'].factorize()[0]
    label_df = df[['class', 'id']].drop_duplicates().sort_values('id')
    label_dict = dict(label_df.values)
    labels=df['id']
    df['text'] = df['text'].fillna("None")
    df_test['text'] = df_test['text'].fillna("None")   
    
    rf = RandomForestClassifier(bootstrap=True,
                  min_samples_leaf=3,
                  n_estimators=1000, 
                  min_samples_split=4,
                  max_features='sqrt',
                  max_depth= 9,
                  max_leaf_nodes=None,
                  random_state=1,
                  n_jobs=1,
                  class_weight='balanced', #for balancing
                  criterion='gini')

    models = [           
        LinearSVC(random_state=0, class_weight='balanced'),         
        MultinomialNB(),
        rf
    ]
    
    #### Vectorising userid and text features
    vocalist = []      
    vocalist = return_topval(vocalist, 'text', label_dict, labels, df)
    tfidf_vec = TfidfVectorizer(analyzer='word', sublinear_tf=True, norm='l2', min_df=2, max_df= 0.5, vocabulary=set(vocalist)  )
    tf = tfidf_vec.fit_transform(df['text'])
    tf_test = tfidf_vec.transform(df_test['text'])
    
    user_vec = TfidfVectorizer( analyzer='word', sublinear_tf=True, norm='l2', min_df=2, max_df= 0.5)
    user = user_vec.fit_transform(df['user'])
    user_test = user_vec.transform(df_test['user'])
    meta_list = ['emo_cnt', 'textlen', 'f_cnt', 'repeated_cnt', 'upper_ratio'] #'ny', 'gg', 'cl', 
    
    ### Stacking all features together
    train_X = hstack((tf, user ),  format='csr')
    test_X = hstack((tf_test, user_test) , format='csr')
    for col in meta_list:
        train_X = hstack((train_X,  df[col].values.reshape(df.shape[0], 1))  )
        test_X = hstack((test_X,  df_test[col].values.reshape(df_test.shape[0], 1))  )    

    train_X, train_y = SMOTE(random_state=42).fit_resample(train_X, train_y)  ## sampling technique: SMOTE, SMOTEEN, etc.
    
    return models, train_X, test_X, train_y, test_y
    

def evaluate(models, train_X, test_X, train_y, test_y ):
    estimators=[]    
    for classifier in models:
        model_name = classifier.__class__.__name__
        model = classifier.fit(train_X, train_y)   
        
        ''' ....  for ensemble voting classifier  
        #clf = (model_name, model) 
        #estimators.append(clf)  
        #model = VotingClassifier(estimators, voting='hard')    
        #model.fit(train_X, train_y)  
        '''
        predictions = model.predict(test_X)   
        accuracies = accuracy_score(test_y ,predictions)
        print(model_name , ':')
        print(accuracies)
        print(classification_report(test_y, predictions))
        
        ### IN THE TEST MODE
        #df_test["class"] = predictions
        #df_test = df_test.drop(['user', 'text', 'ny', 'gg', 'cl', 'emo_cnt', 'textlen', 'f_cnt', 'repeated_cnt', 'upper_ratio'], axis = 1)
        #df_test.to_csv("prediction.csv")

if __name__ == "__main__":
    # feature engineering
    preprocess()
    
    # modelling
    models, train_X, test_X, train_y, test_y = model() 
    
    # evaluation metrics
    evaluate(models, train_X, test_X, train_y, test_y )
