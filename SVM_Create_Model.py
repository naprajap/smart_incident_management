#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/

import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

from sklearn import svm


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[2]:


#Import data
#file_path = 
file_name = 'incident_30032019.csv'
data_file = file_path+file_name
data = pd.read_csv(data_file, delimiter = ",",parse_dates=True,encoding='ISO-8859-1')


# In[3]:


# Convert date into Python format
data['opened_at'] = pd.to_datetime(data.opened_at)


# In[4]:


data.head()


# In[5]:


#Clean text
def review_to_wordlist( review, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    #4.1 Remove words where lenght of word in 1
    words = [w for w in words if len(w) != 1]
    
    
    #Lemmatization: Lemmatization, on the other hand, is an organized & step by step procedure of 
    #obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) 
    #and morphological analysis (word structure and grammar relations).
    c=[]
    lem = WordNetLemmatizer()
    for word in words:
        c.append(lem.lemmatize(word, "v"))
       
    words = c   

    #Stemming:  Stemming is a rudimentary rule-based process of stripping the suffixes 
    #(“ing”, “ly”, “es”, “s” etc) from a word.
    
    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))
    
    words = b

    # 5. Return a list of words
    return(words)


# In[6]:


train, test = train_test_split(data, test_size = 0.3, random_state =123)


# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


clean_train_reviews = []
for review in train['short_description']:
    clean_train_reviews.append( " ".join(review_to_wordlist(review))) 
    
clean_test_reviews = []
for review in test['short_description']:
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))


# In[10]:


# Create feature vectors 
vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(clean_train_reviews)
# Apply model on test data 
test_vectors = vectorizer.transform(clean_test_reviews)


# In[11]:


clean_train_reviews


# In[12]:


model = svm.SVC(kernel='linear') 
model.fit(train_vectors, train['assignment_group']) 
prediction = model.predict(test_vectors)


# In[13]:


print (classification_report(test['assignment_group'], prediction))


# In[14]:


print (accuracy_score(test['assignment_group'], prediction))


# In[15]:


print (confusion_matrix(test['assignment_group'], prediction))


# In[16]:



import pickle
#file_path = 
file_name = 'svm_servicenow_30032019.pickle'
file_svm = file_path+file_name

save_classifier = open(file_svm,"wb")
pickle.dump(model, save_classifier)
save_classifier.close()


# In[18]:


file_tdif = file_path+'tdif_file30032019.pickle'
save_tdif = open(file_tdif,"wb")
pickle.dump(vectorizer, save_tdif)
save_tdif.close()


# In[19]:


open_svm = open(file_svm, "rb")
model_svm = pickle.load(open_svm)
open_svm.close()


# In[21]:


pred1 = model_svm.predict(test_vectors)


# In[22]:


print (classification_report(test['assignment_group'], pred1))


# In[38]:


text = ['Remove device XXXXXXXXXX from MPAN XXXXXXXXXXXX']
#for word in clean_text : 
 #   review_to_wordlist(text)
    
clean_test_reviews = []
for review in text:
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))   

clean_text = clean_test_reviews


# In[39]:


clean_text


# In[40]:


open_tdif = open(file_tdif, "rb")
vector_tdif = pickle.load(open_tdif)
open_tdif.close()


# In[41]:


test_vectors1 = vector_tdif.transform(clean_text)


# In[42]:


test_vectors1


# In[43]:


test_vectors1.shape


# In[44]:


pred1 = model_svm.predict(test_vectors1)


# In[45]:


pred1


# In[ ]:




