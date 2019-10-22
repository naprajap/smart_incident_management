#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load required modules
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
import pysnow
import requests
import pandas as pd
import json
from pandas.io.json import json_normalize
import pickle

from sklearn.metrics.pairwise import cosine_similarity

#from matplotlib import cm
#%matplotlib inline
#plt.style.use('ggplot')


# In[ ]:


#file_path = ''
file_name_svm = 'svm_servicenow_30032019.pickle'
file_name_tdif = 'tdif_file30032019.pickle'

file_svm = file_path+file_name_svm
file_tdif = file_path+ file_name_tdif

open_svm = open(file_svm, "rb")
model_svm = pickle.load(open_svm)
open_svm.close()

open_tdif = open(file_tdif, "rb")
vector_tdif = pickle.load(open_tdif)
open_tdif.close()


# In[ ]:


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


# In[ ]:


# Selected all incident which are created without assignment group
# Fetch System ID of the incidents
# Create new client with SN request parameters
# Set Connection 


#https://pysnow.readthedocs.io/en/latest/full_examples/create.html
# Connect to Service Now instance
sn = pysnow.Client(instance='dev87486',
                   user='admin',
                   password='Centrica@123')

# Select records Where assigment group is blank
r = sn.query(table='incident', query={'assignment_group': ''})

all_records=[]

# Fetch one record and filter out everything but 'number' and 'sys_id'
res = r.get_multiple(fields=['number', 'short_description','assignment_group','sys_id','Opened'], order_by=['-Opened'])
# Print out the result

# Create Data Fream 
inc_data = pd.DataFrame(columns = ['number','short_description','assignment_group','sys_id','Opened'])

for records in res:
    inc_data = inc_data.append(pd.DataFrame([records]))

inc_data['clean_short_des'] = ''


# In[ ]:


clean_test_reviews = []
for review in inc_data['short_description'] :
    #print(review)
    clean_test_reviews.append( " ".join(review_to_wordlist(review))) 
    
inc_data['clean_short_des'] = clean_test_reviews


# In[ ]:


#inc_data[['number','short_description','clean_short_des']]


# In[ ]:


#Select All Assignment groups from Service now
#Use Table sys_user_group
grp_r = sn.query(table='sys_user_group',query = {'active':'true'})

# Fetch one record 
grp_res = grp_r.get_multiple(fields=['sys_id', 'description', 'name'])

#Create Data Freame
grp_df = pd.DataFrame(columns = ['sys_id','description', 'name'])

for records in grp_res:
    if records['name'] != '':
        grp_df = grp_df.append(pd.DataFrame([records]))


# In[ ]:


# Rename Column Name 
grp_df.rename(columns={"name": "assignment_group", 'sys_id':'group_sys_id'}, inplace=True)
#grp_df


# In[ ]:


#grp_df


# In[ ]:


#text = ['Remove device 18L2172442 from MPAN 2000015733921']
#text = ['<RMJF1000A> : pgb1-p-vapp135-prod#SC_J3B_DCARETNTDX.J3B_FTP_RPT_PI_DCARETNTDX_04']
#text = ['gpf guid']

#clean_test_reviews = []
#for review in text:
#    clean_test_reviews.append( " ".join(review_to_wordlist(review)))   

clean_text = clean_test_reviews
# Convert to TDIF Vector 
test_vectors = vector_tdif.transform(clean_text)
#Predict
pred = model_svm.predict(test_vectors)
#pred
#print(model_svm.score(test_vectors,pred))


# In[ ]:


#inc_data[['number','short_description','assignment_group']]


# In[ ]:


inc_data['assignment_group'] = pred


# In[ ]:


inc_data


# In[ ]:


#Join Incident and assignment group
inc_final_df = pd.merge(inc_data,grp_df, on='assignment_group')

#inc_final_df['short_description']


# In[ ]:


#inc_final_df[['short_description']['assignment_group']]
#inc_final_df
#inc_data


# In[ ]:


# find similar incident
# Load Incident Data for finding Similarity - at present we are loading data
# Ideally live - will fetch latest closed incident from service now
#Import data
#file_path = ''
file_name = 'incident_30032019.csv'
data_file = file_path+file_name
data = pd.read_csv(data_file, delimiter = ",",parse_dates=True,encoding='ISO-8859-1')

# Convert date into 
data['opened_at'] = pd.to_datetime(data.opened_at)
data['closed_at'] = pd.to_datetime(data.closed_at)


# In[ ]:


def fetch_last_3_incident(cleat_text_update,incident_group):

    
    data_closed_inc = data[ (data['assignment_group'] == incident_group) & (data['state'] == 'Closed')]
        
    clean_closed_inc = []
    for review in data_closed_inc['short_description']:
        clean_closed_inc.append( " ".join(review_to_wordlist(review))) 
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_closed_inc)
    tfidf_new_text = tfidf_vectorizer.transform([cleat_text_update])
    
    similarity_score = cosine_similarity(tfidf_new_text, tfidf_matrix)
    
    score = pd.DataFrame(similarity_score[0],columns =["score"])
    #print(data_closed_inc.head())
    #print(score)
    #new_inc = pd.concat([data_closed_inc,score], axis = 1,ignore_index=False))
    data_closed_inc['score']  = np.round (similarity_score[0] * 100,  decimals=0)
    
    #print(data_closed_inc.head(3))
    
    #new_inc['score'] = new_inc['score']*100
    data_closed_inc.sort_values(by = ['score','closed_at'], inplace=True, ascending=False)
    
    comment = data_closed_inc[['number','closed_at','assigned_to','short_description','score']].head(3)
    worknotes = comment.to_csv(sep='\t',index=False)

    return(worknotes)
    
    


# In[ ]:


#work_note = fetch_last_3_incident('<RMJF1000A> : pgb1-p-vapp135-prod#SC_J3B_DCARETNTDX.J3B_FTP_RPT_PI_DCARETNTDX_04','CORE SAP L2 DEVICE MANAGEMENT')
#work_note


# In[ ]:


#aed84364db41330096919fd2ca9619c5
# Define a resource, here we'll use the incident table API
incident = sn.resource(api_path='/table/incident')

print('Updation Started...')

for index, row in inc_final_df.iterrows():
    print ('Updating .. Incident Work Group')
    print (row['assignment_group'], '-', row['number'])
    update = { 'assignment_group': row['group_sys_id'] }
    print ('Incidet No '+row['number']+' updating.....')
    # Update 'short_description'
    updated_inc_record = incident.update(query={'number': row['number']}, payload=update)
    
    print ('Updating .. Incident Work Items with last closed incident')
    
    work_note = ''
    work_note = fetch_last_3_incident(row['clean_short_des'],row['assignment_group'])
    update = {'work_notes':work_note}
    updated_record = incident.update(query={'sys_id': row['sys_id']}, payload=update)
    
    print ('Incident No '+row['number']+' updated.')
    
#Print out the updated record
#print(updated_record)
print('Updated all records...')


# In[ ]:




