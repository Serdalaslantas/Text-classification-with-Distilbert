# %% [markdown]
# # ML Aproach To A Binary Classificaion of A Text Dataset

# %% [markdown]
# #### In this notebook I explained how to train machine learning models on a classification task with a text dataset. I used two models as Logistic Regression and Multinomial Naive Bayes with two different vectorizers as countVectorizer and tfidfVectorizer. Whie CountVectorizer use the architecture of CBOW and skip-gram methods TfidfVectorizer use Tfidf method of vectorization. Let's dive into the model construction!

# %% [markdown]
# # 1. Data Analysis

# %% [markdown]
# ## Dependencies

# %%
# for feature engineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#for model-building
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for word embedding
import gensim
from gensim.models import Word2Vec
#for word cloud
from wordcloud import WordCloud

# %% [markdown]
# ## Data Preprocessing and Getting to know the dataset

# %%
# Load the datasets
df_train = pd.read_csv('/workspaces/codespaces-jupyter/train.csv')
df_test = pd.read_csv('/workspaces/codespaces-jupyter/test.csv')
print(df_train.head())
print(df_test.head())



# %%
#check all the columns: 4 columns: 3 columns numeric and 1 column object type.
df_train.columns 

# %%
# Let's check the info and missing values: The column 'cyber_label' has 1173 missing values out of 1300.
df_train.info() 
df_train.isnull().sum() 

# %%
 # Check only the values with cyber_label is 1.0 and compare them with environmental_issue 
df_train[df_train.loc[:,'cyber_label']==1]

# %%
# Let's see the most frequent words in the 'content' grouped by cyber_label=1 by wordcloud.
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(''.join(i for i in df_train[df_train['cyber_label']==1]['content']))
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# %%
# Check the stats of the numerical features.
df_train.describe() 

# %%
#check the distibution of the values of the column 'environmental_issue'
df_train.environmental_issue.value_counts() 

# %%
 #visualize the dist. of the values.
df_train['environmental_issue'].value_counts().plot(kind = 'bar')

# %% [markdown]
# ## Preprocessing Pipeline

# %%
#nltk text preprocessing
# create preprocess_text function
def preprocess_text(text):
    r = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenize the text
    tokens = word_tokenize(r.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text
# apply the function df_train and df_test
df_train['content'] = df_train['content'].apply(preprocess_text)
df_test['content'] = df_test['content'].apply(preprocess_text)
print(df_train.head())

# %% [markdown]
# ### Word Count

# %%
# Word count in each class groupby the target : Class positive has slightly more words in average
df_train['word_count'] = df_train['content'].apply(lambda x: len(str(x).split()))
print(df_train[df_train['environmental_issue']==1]['word_count'].mean()) #risk
print(df_train[df_train['environmental_issue']==0]['word_count'].mean()) #Non-risk

# %%
# Plotting the distribution of word count
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
train_words=df_train[df_train['environmental_issue']==1]['word_count']
ax1.hist(train_words,color='red')
ax1.set_title('Risk')
train_words=df_train[df_train['environmental_issue']==0]['word_count']
ax2.hist(train_words,color='green')
ax2.set_title('No-risk')
fig.suptitle('Words per message')
plt.show()

# %% [markdown]
# ### WordCloud for the frequent words groupby environmental_issue

# %%
# Let's see the most frequent words in the 'content' by wordcloud in Risky messages.
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(''.join(i for i in df_train[df_train['environmental_issue']==1]['content']))
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# %%
# Let's see the most frequent words in the 'content' by wordcloud in Non-Risky messages.
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(''.join(i for i in df_train[df_train['environmental_issue']==0]['content']))
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# %% [markdown]
# # 2. Machine Learning Models

# %%
# Create Feature and Label sets
X = df_train['content']
y = df_train['environmental_issue']
# train test split (80% train - 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

# %% [markdown]
# ## Ferature Extraction

# %% [markdown]
# ### CountVectorizer vs Tfidf Vectorizer

# %%
# Train Bag of Words model
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_train_cv.shape

# %%
# Tfidf Vectorizer
tfidf = TfidfVectorizer() 
X_train_tfidf = tfidf.fit_transform(X_train)
X_train_tfidf.shape

# %% [markdown]
# ## Logistic Regression

# %%
lr_tfidf = make_pipeline(TfidfVectorizer(), LogisticRegression(solver = 'saga', C=10, penalty = 'l1'))
lr_count = make_pipeline(CountVectorizer(), LogisticRegression(solver = 'saga', C=10, penalty = 'l1'))

lr_tfidf.fit(X_train, y_train)
lr_count.fit(X_train, y_train)

(Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer()),
                 ('multinomialnb', LogisticRegression(solver = 'saga', C=10, penalty = 'l1'))]),
 Pipeline(steps=[('countvectorizer', CountVectorizer()),
                 ('multinomialnb', LogisticRegression(solver = 'saga', C=10, penalty = 'l1'))]))

y_pred_tfidf = lr_tfidf.predict(X_test)
y_pred_count = lr_count.predict(X_test)

f1_tfidf = f1_score(y_test, y_pred_tfidf, average='weighted')
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print('Logistic Regression with TF-IDF:')
print('-' * 40)
print(f'f1: {f1_tfidf:.4f}')
print(f'accuracy: {accuracy_tfidf:.4f}')

f1_cv = f1_score(y_test, y_pred_count, average='weighted')
accuracy_cv = accuracy_score(y_test, y_pred_count)
print('Logistic Regression with count:')
print('-' * 40)
print(f'f1: {f1_cv:.4f}')
print(f'accuracy: {accuracy_cv:.4f}')

#generate the labels for the test dataset 
pred_lr_cv = lr_count.predict(df_test['content'])
pred_lr_tfidf = lr_tfidf.predict(df_test['content'])
print(pred_lr_cv)
print(pred_lr_tfidf)

df_test['lr_count'] = pred_lr_cv
df_test['lr_tfidf'] = pred_lr_tfidf
df_test.head()

# %%
# confusion matrix for cv

from sklearn import metrics
df_cv = pd.DataFrame(metrics.confusion_matrix(y_test,y_pred_count), index=['pos','neg'], columns=['pos','neg'])
df_cv

# %%
# confusion matrix for tfidf

from sklearn import metrics
df_tfidf = pd.DataFrame(metrics.confusion_matrix(y_test,y_pred_tfidf), index=['pos','neg'], columns=['pos','neg'])
df_tfidf

# %%
# CountVectorizer has higher accuracy than Tfidf Vectorizer 
print(classification_report(y_test, y_pred_count)) 
print(classification_report(y_test, y_pred_tfidf))



# %% [markdown]
# ### Grid Search for the best parameters

# %%
# Grid search cross validation for the Best Parameters
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
lr = LogisticRegression(solver = 'saga', C=10, penalty = 'l1')
lr_cv = lr.fit(X_train_cv, y_train)
parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
              {'penalty':['none', 'elasticnet', 'l1', 'l2']},
              {'C':[0.001, 0.01, 0.1, 1, 10, 100]}]



grid_search = GridSearchCV(estimator = lr,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose=0)


grid_search.fit(X_train_cv, y_train)

# %% [markdown]
# LogisticRegression(C=10, penalty='l1', solver='saga') is the best parameter

# %%
# Grid search cross validation for the Best Parameters
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
grid_search.fit(X_train_tfidf, y_train)

# %% [markdown]
# ## Naive Bayes Model

# %%
model_tfidf = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_count = make_pipeline(CountVectorizer(), MultinomialNB())

model_tfidf.fit(X_train, y_train), \
model_count.fit(X_train, y_train)

(Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer()),
                 ('multinomialnb', MultinomialNB())]),
 Pipeline(steps=[('countvectorizer', CountVectorizer()),
                 ('multinomialnb', MultinomialNB())]))

y_pred_tfidf = model_tfidf.predict(X_test)
y_pred_count = model_count.predict(X_test)

f1_tfidf = f1_score(y_test, y_pred_tfidf, average='weighted')
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print('Multinomial Naive Bayes with TF-IDF:')
print('-' * 40)
print(f'f1: {f1_tfidf:.4f}')
print(f'accuracy: {accuracy_tfidf:.4f}')

f1_cv = f1_score(y_test, y_pred_count, average='weighted')
accuracy_cv = accuracy_score(y_test, y_pred_count)
print('Multinomial Naive Bayes with Count:')
print('-' * 40)
print(f'f1: {f1_cv:.4f}')
print(f'accuracy: {accuracy_cv:.4f}')

# %%
# CountVectorizer has higher accuracy than Tfidf Vectorizer 
print(classification_report(y_test, y_pred_count)) 
print(classification_report(y_test, y_pred_tfidf))

# %%
# confusion matrix for cv

from sklearn import metrics
df_cv = pd.DataFrame(metrics.confusion_matrix(y_test,y_pred_count), index=['pos','neg'], columns=['pos','neg'])
df_cv

# %%
# confusion matrix for tfidf

from sklearn import metrics
df_tfidf = pd.DataFrame(metrics.confusion_matrix(y_test,y_pred_tfidf), index=['pos','neg'], columns=['pos','neg'])
df_tfidf

# %% [markdown]
# ## Conclusion

# %%
df_test.to_csv('test_labelled.csv')

# %% [markdown]
# ### Among the 4 variations of 2 models Decision Tree with CountVectorizer gives the highest accuracy as almost %90. But all the models performed close to each other. 


