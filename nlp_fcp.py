# %%
'''
nlp fcp
train a classifier to predict the movie reviews into binary category
'''

# %%
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# for NLP
import spacy 
import tomotopy as tp
import gensim
from gensim.models import Phrases

# %%
# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# %%
# inspect the data
train.head()
train.shape
train.info()

train.groupby(by='helpfulness_cat').count()
print(sum(train['helpfulness_cat'])/len(train))

document_lengths = np.array(list(map(len, train['imdb_user_review'].str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))

fig, ax = plt.subplots(figsize=(15,6))

ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax)

# %%
train['imdb_user_review'].to_list()
# %%
# basic data cleaning
import re
def basic_clean(text):
    """
    Remove \\,\n,\t,... from text
    Remove whitespace from text
    change to lowercase
    """
    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\'', ' ').replace('"',' ')
    pattern = re.compile(r'\s+')
    Formatted_text = Formatted_text.lower()
    Formatted_text = Formatted_text.replace('\\u00b4',' ').replace('\\',' ')
    Without_whitespace = re.sub(pattern, ' ', Formatted_text)
    Formatted_text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return Formatted_text

# cleaning data
train['imdb_user_review'] = train['imdb_user_review'].apply(basic_clean)
train['imdb_user_review'][0]

#train.loc[:, "imdb_user_review"] = train.loc[:, "imdb_user_review"].str.replace(r'\\u00b4', "")
#train.loc[:, "imdb_user_review"] = train.loc[:, "imdb_user_review"].str.replace(r'\\u0085', "")

#%%
# sentiment analysis
# using roberta-large-mnli model to analyse sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

#%%
pipe = pipeline(model="roberta-large-mnli")
#new
#%%
# split the data into 5 data sets
train_1 = train[:int(len(train)*0.2)]
train_2 = train[int(len(train)*0.2):int(len(train)*0.4)]
train_3 = train[int(len(train)*0.4):int(len(train)*0.6)]
train_4 = train[int(len(train)*0.6):int(len(train)*0.8)]
train_5 = train[int(len(train)*0.8):]

#%%
CONTRADICTION = []
NEUTRAL = []
ENTAILMENT = []

for review in train_4['imdb_user_review']:
   try:
    temp = 0
    temp = pipe(review)[0][0]['score']
    CONTRADICTION.append(temp)

    temp = pipe(review)[0][1]['score']
    NEUTRAL.append(temp)

    temp = pipe(review)[0][2]['score']
    ENTAILMENT.append(temp)
   except:
     pass

#%%
# using nltk vader to carry out sentiment analysis
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%
# create a function to calculate the sentiment score
sent_analyzer = SentimentIntensityAnalyzer()
def sentiment_score(text):
    score = sent_analyzer.polarity_scores(text)
    score.pop('compound')
    return score


#%%
# calculate the sentiment score for each review and create neg, neu and pos columns
train_4['neg'] = train_2['imdb_user_review'].apply(sentiment_score).apply(lambda x: x['neg'])
train_4['neu'] = train_2['imdb_user_review'].apply(sentiment_score).apply(lambda x: x['neu'])
train_4['pos'] = train_2['imdb_user_review'].apply(sentiment_score).apply(lambda x: x['pos'])
train_4.head()

#%%
# combine contradiction, neutral and entailment into a data frame
train_4_sentiment = pd.DataFrame({'CONTRADICTION': CONTRADICTION, 'NEUTRAL': NEUTRAL, 'ENTAILMENT': ENTAILMENT})
train_4_sentiment.to_csv('train_4_sentiment.csv', index=False)

#%% split the data into validation and training set 
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)

#%%
# doc2vec
from nltk.tokenize import wordpunct_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#tokenize the quotes
tkn_quotes = [wordpunct_tokenize(quote.lower()) for quote in train['imdb_user_review'].to_list()] 
#tkn_quotes_train = [wordpunct_tokenize(quote.lower()) for quote in train_set['imdb_user_review'].to_list()] 
#tkn_quotes_val = [wordpunct_tokenize(quote.lower()) for quote in val_set['imdb_user_review'].to_list()] 
# tagged documents objects
#tgd_quotes_train = [TaggedDocument(d, [i]) for i, d in enumerate(tkn_quotes_train)]
tgd_quotes = [TaggedDocument(d, [i]) for i, d in enumerate(tkn_quotes)]
#Build Doc2Vec model
model = Doc2Vec(
        tgd_quotes, vector_size=100, window=2, min_count=1, workers=4, epochs=100
    )
# let's save the module for future applications
model.save("Doc2vec.model")
# and load it again for the sake of redundancy
model = Doc2Vec.load("Doc2vec.model")

#%%
# get the vector on each row
quote_vectors  = [model.infer_vector(tgd_quotes[i][0]) for i in range(len(tgd_quotes))]
#quote_vectors_train  = [model.infer_vector(quote) for quote in tkn_quotes_train]
#quote_vectors_val = [model.infer_vector(quote) for quote in tkn_quotes_val]

#%%
# calculate simarility
model.build_vocab(tgd_quotes)

#%%
# convert the vector to a pandas dataframe
df_vectors = pd.DataFrame(quote_vectors)
#df_vectors_val = pd.DataFrame(quote_vectors_val)
df_vectors.head()

#%%
# Dimension reduction with TSNE
from sklearn.manifold import TSNE
TSNE_model = TSNE(n_components=2, random_state=0)
TSNE_result = TSNE_model.fit_transform(df_vectors)
TSNE_result_df = pd.DataFrame(TSNE_result, columns=['PCA1', 'PCA2'])
TSNE_result_df['helpfulness_cat'] = train['helpfulness_cat'].to_list()
print(TSNE_result_df)

#%%
# visualise the reduced data based on the helpfulness class they belong to
import seaborn as sns
sns.lmplot(x="PCA1", y="PCA2", data=TSNE_result_df, fit_reg=False,legend=True,size=9,hue='helpfulness_cat',scatter_kws={'s':200,'alpha':0.3})
plt.title("TSNE Dimensionality Reduction")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")


#%% pytorch
from torchdata.datapipes.iter import FileLister, FileOpener
import os
def get_name(path_and_stream):
   return os.path.basename(path_and_stream[0]), path_and_stream[1]
datapipe1 = FileLister(".", "*.csv")
datapipe2 = FileOpener(datapipe1, mode="b")
datapipe3 = datapipe2.map(get_name)
csv_dict_parser_dp = datapipe3.parse_csv_as_dict()
list(csv_dict_parser_dp)
# %%
train['helpfulness_cat'].to_list()
# %%
