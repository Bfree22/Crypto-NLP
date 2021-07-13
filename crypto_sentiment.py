#!/usr/bin/env python
# coding: utf-8

# # Unit 12 - Tales from the Crypto
# 
# ---
# 

# ## 1. Sentiment Analysis
# 
# Use the [newsapi](https://newsapi.org/) to pull the latest news articles for Bitcoin and Ethereum and create a DataFrame of sentiment scores for each coin.
# 
# Use descriptive statistics to answer the following questions:
# 1. Which coin had the highest mean positive score?
# 2. Which coin had the highest negative score?
# 3. Which coin had the highest positive score?

# In[1]:


# Initial imports
import os
import pandas as pd
from dotenv import load_dotenv
import nltk as nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi.newsapi_client import NewsApiClient
analyzer = SentimentIntensityAnalyzer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


load_dotenv()
api_key = os.getenv("NEWS_API")
type(api_key)


# In[3]:


# Read your api key environment variable
api_key = os.getenv("NEWS_API")


# In[4]:


# Create a newsapi client
newsapi = NewsApiClient(api_key=api_key)
print(newsapi)


# In[5]:


# Fetch the Bitcoin news articles
btc_headlines = newsapi.get_everything(
    q="Bitcoin",
    language="en",
    
)
# Print total articles
print(f"Total articles about Bitcoin: {btc_headlines['totalResults']}")

# Show sample article
btc_article = btc_headlines["articles"][0]
btc_article


# In[6]:


# Fetch the Ethereum news articles
eth_headlines = newsapi.get_everything(
    q="Ethereum",
    language="en",
    
)
# Print total articles
print(f"Total articles about Ethereum: {eth_headlines['totalResults']}")

# Show sample article
eth_article =eth_headlines["articles"][0]
eth_article


# In[7]:


# Create the Bitcoin sentiment scores DataFrame
bitcoin_sentiment_score = []

for x in btc_headlines["articles"]:
    try:
        text = x["content"]
        
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neg = sentiment["neg"]
        neu = sentiment["neu"]
        
        bitcoin_sentiment_score.append({
            
            "Compound": compound,
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu,
            "Text": text
        })
        
    except AttributeError:
        pass
        
btc_df = pd.DataFrame(bitcoin_sentiment_score)

btc_df.head()


# In[8]:


# Create the Ethereum sentiment scores DataFrame
eth_sentiment_score = []

for x in eth_headlines["articles"]:
    try:
        text = x["content"]
        
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neg = sentiment["neg"]
        neu = sentiment["neu"]
        
        eth_sentiment_score.append({
            
            "Compound": compound,
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu,
            "Text": text
        })
        
    except AttributeError:
        pass
        
eth_df = pd.DataFrame(eth_sentiment_score)

eth_df.head()


# In[9]:


# Describe the Bitcoin Sentiment
btc_df.describe()


# In[10]:


# Describe the Ethereum Sentiment
eth_df.describe()


# ### Questions:
# 
# Q: Which coin had the highest mean positive score?
# 
# A: Bitcoin has the higher mean positive score (0.049) compared to Ethereum (0.040).
# 
# Q: Which coin had the highest compound score?
# 
# A: Ethereum has the highest compound score at 0.77 compared to the compound score of 0.69 from Bitcoin.
# 
# Q. Which coin had the highest positive score?
# 
# A: Ethereum has the highest positive score at 0.19 compared to the 0.16 maximum received by Bitcoin.

# ---

# ## 2. Natural Language Processing
# ---
# ###   Tokenizer
# 
# In this section, you will use NLTK and Python to tokenize the text for each coin. Be sure to:
# 1. Lowercase each word.
# 2. Remove Punctuation.
# 3. Remove Stopwords.

# In[11]:


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re
import nltk


# In[12]:


nltk.download('stopwords')


# In[13]:


# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Create a list of stopwords
sw = set(stopwords.words("English"))

# Expand the default stopwords list if neccesary 
sw_addon = {"pca","ec"}


# In[32]:


# Complete the tokenizer function
def tokenizer(text):
    sw= set(stopwords.words('english'))
    regex= re.compile("[^a-zA-Z ]")
    
    re_clean = regex.sub('', str(text))
    words= word_tokenize(re_clean)
    lem=[lemmatizer.lemmatize(word) for word in words]
    tokens= [word.lower() for word in lem if word.lower() not in sw ]
    
    return tokens


# In[33]:


# Create a new tokens column for bitcoin
btc_df["tokens"] = btc_df.Text.apply(tokenizer)
btc_df.head()


# In[34]:


# Create a new tokens column for bitcoin
eth_df["tokens"] = eth_df.Text.apply(tokenizer)
eth_df.head()


# ---

# In[ ]:





# ### NGrams and Frequency Analysis
# 
# In this section you will look at the ngrams and word frequency for each coin. 
# 
# 1. Use NLTK to produce the n-grams for N = 2. 
# 2. List the top 10 words for each coin. 

# In[35]:


from collections import Counter
from nltk import ngrams


# In[41]:


# Tokenize BTC article
btc_tokenized = tokenizer(btc_df.Text.str.cat())
btc_tokenized


# In[42]:


# Tokenize ETH article
eth_tokenized = tokenizer(eth_df.Text.str.cat())
eth_tokenized


# In[45]:


# Generate the Bitcoin N-grams where N=2
btc_ngrams = Counter(ngrams(btc_tokenized, n=2))
print(dict(btc_ngrams))


# In[46]:


# Generate the Ethereum N-grams where N=2
eth_ngrams = Counter(ngrams(eth_tokenized, n=2))
print(dict(eth_ngrams))


# In[47]:


# Function token_count generates the top 10 words for a given coin
def token_count(tokens, N=3):
    """Returns the top N tokens from the frequency count"""
    return Counter(tokens).most_common(N)


# In[48]:


# Use token_count to get the top 10 words for Bitcoin
btc_common= token_count(btc_tokenized, 10)
btc_common


# In[49]:


# Use token_count to get the top 10 words for Ethereum
eth_common= token_count(eth_tokenized, 10)
eth_common


# ---

# ### Word Clouds
# 
# In this section, you will generate word clouds for each coin to summarize the news for each coin

# In[50]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [20.0, 10.0]


# In[51]:


# Generate the Bitcoin word cloud
wordcloud = WordCloud(colormap="RdYlBu").generate(btc_df.Text.str.cat())
plt.imshow(wordcloud)
plt.axis("off")
fontdict = {"fontsize": 50, "fontweight": "bold"}
plt.title("Bitcoin Word Cloud", fontdict=fontdict)
plt.show()


# In[52]:


# Generate the Ethereum word cloud
wordcloud = WordCloud(colormap="RdYlBu").generate(eth_df.Text.str.cat())
plt.imshow(wordcloud)
plt.axis("off")
fontdict = {"fontsize": 50, "fontweight": "bold"}
plt.title("Ethereum Word Cloud", fontdict=fontdict)
plt.show()


# ---
# ## 3. Named Entity Recognition
# 
# In this section, you will build a named entity recognition model for both Bitcoin and Ethereum, then visualize the tags using SpaCy.

# In[53]:


import spacy
from spacy import displacy


# In[55]:


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')


# ---
# ### Bitcoin NER

# In[56]:


# Concatenate all of the Bitcoin text together
btc_text_combined = btc_df.Text.str.cat()
btc_text_combined


# In[57]:


# Run the NER processor on all of the text
btc_ner = nlp(btc_text_combined)
btc_ner
# Add a title to the document
btc_ner.user_data["title"] = "Bitcoin NER"


# In[58]:


# Render the visualization
displacy.render(btc_ner, style="ent")


# In[59]:


# List all Entities
for ent in btc_ner.ents:
    print(ent.text, ent.label_)


# ---

# ### Ethereum NER

# In[60]:


# Concatenate all of the Ethereum text together
eth_text_combined = eth_df.Text.str.cat()
eth_text_combined


# In[61]:


# Run the NER processor on all of the text
eth_ner = nlp(eth_text_combined)

# Add a title to the document
eth_ner.user_data["title"] = "Ethereum NER"


# In[62]:


# Render the visualization
displacy.render(eth_ner, style='ent')


# In[63]:


# List all Entities
for ent in eth_ner.ents:
    print(ent.text, ent.label_)


# ---
