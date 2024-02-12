import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import (word_tokenize,
                           sent_tokenize)
from nltk.corpus import stopwords
from textblob import TextBlob
import seaborn as sns

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

data = pd.read_csv("merged.csv")
data.columns

# ---Data Cleaning---

# remove punctuations
data['human_written_processed'] = data['human_written_text'].map(lambda x: re.sub('[,\.!?]', '', x))
data['human_written_processed'].head()

data['ai_processed'] = data['ai_generated_text'].map(lambda x: re.sub('[,\.!?]', '', x))
data['ai_processed'].head()

# sentence and words tokenization
real_tokens = data['human_written_processed'].map(lambda x: word_tokenize(x))
real_sentences = data['human_written_text'].map(lambda x: sent_tokenize(x))

ai_tokens = data['ai_processed'].map(lambda x: word_tokenize(x))
ai_sentences = data['ai_generated_text'].map(lambda x: sent_tokenize(x))

# remove stop words
stop_words = stopwords.words('english')
real_fil = real_tokens.map(lambda x: [word for word in x if not word.lower() in stop_words])
ai_fil = ai_tokens.map(lambda x: [word for word in x if not word.lower() in stop_words])

# ---Exploratory Analysis - wordcloud---

# functions
def remove_verbs(tokens):
    tagged_tokens = tokens.map(lambda x: nltk.pos_tag(x))
    non_verbs = tagged_tokens.map(lambda x: [word for word, tag in x if not tag.startswith('VB')])
    return non_verbs

def generate_wordcloud(non_verbs):
    string = non_verbs.map(lambda x: ' '.join(x))
    long_string = ','.join(list(string.values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    return wordcloud

def display_wordcloud(wordcloud):
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation = "bilinear")
    plt.axis('off')

# human-written
real_non_verbs = remove_verbs(real_fil)
real_wordcloud = generate_wordcloud(real_non_verbs)
display_wordcloud(real_wordcloud)
plt.savefig(f'real_wordcloud.png', dpi = 300)
plt.show()

# AI-generated
ai_non_verbs = remove_verbs(ai_fil)
ai_wordcloud = generate_wordcloud(ai_non_verbs)
display_wordcloud(ai_wordcloud)
plt.savefig(f'ai_wordcloud.png', dpi = 300)
plt.show()

# ---Language Feature---

# word length
def word_len(tokens):
    word_len_each_row = tokens.map(lambda x: [len(word) for word in x])
    return word_len_each_row.sum()

real_word_len_list = word_len(real_tokens) # list for word lengths
real_avg_word_len = np.mean(real_word_len_list)

ai_word_len_list = word_len(ai_tokens) # list for word lengths
ai_avg_word_len = np.mean(ai_word_len_list)

# sentence length
def sentence_len(sentences):
    sentence_lengths = sentences.map(lambda x: [len(word_tokenize(sentence)) for sentence in x])
    return sentence_lengths.sum()

real_sentence_len_list = sentence_len(real_sentences) # list for sentence lengths
real_avg_sentence_len = np.mean(real_sentence_len_list)

ai_sentence_len_list = sentence_len(ai_sentences) # list for sentence lengths
ai_avg_sentence_len = np.mean(ai_sentence_len_list)

# article length
def article_len(sentences):
    article_len = [len(x) for x in sentences]
    return article_len

real_article_len_list = article_len(real_sentences) # list for article lengths
real_avg_article_len = np.mean(real_article_len_list)

ai_article_len_list = article_len(ai_sentences) # list for article lengths
ai_avg_article_len = np.mean(ai_article_len_list)

# ---Sentimental Analysis---

# semantics
def cal_sentence_polarity(sentences):
    return sentences.map(lambda x: [TextBlob(sentence).sentiment.polarity for sentence in x])

def cal_article_polarity(sentence_pol):
    sentence_pol_long_list = sentence_pol.sum()
    article_pol = [np.mean(row) for row in sentence_pol_long_list]
    return article_pol

real_sentence_pol = cal_sentence_polarity(real_sentences)
real_article_pol = cal_article_polarity(real_sentence_pol)

ai_sentence_pol = cal_sentence_polarity(ai_sentences)
ai_article_pol = cal_article_polarity(ai_sentence_pol)

# (histograms)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].hist(real_article_pol, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(ai_article_pol, color='green', alpha=0.7, label='Group 2')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('AI-generated News')

for ax in axs:
    ax.set_xlabel('Polarity Score')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# sentiment volatility
real_sv = [np.std(sentences) for sentences in real_sentence_pol]
ai_sv = [np.std(sentences) for sentences in ai_sentence_pol]

# (histograms)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].hist(real_sv, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(ai_sv, color='green', alpha=0.7, label='Group 2')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('AI-generated News')

for ax in axs:
    ax.set_xlabel('Sentiment Volatility')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# no. of affect words
def cal_no_of_affect_words(tokens):
    word_pol = tokens.map(lambda x: [TextBlob(word).polarity for word in x])
    affect_word = word_pol.map(lambda x: [1 if val != 0 else 0 for val in x])
    no_affect_word = [np.sum(row) for row in affect_word]
    return no_affect_word

real_no_affect_word = cal_no_of_affect_words(real_fil)
real_avg_no_affect_word = np.mean(real_no_affect_word)

ai_no_affect_word = cal_no_of_affect_words(ai_fil)
ai_avg_no_affect_word = np.mean(ai_no_affect_word)

# (histograms)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].hist(real_no_affect_word, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(ai_no_affect_word, color='green', alpha=0.7, label='Group 2')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('AI-generated News')

for ax in axs:
    ax.set_xlabel('Number of Affection Words')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# subjectivity
def cal_sentence_subjectivity(sentences):
    return sentences.map(lambda x: [TextBlob(sentence).sentiment.subjectivity for sentence in x])

def cal_article_subjectivity(sentence_pol):
    sentence_pol_long_list = sentence_pol.sum()
    article_pol = [np.mean(row) for row in sentence_pol_long_list]
    return article_pol

real_sentence_sub = cal_sentence_subjectivity(real_sentences)
real_article_sub = cal_article_subjectivity(real_sentence_sub)

ai_sentence_sub = cal_sentence_subjectivity(ai_sentences)
ai_article_sub = cal_article_subjectivity(ai_sentence_sub)

# (histograms)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].hist(real_article_sub, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(ai_article_sub, color='green', alpha=0.7, label='Group 2')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('AI-generated News')

for ax in axs:
    ax.set_xlabel('Subjectivity Score')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# ---T-tests---

# ---Topic Distribution---

