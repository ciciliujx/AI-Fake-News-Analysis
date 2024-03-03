import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import (word_tokenize,
                           sent_tokenize)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from textblob import TextBlob
import gensim
from better_profanity import profanity
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("merged.csv")
fake_data = pd.read_csv("fake_news.csv")
data.columns
fake_data.columns

# ---Data Cleaning---

# remove punctuations
data['human_written_processed'] = data['human_written_text'].map(lambda x: re.sub(r'[,\.!?]', '', x))
data['human_written_processed'].head()

fake_data['fake_processed'] = fake_data['main_text'].map(lambda x: re.sub(r'[,\.!?]', '', x))
fake_data['fake_processed'].head()

data['ai_processed'] = data['ai_generated_text'].map(lambda x: re.sub(r'[,\.!?]', '', x))
data['ai_processed'].head()

# sentence and words tokenization
real_tokens = data['human_written_processed'].map(lambda x: word_tokenize(x))
real_sentences = data['human_written_text'].map(lambda x: sent_tokenize(x))

fake_tokens = fake_data['fake_processed'].map(lambda x: word_tokenize(x))
fake_sentences = fake_data['main_text'].map(lambda x: sent_tokenize(x))

ai_tokens = data['ai_processed'].map(lambda x: word_tokenize(x))
ai_sentences = data['ai_generated_text'].map(lambda x: sent_tokenize(x))

# remove stop words
stop_words = stopwords.words('english')
real_fil = real_tokens.map(lambda x: [word for word in x if not word.lower() in stop_words])
fake_fil = fake_tokens.map(lambda x: [word for word in x if not word.lower() in stop_words])
ai_fil = ai_tokens.map(lambda x: [word for word in x if not word.lower() in stop_words])

# lemmatization/stemming
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    long_token = [token for token in text if len(token) > 3]
    result = [lemmatize_stemming(token) for token in long_token]   
    return result

real_docs = [preprocess(row) for row in real_fil]
fake_docs = [preprocess(row) for row in fake_fil]
ai_docs = [preprocess(row) for row in ai_fil]

# ---Exploratory Analysis - wordcloud---

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

# human-written real
real_non_verbs = remove_verbs(real_fil)
real_wordcloud = generate_wordcloud(real_non_verbs)
display_wordcloud(real_wordcloud)
plt.savefig(f'wordcloud/real_wordcloud.png', dpi = 300)
plt.show()

# human-written fake
fake_non_verbs = remove_verbs(fake_fil)
fake_wordcloud = generate_wordcloud(fake_non_verbs)
display_wordcloud(fake_wordcloud)
plt.savefig(f'wordcloud/fake_wordcloud.png', dpi = 300)
plt.show()

# AI-generated
ai_non_verbs = remove_verbs(ai_fil)
ai_wordcloud = generate_wordcloud(ai_non_verbs)
display_wordcloud(ai_wordcloud)
plt.savefig(f'wordcloud/ai_wordcloud.png', dpi = 300)
plt.show()

# ---Language Feature---

# word length
def word_len(tokens):
    word_len_each_row = tokens.map(lambda x: [len(word) for word in x])
    return word_len_each_row.sum()

real_word_len_list = word_len(real_tokens) # list for word lengths
real_avg_word_len = np.mean(real_word_len_list)

fake_word_len_list = word_len(fake_tokens) # list for word lengths
fake_avg_word_len = np.mean(fake_word_len_list)

ai_word_len_list = word_len(ai_tokens) # list for word lengths
ai_avg_word_len = np.mean(ai_word_len_list)

# sentence length
def sentence_len(sentences):
    sentence_lengths = sentences.map(lambda x: [len(word_tokenize(sentence)) for sentence in x])
    return sentence_lengths.sum()

real_sentence_len_list = sentence_len(real_sentences) # list for sentence lengths
real_avg_sentence_len = np.mean(real_sentence_len_list)

fake_sentence_len_list = sentence_len(fake_sentences) # list for sentence lengths
fake_avg_sentence_len = np.mean(fake_sentence_len_list)

ai_sentence_len_list = sentence_len(ai_sentences) # list for sentence lengths
ai_avg_sentence_len = np.mean(ai_sentence_len_list)

# article length (sentences)
def article_len(sentences):
    article_len = [len(x) for x in sentences]
    return article_len

real_article_len_list = article_len(real_sentences) # list for article lengths
real_avg_article_len = np.mean(real_article_len_list)

fake_article_len_list = article_len(fake_sentences) # list for article lengths
fake_avg_article_len = np.mean(fake_article_len_list)

ai_article_len_list = article_len(ai_sentences) # list for article lengths
ai_avg_article_len = np.mean(ai_article_len_list)

# article length (tokens)
real_no_word_list = [len(article) for article in real_tokens]
real_no_word = np.mean(real_no_word_list)

fake_no_word_list = [len(article) for article in fake_tokens]
fake_no_word = np.mean(fake_no_word_list)

ai_no_word_list = [len(article) for article in ai_tokens]
ai_no_word = np.mean(ai_no_word_list)

# informality detection and count
profanity.load_censor_words(whitelist_words=['dick', 'Wang', 'HIV'])
# profanity.load_censor_words()

def locate_informal_row(profane_tokens):
    if_informal_words = profane_tokens.map(lambda x: [1 if val == '****' else 0 for val in x])
    no_of_informal_words = if_informal_words.apply(sum)
    informal_loc = no_of_informal_words[no_of_informal_words==1].index
    return informal_loc

real_informal_repl = real_tokens.map(lambda x: [profanity.censor(word) for word in x])
real_informal_row = locate_informal_row(real_informal_repl)

fake_informal_repl = fake_tokens.map(lambda x: [profanity.censor(word) for word in x])
fake_informal_row = locate_informal_row(fake_informal_repl)

ai_informal_repl = ai_tokens.map(lambda x: [profanity.censor(word) for word in x])
ai_informal_row = locate_informal_row(ai_informal_repl)

def find_informal_words(informal_loc, original_tokens, profane_tokens):
    diff_list = list()
    for ind in informal_loc:
        orig = original_tokens.iloc[ind]
        repl = profane_tokens.iloc[ind]
        difference = set(orig).difference(set(repl))
        diff_list.append(difference)
    return diff_list

real_diff_list = find_informal_words(real_informal_row, real_tokens, real_informal_repl)
real_no_informal_words = len(real_diff_list)

fake_diff_list = find_informal_words(fake_informal_row, fake_tokens, fake_informal_repl)
fake_no_informal_words = len(fake_diff_list)

ai_diff_list = find_informal_words(ai_informal_row, ai_tokens, ai_informal_repl)
ai_no_informal_words = len(ai_diff_list)

# No of verificable Facts
nlp = spacy.load("en_core_web_sm")

real_doc = data['human_written_processed'].map(lambda x: nlp(x))
real_no_verifiable_facts = real_doc.map(lambda x: len(x.ents)).to_list()
real_avg_no_verifiable_facts = np.mean(real_no_verifiable_facts)

fake_doc = fake_data['fake_processed'].map(lambda x: nlp(x))
fake_no_verifiable_facts = fake_doc.map(lambda x: len(x.ents)).to_list()
fake_avg_no_verifiable_facts = np.mean(fake_no_verifiable_facts)

ai_doc = data['ai_processed'].map(lambda x: nlp(x))
ai_no_verifiable_facts = ai_doc.map(lambda x: len(x.ents)).to_list()
ai_avg_no_verifiable_facts = np.mean(ai_no_verifiable_facts)

# Percentage of verifiable facts
def derive_percentage(interest, article):
    result = []
    for i in range(len(interest)):
        no = interest[i]
        whole = article[i]
        result.append(no/whole)
    return result

real_percentage_verifiable_facts = derive_percentage(real_no_verifiable_facts, real_no_word_list)
real_avg_percentage_verifiable_facts = np.mean(real_percentage_verifiable_facts)

fake_percentage_verifiable_facts = derive_percentage(fake_no_verifiable_facts, fake_no_word_list)
fake_avg_percentage_verifiable_facts = np.mean(fake_percentage_verifiable_facts)

ai_percentage_verifiable_facts = derive_percentage(ai_no_verifiable_facts, ai_no_word_list)
ai_avg_percentage_verifiable_facts = np.mean(ai_percentage_verifiable_facts)

# (stacked barplot)
categories = ['Real', 'Fake', 'AI']
no_of_verifiable_facts = [real_avg_no_verifiable_facts, fake_avg_no_verifiable_facts, ai_avg_no_verifiable_facts]
article_length_in_tokens = [real_no_word-real_avg_no_verifiable_facts, fake_no_word-fake_avg_no_verifiable_facts, ai_no_word-ai_avg_no_verifiable_facts]
percentage_of_verifiable_facts = [real_avg_percentage_verifiable_facts, fake_avg_percentage_verifiable_facts, ai_avg_percentage_verifiable_facts]
ind = np.arange(len(categories))

plt.figure(figsize=(8, 6))
bars_no = plt.bar(ind, no_of_verifiable_facts, label='No. of Verifiable Facts', color='#ff5722')
plt.bar(ind, article_length_in_tokens, bottom=no_of_verifiable_facts, label='Article Length in Words', color='#4285f4')

def add_bar_labels(bars, data):
    """
    Adds labels to the bars in a bar chart.
    
    Parameters:
    - bars: The bar container returned by plt.bar()
    - data: The data used to create the bars, to get the label values.
    """
    for bar, value in zip(bars, data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_y() + height + 30,
                 f'{"%.2f" % (value.round(4)*100)}%', ha='center', va='center', color='white', fontsize=20)

add_bar_labels(bars_no, percentage_of_verifiable_facts)  
        
plt.ylabel('No. of Words', fontsize=20)
plt.title('Percentage of Verifiable Facts per Article', fontsize=20)
plt.xticks(ind, categories, fontsize=20)
plt.legend()
plt.savefig("plot/verifiable_facts.png", dpi=300, bbox_inches='tight')
plt.show()

# ---Sentimental Analysis---

# polarity
def cal_sentence_polarity(sentences):
    return sentences.map(lambda x: [TextBlob(sentence).sentiment.polarity for sentence in x])

def cal_article_polarity(sentence_pol):
    article_pol = [np.mean(row) for row in sentence_pol]
    return article_pol

real_sentence_pol = cal_sentence_polarity(real_sentences)
real_article_pol = cal_article_polarity(real_sentence_pol)
real_avg_article_pol = np.mean(real_article_pol)

fake_sentence_pol = cal_sentence_polarity(fake_sentences)
fake_article_pol = cal_article_polarity(fake_sentence_pol)
fake_avg_article_pol = np.mean(fake_article_pol)

ai_sentence_pol = cal_sentence_polarity(ai_sentences)
ai_article_pol = cal_article_polarity(ai_sentence_pol)
ai_avg_article_pol = np.mean(ai_article_pol)

# (histograms)
fig, axs = plt.subplots(1, 3, figsize=(6, 4), sharey=True, sharex=True)

data_min, data_max = -0.2, 0.3
bin_width = 0.03
bin_edges_pol = np.arange(start=data_min, stop=data_max + bin_width, step=bin_width)
def ax_plot(i, title, score_list, avg_score, color, bin_edges):
    axs[i].hist(score_list, bins=bin_edges, color=color, alpha=0.7, edgecolor='black', orientation = "horizontal")
    axs[i].axhline(avg_score, color='red', lw = 2, linestyle='--')
    axs[i].text(20,avg_score-0.075,f'{avg_score.round(2)}', color='red',fontsize=15)
    axs[i].set_title(title, fontsize=15)

ax_plot(0, 'Real', real_article_pol, real_avg_article_pol, '#4285f4', bin_edges_pol)
ax_plot(1, 'Fake', fake_article_pol, fake_avg_article_pol, 'gray', bin_edges_pol)
ax_plot(2, 'AI', ai_article_pol, ai_avg_article_pol, '#ff5722', bin_edges_pol)

for ax in axs:
    ax.xaxis.set_tick_params(labelbottom=True)

fig.supylabel('Polarity Score', fontsize=20)
fig.supxlabel('Frequency', fontsize=20)
plt.tight_layout()
plt.savefig('plot/polarity.png', dpi=300, bbox_inches='tight')
plt.show()

# polarity volatility
real_sv = [np.std(sentences) for sentences in real_sentence_pol]
real_avg_sv = np.mean(real_sv)
fake_sv = [np.std(sentences) for sentences in fake_sentence_pol]
fake_avg_sv = np.mean(fake_sv)
ai_sv = [np.std(sentences) for sentences in ai_sentence_pol]
ai_avg_sv = np.mean(ai_sv)

# (histograms)
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

axs[0].hist(real_sv, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(fake_sv, color='green', alpha=0.7, label='Group 2')
axs[2].hist(ai_sv, color='orange', alpha=0.7, label='Group 3')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('Human-Written Fake News')
axs[2].set_title('AI-generated News')

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

fake_no_affect_word = cal_no_of_affect_words(fake_fil)
fake_avg_no_affect_word = np.mean(fake_no_affect_word)

ai_no_affect_word = cal_no_of_affect_words(ai_fil)
ai_avg_no_affect_word = np.mean(ai_no_affect_word)

# (histograms)
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

axs[0].hist(real_no_affect_word, color='blue', alpha=0.7, label='Group 1')
axs[1].hist(fake_no_affect_word, color='green', alpha=0.7, label='Group 2')
axs[2].hist(ai_no_affect_word, color='orange', alpha=0.7, label='Group 3')

axs[0].set_title('Human-Written Real News')
axs[1].set_title('Human-Written Fake News')
axs[2].set_title('AI-generated News')

for ax in axs:
    ax.set_xlabel('Number of Affection Words')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# percentage of affect words
real_percentage_affect_words = derive_percentage(real_no_affect_word, real_no_word_list)
real_percentage_avg_affect_words = np.mean(real_percentage_affect_words)

fake_percentage_affect_words = derive_percentage(fake_no_affect_word, fake_no_word_list)
fake_percentage_avg_affect_words = np.mean(fake_percentage_affect_words)

ai_percentage_affect_words = derive_percentage(ai_no_affect_word, ai_no_word_list)
ai_percentage_avg_affect_words = np.mean(ai_percentage_affect_words)

# (stacked barplot)
no_of_affect_word = [real_avg_no_affect_word, fake_avg_no_affect_word, ai_avg_no_affect_word]
article_length_in_tokens_aw = [real_no_word-real_avg_no_affect_word, fake_no_word-fake_avg_no_affect_word, ai_no_word-ai_avg_no_affect_word]
percentage_of_affect_words = [real_percentage_avg_affect_words, fake_percentage_avg_affect_words, ai_percentage_avg_affect_words]
ind = np.arange(len(categories))

plt.figure(figsize=(8, 6))
bars_no_aw = plt.bar(ind, no_of_affect_word, label='No. of Affect Words', color='#ff5722')
plt.bar(ind, article_length_in_tokens_aw, bottom=no_of_affect_word, label='Article Length in Words', color='#4285f4')
        
add_bar_labels(bars_no_aw, percentage_of_affect_words)  
        
plt.ylabel('No. of Words', fontsize=20)
plt.title('Percentage of Affect Words per Article', fontsize=20)
plt.xticks(ind, categories, fontsize=20)
plt.legend()
plt.savefig("plot/emotiveness.png", dpi=300, bbox_inches='tight')
plt.show()

# subjectivity
def cal_sentence_subjectivity(sentences):
    return sentences.map(lambda x: [TextBlob(sentence).sentiment.subjectivity for sentence in x])

def cal_article_subjectivity(sentence_sub):
    article_pol = [np.mean(row) for row in sentence_sub]
    return article_pol

real_sentence_sub = cal_sentence_subjectivity(real_sentences)
real_article_sub = cal_article_subjectivity(real_sentence_sub)
real_avg_sub = np.mean(real_article_sub)

fake_sentence_sub = cal_sentence_subjectivity(fake_sentences)
fake_article_sub = cal_article_subjectivity(fake_sentence_sub)
fake_avg_sub = np.mean(fake_article_sub)

ai_sentence_sub = cal_sentence_subjectivity(ai_sentences)
ai_article_sub = cal_article_subjectivity(ai_sentence_sub)
ai_avg_sub = np.mean(ai_article_sub)

# (histograms)
data_min, data_max = 0, 0.7
bin_width = 0.05
bin_edges_sub = np.arange(start=data_min, stop=data_max + bin_width, step=bin_width)

fig, axs = plt.subplots(1, 3, figsize=(6, 4), sharey=True, sharex=True)

ax_plot(0, 'Real', real_article_sub, real_avg_sub, '#4285f4', bin_edges_sub)
ax_plot(1, 'Fake', fake_article_sub, fake_avg_sub, 'gray', bin_edges_sub)
ax_plot(2, 'AI', ai_article_sub, ai_avg_sub, '#ff5722', bin_edges_sub)

for ax in axs:
    ax.xaxis.set_tick_params(labelbottom=True)

fig.supylabel('Subjectivity Score', fontsize=20)
fig.supxlabel('Frequency', fontsize=20)

plt.tight_layout()
plt.savefig('plot/subjectivity.png', dpi=300, bbox_inches='tight')
plt.show()

# ---Topic Distribution---

def find_topic_distribution(docs):
    dic = gensim.corpora.Dictionary(docs) # dictionary - the number of times a word appears
    dic.filter_extremes(no_below=15, keep_n= 100000) # remove very rarewords
    bow_corpus = [dic.doc2bow(doc) for doc in docs] # create the Bag-of-words model for each document
    lda_model =  gensim.models.LdaMulticore(bow_corpus, # running LDA
                                   num_topics = 5, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    return lda_model

real_lda_model = find_topic_distribution(real_docs)
fake_lda_model = find_topic_distribution(fake_docs)
ai_lda_model = find_topic_distribution(ai_docs)

for idx, topic in real_lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

for idx, topic in fake_lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

for idx, topic in ai_lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

# ---Export Features---
sentiment_df = pd.DataFrame({
    'real_article_pol': real_article_pol, 
    'fake_article_pol': fake_article_pol,
    'ai_article_pol': ai_article_pol, 
    'real_sv': real_sv,
    'fake_sv': fake_sv,
    'ai_sv': ai_sv,
    'real_no_affect_word': real_no_affect_word,
    'fake_no_affect_word': fake_no_affect_word,
    'ai_no_affect_word': ai_no_affect_word,
    'real_percentage_affect_words': real_percentage_affect_words,
    'fake_percentage_affect_words': fake_percentage_affect_words,
    'ai_percentage_affect_words': ai_percentage_affect_words,
    'real_article_sub': real_article_sub,
    'fake_article_sub': fake_article_sub,
    'ai_article_sub': ai_article_sub})

sentiment_df.to_csv("features/sentiment_features.csv", index=False, encoding='utf-8-sig')

language_lists = [real_word_len_list, fake_word_len_list, ai_word_len_list, 
                  real_sentence_len_list, fake_sentence_len_list, ai_sentence_len_list,
                  real_article_len_list, fake_article_len_list, ai_article_len_list, 
                  real_no_word_list, fake_no_word_list, ai_no_word_list,
                  real_no_verifiable_facts, fake_no_verifiable_facts, ai_no_verifiable_facts,
                  real_percentage_verifiable_facts, fake_percentage_verifiable_facts, ai_percentage_verifiable_facts]

# Find the index and the longest list
max_length_index, max_length_list = max(enumerate(language_lists), key=lambda x: len(x[1]))

# Length of the longest list
max_length = len(max_length_list)

# Extend each list to match the max_length
extended_lists = [lst + [np.nan] * (max_length - len(lst)) for lst in language_lists]

language_df = pd.DataFrame({
    'real_word_len_list': extended_lists[0],
    'fake_word_len_list': extended_lists[1],
    'ai_word_len_list': extended_lists[2],
    'real_sentence_len_list': extended_lists[3],
    'fake_sentence_len_list': extended_lists[4],
    'ai_sentence_len_list': extended_lists[5],
    'real_article_len_list': extended_lists[6],
    'fake_article_len_list': extended_lists[7],
    'ai_article_len_list': extended_lists[8],
    'real_no_word_list': extended_lists[9],
    'fake_no_word_list': extended_lists[10],
    'ai_no_word_list': extended_lists[11],
    'real_no_verifiable_facts': extended_lists[12],
    'fake_no_verifiable_facts': extended_lists[13],
    'ai_no_verifiable_facts': extended_lists[14],
    'real_percentage_verifiable_facts': extended_lists[15],
    'fake_percentage_verifiable_facts': extended_lists[16],
    'ai_percentage_verifiable_facts': extended_lists[17]})

language_df.to_csv("features/language_features.csv", index=False, encoding='utf-8-sig')
