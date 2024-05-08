import os
import numpy as np
import pandas as pd
import random
import nltk
import csv
from openai import OpenAI

#----------- setup ---------------
random.seed(100)

train = pd.read_csv("train.tsv", sep="\t")
health_subject = train["subjects"].str.contains("Health", na=False)
true_news = train["label"]=="true"
train_health = train.loc[health_subject & true_news]
# extract a sample with 100 rows
train_set = train_health.sample(100)

# the average number of tokens in the comparison dataset
cal_token = lambda t : len(nltk.word_tokenize(t))
train_tokens = train_set["main_text"].apply(cal_token)
np.average(train_tokens) # 500-700

# the average number of words in the comparison dataset
cal_word = lambda t : len(t.split())
train_word = train_set["main_text"].apply(cal_word)
np.average(train_word) # 450-660

#------------ generate output ------------------

client = OpenAI(api_key= my_key)

# List of prompts to loop over
prompts = train_set['claim']

# Function to make API call
def generate_news_article(summary):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        presence_penalty=1,
        temperature=0.5,
        messages=[{"role": "system", 
                   "content": "You are a news reporter specialized in reporting on current public health news, focusing on delivering accurate and timely information."},
                   {"role": "user", "content": "Please write a 500-600 word newspaper article with this summary or claim: \n\n'{summary}'. Only produce the main content (no titles, subtitles, dates, etc.)."}
        ]
        )
    return completion.choices[0].message.content

# a single test case
generate_news_article("Illinois gets federal grant to improve maternal health.")

# File to store the results
filename = 'ai-generated.csv'

# Write headers and responses to CSV
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # Write the header
#     writer.writerow(['claim_id', 'generated_news_article'])
    
#     # Loop over news summaries, generate article and write to CSV
#     for summary in prompts:
#         index = prompts[prompts == summary].index[0]
#         claim_id = train_set.loc[[index]]["claim_id"]
#         article = generate_news_article(summary)
#         # Write the columns to the CSV
#         writer.writerow([claim_id, article])

#----------- merge human-written real news and ai-generated news ---------------

ai2 = pd.read_excel("ai2.xlsx")
ai2['claim_id'] = ai2['claim_id'].apply(lambda x : x.split()[1])

train_health.loc[:,"claim_id"] = train_health.loc[:,"claim_id"].astype(int)
ai2.loc[:,"claim_id"] = ai2.loc[:,"claim_id"].astype(int)

df = pd.merge(ai2, train_health, on="claim_id")

colnames = ["claim_id", "claim_y", "main_text_y", "generated_news_article", "label"]
df = df.loc[:, colnames]
df = df.rename(columns={'claim_y': 'claim', 
                        'main_text_y' : 'human_written_text',
                        'generated_news_article' : 'ai_generated_text'})

os.chdir("C:/Users/cicil/Desktop/py")
df.to_csv("merged.csv", index=False, encoding='utf-8-sig')

#----------- create human-written fake news dataset ---------------
fake_news = train["label"]=="false"
fake_health = train.loc[health_subject & fake_news]
# extract a sample with 100 rows
fake_news_set = fake_health.sample(100)

fake_news_set.to_csv("fake_news.csv", index=False, encoding='utf-8-sig')