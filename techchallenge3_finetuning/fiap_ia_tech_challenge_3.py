# -*- coding: utf-8 -*-
"""Fiap_ia_tech_challenge_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lE2XDPRPcHa1RU7-lysdyoHybBdfV5Wm

# Tech Challenge 3

Fine tunning com dataset da amazon

O objetivo é utilizar o dataset fornecido com dados sobre produtos da amazon para treinar o modelo gpt-4o-mini da openai de formar que ele saiba responder perguntas sobre os produtos utilizados no treinamento. É uma forma de dar ao modelo conhecimento sobre informações "privadas" com as quais ele não foi originalmente treinado.
"""

!pip install openai tqdm

"""# Carregar o arquivo de dataset a partir do google drive"""

# mount google drive
from google.colab import drive
drive.mount('/content/drive')

#utilizando um dataset limitado em 100 mil linhas
dir_path = '/content/drive/My Drive/leandrodvd@gmail.com/FIAP Pós IA/tech challenge 3'
file_path = dir_path + '/trn_100k.json'

# Example: Reading a CSV file from Google Drive
import pandas as pd

# Replace 'your_file.csv' with the actual file name
df = pd.read_json(file_path, lines=True)

# Display the DataFrame
df.head()

df.info()

# remover linhas com content vazio

# Remove rows where 'title' or 'content' is NaN
df = df.dropna(subset=['title', 'content'])

# Remove rows where 'title' or 'content' is an empty string
df = df[(df['title'].str.strip() != '') & (df['content'].str.strip() != '')]

df.head(100)

"""# separar um pequeno subset para testar chamadas de llm sem fine tuning"""

test_df = df.head(500).copy()

test_df.head(10)

"""# Gerar algumas perguntas sobre os produtos"""

import random

random.seed(42)

question_options = [
      'Please describe the ',
      'What can you tell me about the ',
      'Tell me about  ',
      'What is the product ',
      'Give me description of '
  ]

def get_question_for_product(title):
  question = random.choice(question_options) + '\"' + title + '\" from amazon'
  return question

test_df['question'] = test_df['title'].apply(get_question_for_product)

test_df.head(100)

"""# Chamar o modelo gpt-4o-mini perguntando sobre os produtos"""

from openai import OpenAI
from google.colab import userdata

openai_api_key = userdata.get('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

def call_open_ai(question):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": question}
    ]
    )
  return completion.choices[0].message.content

from tqdm import tqdm
tqdm.pandas()

test_df['pre_test_response'] = test_df['question'].progress_apply(call_open_ai)

test_df[['title','content','question', 'pre_test_response']].head(10)

"""# Preparar os dados para fine tunning do modelo"""

def build_fine_tunning_prompt(row):
  return {"messages": [{"role": "user", "content": row['question']}, {"role": "assistant", "content": "This is what I know about '"+row['title']+ "' from amazon: "+row['content']}]}

test_df['fine_tunning_prompt'] = test_df.apply(build_fine_tunning_prompt, axis=1)

test_df.head(10)

import json
# save fine tunning prompt to jsonl file
prompts_file_path = dir_path + '/fine_tunning_prompt.jsonl'
# test_df[['fine_tunning_prompt']].to_json(prompts_file_path, orient='records', lines=True)
with open(prompts_file_path, 'w') as fine_tunning_prompt_file:
    for item in test_df['fine_tunning_prompt']:
        fine_tunning_prompt_file.write(f"{json.dumps(item)}\n")

with open(prompts_file_path, 'r') as fine_tunning_prompt_file:
  print(fine_tunning_prompt_file.read())

"""# Executar o fine tunning do modelo gpt-4o-mini

"""

fine_tunning_file_create_response = client.files.create(
  file=open(prompts_file_path, "rb"),
  purpose="fine-tune"
)

print(fine_tunning_file_create_response)
fine_tunning_file_id = fine_tunning_file_create_response.id
print('fine_tunning_file_id:'+fine_tunning_file_id)

fine_tunning_job_create_response = client.fine_tuning.jobs.create(
  training_file=fine_tunning_file_id,
  model="gpt-4o-mini-2024-07-18"
)

print(fine_tunning_job_create_response)
fine_tunning_job_id = fine_tunning_job_create_response.id
print('fine_tunning_job_id:'+fine_tunning_job_id)

# Retrieve the state of a fine-tune
fine_tuning_job_retrieve_response = client.fine_tuning.jobs.retrieve(fine_tunning_job_id)
print(fine_tuning_job_retrieve_response)
fine_tuning_job_status = fine_tuning_job_retrieve_response.status
fine_tuned_model = fine_tuning_job_retrieve_response.fine_tuned_model
print(fine_tuning_job_status)
print('fine_tuned_model:'+str(fine_tuned_model))

"""# Test the fine tuned model"""

def call_open_ai_fine_tuned(question):
  completion = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[{"role": "user", "content": question}]
  )
  return completion.choices[0].message.content

test_df['fine_tune_test_response'] = test_df['question'].progress_apply(call_open_ai_fine_tuned)

test_df[['question','fine_tunning_prompt', 'fine_tune_test_response']].head(500)

# make some new questions
def get_alternate_question_for_product(title):
  question = 'I\'m looking for some information about a product that\'s sold by amazon. The title of the product is: ' + '\"' + title + '\"'
  return question

test_df['alternate_question'] = test_df['title'].apply(get_alternate_question_for_product)

test_df.head()

test_df['fine_tune_test_alternate_response'] = test_df['alternate_question'].progress_apply(call_open_ai_fine_tuned)

test_df.head()

test_df.to_csv(dir_path + '/test_df.csv', index=False)

# try with some other questions
question = "What is the product 'A Day in the Life of Canada' from amazon"

print('question:'+question)
answer = call_open_ai_fine_tuned(question)
print('answer:'+answer)