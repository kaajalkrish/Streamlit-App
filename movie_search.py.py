#!/usr/bin/env python
# coding: utf-8

# In[37]:


get_ipython().system('pip install kaggle')


# In[38]:


import kaggle
import pandas as pd
import zipfile

kaggle.api.authenticate()
dataset = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
kaggle.api.dataset_download_files(dataset)

with zipfile.ZipFile(
    "imdb-dataset-of-top-1000-movies-and-tv-shows.zip", "r"
) as zip_ref:
    zip_ref.extractall(".")

movies = pd.read_csv("imdb_top_1000.csv")
print(movies.columns)
print(movies[["Series_Title", "Overview"]].head(10))


# # WHICH COLUMN WILL YOU USE? 

# In[39]:


import pandas as pd

# Load the dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Print the column names to see what's available
print("Available columns:", movies.columns)

# Based on the column names, you can decide which columns to use for your semantic search
# For example, you might choose "Series_Title" and "Overview" columns
columns_to_use = ["Series_Title", "Overview"]

print("Columns to use:", columns_to_use)


# # CLEANING THE COLUMNS 

# In[40]:


pip install nltk


# In[41]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords if not already downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Define function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stopwords_list = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords_list]
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Clean the text in selected columns
columns_to_clean = ["Series_Title", "Overview"]

for col in columns_to_clean:
    movies[col] = movies[col].apply(clean_text)

# Display the cleaned DataFrame
print(movies[["Series_Title", "Overview"]].head())


# In[42]:


print(movies[["Series_Title", "Overview"]].head(10))


# # Concatenate the columns needed for your embedding

# In[43]:


movies["Concatenated_Text"] = movies["Series_Title"] + " " + movies["Overview"]


# In[44]:


print(movies[["Concatenated_Text"]].head())


# # Create new column with concatenated and clean text

# In[45]:


# Define function to concatenate and clean text
def concatenate_and_clean_text(series_title, overview):
    # Concatenate the series title and overview
    concatenated_text = series_title + " " + overview
    # Clean the concatenated text
    cleaned_text = clean_text(concatenated_text)
    return cleaned_text

# Apply the function to create a new column
movies["Concatenated_and_Clean_Text"] = movies.apply(lambda row: concatenate_and_clean_text(row["Series_Title"], row["Overview"]), axis=1)

# Display the DataFrame with the new column
print(movies[["Concatenated_and_Clean_Text"]].head())


# # PREPARING THE TEXT FOR EMBEDDING 

# In[46]:


# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords if not already downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stopwords_list = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords_list]
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Load the dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Clean and concatenate text for each record
def prepare_text_to_embed(row):
    # Clean the text in Series_Title and Overview columns
    series_title_cleaned = clean_text(row['Series_Title'])
    overview_cleaned = clean_text(row['Overview'])
    # Concatenate cleaned text
    concatenated_text = series_title_cleaned + " " + overview_cleaned
    return concatenated_text

# Apply the function to create a new column
movies['Text_to_Embed'] = movies.apply(prepare_text_to_embed, axis=1)

# Display the DataFrame with the new column
print(movies[['Series_Title', 'Overview', 'Text_to_Embed']].head())


# In[47]:


get_ipython().system('pip install tensorflow')


# In[48]:


get_ipython().system('pip install tensorflow_hub')


# # LOADING UNIVERSAL SENTENCE ENCODER IN PROGRAM

# In[49]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow_hub as hub

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# # SENTENCE EMBEDDING: UNIVERSAL SENTENCE ENCODER

# In[50]:


# Encode text from the 'Text_to_Embed' column using the Universal Sentence Encoder
embeddings = embed(movies['Text_to_Embed'])

# Print the embeddings
for i, embedding in enumerate(embeddings):
    print("Movie Title:", movies.iloc[i]['Series_Title'])
    print("Overview:", movies.iloc[i]['Overview'])
    print("Embedding:", embedding)
    print("Shape:", embedding.shape)
    print()


# # EMBEDDING TEXT GENERATED FOR EACH RECORD

# In[51]:


# Encode text from the 'Text_to_Embed' column using the Universal Sentence Encoder
embeddings = embed(movies['Text_to_Embed'])

# Print the embeddings
print("Number of embeddings:", len(embeddings))
for i, embedding in enumerate(embeddings):
    print("Embedding for record", i+1, ":", embedding)
    print("Shape:", embedding.shape)
    print()


# # STORING EMBEDDINGS IN VECTOR DATABASE: SQLITE3

# In[52]:


import sqlite3
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('movie_embedding.db')
c = conn.cursor()

# Create a table to store embeddings
c.execute('''CREATE TABLE IF NOT EXISTS embeddings
             (id INTEGER PRIMARY KEY, embedding BLOB)''')

# Insert embeddings into the SQLite database
for i, embedding in enumerate(embeddings):
    c.execute("INSERT INTO embeddings (id, embedding) VALUES (?, ?)",
              (i+1, np.array(embedding).tobytes()))

# Commit changes and close connection
conn.commit()
conn.close()


# In[53]:


get_ipython().system('pip install streamlit')
get_ipython().run_line_magic('load_ext', 'streamlit')


# In[54]:


# Step 1: Import necessary libraries
import streamlit as st
import sqlite3
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Step 2: Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Step 3: Connect to the SQLite database containing movie embeddings
conn = sqlite3.connect('movie_embedding.db')
c = conn.cursor()

# Step 4: Define function to embed user input
def embed_text(text):
    return embed([text])[0]

# Step 5: Define function to search for closest embeddings
def search_closest_embeddings(input_embedding, threshold=0.9):
    c.execute("SELECT id, embedding FROM embeddings")
    embeddings = c.fetchall()
    closest_embeddings = []
    for movie_id, movie_embedding_bytes in embeddings:
        movie_embedding = np.frombuffer(movie_embedding_bytes, dtype=np.float32)
        similarity = np.dot(input_embedding, movie_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(movie_embedding))
        if similarity >= threshold:
            closest_embeddings.append((movie_id, similarity))
    closest_embeddings.sort(key=lambda x: x[1], reverse=True)
    return closest_embeddings

# Step 6: Define the Streamlit app
def main():
    st.title("Movie Search Engine")

    # Step 7: Input text from the user
    user_input = st.text_input("Enter your search query:")

    # Step 8: Embed the user input
    if user_input:
        input_embedding = embed_text(user_input)

        # Step 9: Search for closest embeddings
        closest_embeddings = search_closest_embeddings(input_embedding)

        # Step 10: Output the closest entries found
        st.subheader("Results:")
        if closest_embeddings:
            for movie_id, similarity in closest_embeddings:
                st.write(f"Movie ID: {movie_id}, Similarity: {similarity}")
        else:
            st.write("No matching movies found.")

# Step 11: Run the Streamlit app
if __name__ == "__main__":
    main()


# In[ ]:




