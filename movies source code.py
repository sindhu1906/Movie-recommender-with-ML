#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

#reading file
pd.set_option('display.max_columns', 999)
movies = pd.read_csv("movies.csv", header=0)
movies = movies.replace({np.nan: None}) # replace NaN with None

print(movies.head())

ratings = pd.read_csv("ratings_small.csv", header=0)
print(ratings.head())

links = pd.read_csv("links.csv", header=0)
print(links.head())

#Total no. of movies
print(len(movies))

links.columns=["movieId","imdb_id","tmdbId"]
print(links.head())



# In[2]:


#Demographic filtering recommender system
"""
Before getting started with this -

we need a metric to score or rate movie
Calculate the score for every movie
Sort the scores and recommend the best rated movie to the users.
We can use the average ratings of the movie as the score but using this won't be fair enough since a movie with 8.9 average rating and only 3 votes cannot be considered better than the movie with 7.8 as as average rating but 40 votes. So, I'll be using IMDB's weighted rating (wr) which is given as :-

where,

v is the number of votes for the movie;
m is the minimum votes required to be listed in the chart;
R is the average rating of the movie; And
C is the mean vote across the whole report
We already have v(vote_count) and R (vote_average) and C can be calculated as
"""

df2=movies[0:5000]
df2.head()


# In[3]:


C= df2['average_vote'].mean()
C

m= df2['num_votes'].quantile(0.9)
m

q_movies = df2.copy().loc[df2['num_votes'] >= m]
q_movies.shape


# In[4]:


def weighted_rating(x, m=m, C=C):
    v = x['num_votes']
    R = x['average_vote']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'num_votes', 'average_vote', 'score']].head(10)


# In[5]:


pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# In[6]:


#Content based filtering

df2['description'].head(5)

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['description'] = df2['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[7]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# In[8]:


get_recommendations('The Godfather')


# In[11]:


#keyword based recommender

filledna=df2.fillna('')
filledna.head(2)


# In[12]:


def clean_data(x):
        return str.lower(x.replace(" ", ""))
    
features=['title','director','cast','genres','description']
filledna=filledna[features]

for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)
    
filledna.head(2)


# In[14]:


def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['genres']+' '+ x['description']

filledna['soup'] = filledna.apply(create_soup, axis=1)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])

def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# In[18]:


get_recommendations_new('Toy Story', cosine_sim2)


# In[23]:


# Install a conda package in the current Jupyter kernel
import sys
pip install scikit-surprise


# In[24]:


#User rating based recommender

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
reader = Reader()

ratings.head()


# In[25]:


ratings.shape


# In[27]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])


# In[28]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[29]:


#Let us pick user with user Id 1 and check the ratings she/he has given.
ratings[ratings['userId'] == 1]

svd.predict(1, 302, 3)


# In[ ]:


#For movie with ID 302, we get an estimated prediction of 2.892. One startling feature of this recommender system is that it doesn't care what the movie is (or what it contains). It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.

