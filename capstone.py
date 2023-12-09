#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


""" load data """

spotify = pd.read_csv('spotify52kData.csv')
star = pd.read_csv('starRatings.csv', header=None)

# spotify.head()
# star.head()


# In[31]:


"""

convert data to proper types (numeric, boolean...)

in spotify52kData,
songNumber, popularity, duration, danceability, energy, key, loudness, mode
speechiness, acoustincness, instrumentalness, liveness, valence, tempo, and time_signature
are all converted to numeric data

all data in star should be numeric

"""

spotify['songNumber'] = pd.to_numeric(spotify['songNumber'], errors='coerce')
spotify['popularity'] = pd.to_numeric(spotify['popularity'], errors='coerce')
spotify['duration'] = pd.to_numeric(spotify['duration'], errors='coerce')
spotify['danceability'] = pd.to_numeric(spotify['danceability'], errors='coerce')
spotify['energy'] = pd.to_numeric(spotify['energy'], errors='coerce')
spotify['key'] = pd.to_numeric(spotify['key'], errors='coerce')
spotify['loudness'] = pd.to_numeric(spotify['loudness'], errors='coerce')
spotify['mode'] = pd.to_numeric(spotify['mode'], errors='coerce')
spotify['speechiness'] = pd.to_numeric(spotify['speechiness'], errors='coerce')
spotify['acousticness'] = pd.to_numeric(spotify['acousticness'], errors='coerce')
spotify['instrumentalness'] = pd.to_numeric(spotify['instrumentalness'], errors='coerce')
spotify['liveness'] = pd.to_numeric(spotify['liveness'], errors='coerce')
spotify['valence'] = pd.to_numeric(spotify['valence'], errors='coerce')
spotify['tempo'] = pd.to_numeric(spotify['tempo'], errors='coerce')
spotify['time_signature'] = pd.to_numeric(spotify['time_signature'], errors='coerce')
star = star.apply(pd.to_numeric, errors='coerce')


# In[79]:


""" row-wise removal overall """

# spotify = spotify.dropna()

# star = star.dropna()

""" row-wise removal based on certain column """

# spotify = spotify.dropna(subset=['Popularity'])

# star = star.drop(star[0].isna().index) # user removal based on whether that user rated a song

""" column-wise removal based on certain row (make more sense to do in star data) """

# star = star.drop(df.columns[star.iloc[0].isna()], axis=1) # song removal based on whether it has missing ratings

""" imputation with average """

# spotify['popularity'] = spotify['popularity'].fillna(spotify['popularity'].mean())

# star = star.fillna(star.mean()) # impute nan values with average ratings of songs

# star = star.fillna(star.mean(axis=1)) # impute nan values with average ratings of users to all songs

