# FunkSVD Recommender Class

## Overview
Simple class which performs different recommendations strategies for movies:
* If user is known the class performs (user-based) collaborative filtering
* If user is unknown the class performs ranking based reommendation and suggests movies based on their overall popularity
* Additionaly a single movie can be passed to the predict method and the class recommends the closest neighbours based on euclidean distance

## Files
* `recommender.py` - Main class which uses different strategies based on input
* `rank_recommender.py` - Implements ranking based recommendations
* `similarity_recommender.py` - Implements similarity based recommendations for movies

## Usage
You can use the class the following way:

```python
import recommender as r

# Instantiate and fit recommender
recommender = r.Recommender()
recommender.fit(reviews_path='data/train_data.csv', movies_path= 'data/movies_clean.csv', 
    learning_rate=.01, iters=100)
# -> [10:46:20.515019] - Reading csv-files...
# -> [10:46:20.947056] - Creating user-item matrix...
# -> [10:46:21.040788] -     Optimizaiton Statistics
# -> [10:46:21.040788] -     Iterations | Mean Squared Error
# -> [10:46:26.306418] -     1               12.136 MSE
# -> [10:46:31.290779] -     2                5.374 MSE
# -> [::]              -     ...
# -> [10:47:12.249033] -     10               0.302 MSE
# -> [10:47:12.249033] - Stopped gradient descent optimization...
# -> [10:47:12.249033] - Performing ranking based fit...
# -> [10:47:16.290893] - Performing similarity based fit...
# -> [10:48:57.614578] - Fitting finished.

# Predict rating for known user and know movie
recommender.predict_rating(user_id=8, movie_id=2844)
# -> [10:32:18.563847] - Prediction for user 8 and movie  Fantômas - À l'ombre de la guillotine (1913): 3

# Make recommendation for know user
recommender.make_recommendations(8,'user')
# -> Movie recommendations for user 8: ['Life of Pi (2012)', 'The Hobbit: An Unexpected Journey (2012)', 'Silver Linings Playbook (2012)', 'The Intouchables (2011)', 'Django Unchained (2012)']

# Make recommendation for unknow user
recommender.make_recommendations(1,'user')
# -> [10:32:18.565847] - User not in database. Falling back to ranking based recommendation.
# -> [10:32:18.566848] - Movie recommendations for user 1: ['Goodfellas (1990)' 'Step Brothers (2008)' 'American Beauty  (1999)' 'There Will Be Blood (2007)' 'Gran Torino (2008)']

# Find neighbours for known movie
recommender.make_recommendations(1853728)
# -> 

# Find neighbours for unknown movie
recommender.make_recommendations(1)
# -> 

```

## TODOs:
* Extract FunkSVD as own component and make the current recommender class an orchestrator
* Write test for the individual components
