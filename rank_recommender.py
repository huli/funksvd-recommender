import numpy as np
import pandas as pd


class RankBasedRecommender():
    '''
    Simple movie recommender based on their overall ratings
    '''

    def fit(self, movies, reviews):
        '''
        Creates internal ranking list
        
        INPUT:
        movies - a data frame with movies
        reviews - a data frame with revies
        '''

        self.ranked_movies = self.create_ranked_df(movies, reviews)

    def predict(self, n_top):
        '''
        INPUT:
        n_top - an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

        OUTPUT:
        top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''

        top_movies = self.ranked_movies['movie'][:n_top]

        return top_movies.index.values, top_movies.values
            
    def create_ranked_df(self, movies, reviews):
        '''
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe

        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, then time, and must have more than 4 ratings
        '''

        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] >= 5]

        return ranked_movies
