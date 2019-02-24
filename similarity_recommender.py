import numpy as np
import pandas as pd


class SimilarityBasedRecommender():
    '''
    Predicts closest movie based on euclidean distance
    '''

    def fit(self, movies):
        self.movies = movies
        
        # Calculate dot product to get similar movies
        movie_content = np.array(self.movies.iloc[:,4:])
        self.dot_prod_movies = movie_content.dot(np.transpose(movie_content))

    def predict(self, movie_id, num_of_recs):
        ''' Recommends the closest movies based on euclidean distance metric '''
        if self.movies.movie_id.isin([movie_id]).any():
            return self.find_similar_movies(movie_id, num_of_recs)
        else:
            return [], []

    def find_similar_movies(self, movie_id, num_of_recs):
        '''
        INPUT
        movie_id - a movie_id
        movies_df - original movies dataframe
        OUTPUT
        similar_movies - an array of the most similar movies by title
        '''

        # find the row of each movie id
        movie_idx = np.where(self.movies['movie_id'] == movie_id)[0][0]

        # find the most similar movie indices - to start we take 
        # only movies with the exact same rating
        similar_idxs = np.where(self.dot_prod_movies[movie_idx] == np.max(self.dot_prod_movies[movie_idx]))[0]

        # pull the movie titles based on the indices
        similar_movies = np.array(self.movies.iloc[similar_idxs, ]['movie'])

        return similar_idxs[:num_of_recs], similar_movies[:num_of_recs]