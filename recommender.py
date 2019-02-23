import numpy as np
import pandas as pd
import datetime
import recommender_functions as rf
import sys 

class Recommender():
    '''
    The Recommender uses FunkSVD to make predictions of exact ratings.  
    * If the user is new collaborative filtering will not work and the class will fallback to a knowlege 
    based approach and return the highest ranked movies for the user.
    * If given a movie the class will provide content based recommendations by calculating the closest
    movies by euclidean distance
    '''

    def __init__(self, ):
        '''
        No initialization at the moment
        '''

    def set_status(self, status)
        print('[{}] - {}'.format(datetime.datetime.now().time(), status))

    def fit(self, reviews_path, movies_path, latent_features=12, learning_rate=0.0001, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        reviews_path - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_path - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        '''
        # Store inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Load input data
        self.set_status('Reading csv-files...')
        self.reviews = pd.read_csv(reviews_path)
        self.movies = pd.read_csv(movies_path)

        # Create user item-matrix from all provided reviews
        self.set_status('Creating user-item matrix...')
        user_items = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = usr_itm.groupby(['user_id','movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_item_df)

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.n_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # Initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        self.set_status("\tOptimizaiton Statistics")
        self.set_status("\tIterations | Mean Squared Error ")

        for iter in range(self.iters):
            
            # Initialize sum of squared error
            sse_accum = 0

            # For each user-item pair
            for user in range(self.n_users):
                for movie in range(self.n_movies):

                    if self.rating_exists(user, movie):
                        
                        # Compute prediction with dot product of user-item matrix
                        prediction = np.dot(user_mat[user, :], movie_mat[:, movie])

                        # Caculate MSE
                        error = self.user_item_mat[user, movie] - prediction

                        # Update accumulator
                        sse_accum += error**2

                        # Update the values in both matrices in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[u, k] += self.learning_rate * (2 * error * movie_mat[k, m])
                            movie_mat[k, m] += self.learning_rate * (2 * error * user_mat[u, k])

            self.set_status('{:>20} iteration: {:>10.5f} MSE'.format(iter+1, sse_accum/self.n_ratings))
                    
        self.set_status('Stopped gradient descent optimization...')

        # Update state according to fitted parameters
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Perform knowlege based fit
        self.set_status('Performing knowlege based fit...')
        self.ranked_movies = rf.create_ranked_df(self.movie, self.reviews)

        self.set_status('Fitting finished.')
        

    def rating_exists(self, user, movie):
        return self.user_item_mat[user, movie] > 0
        
    def predict_rating(self, ):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
