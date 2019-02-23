import numpy as np
import pandas as pd
import datetime
import sys 
from rank_recommender import RankBasedRecommender

class Recommender():
    '''
    The Recommender uses FunkSVD to make predictions of exact ratings.  
    * If the user is new collaborative filtering will not work and the class will fallback to a knowlege 
    based approach and return the highest ranked movies for the user.
    * If given a movie the class will provide content based recommendations by calculating the closest
    movies by euclidean distance
    '''

    def __init__(self, ):
        self.rank_recommender = RankBasedRecommender()

    def set_status(self, status):
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
        self.user_item_df = user_items.groupby(['user_id','movie_id'])['rating'].max().unstack()
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
                            user_mat[user, k] += self.learning_rate * (2 * error * movie_mat[k, movie])
                            movie_mat[k, movie] += self.learning_rate * (2 * error * user_mat[user, k])

            self.set_status('\t{}\t\t{:>6.3f} MSE'.format(iter+1, sse_accum/self.n_ratings))
                    
        self.set_status('Stopped gradient descent optimization...')

        # Update state according to fitted parameters
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Perform knowlege based fit
        self.set_status('Performing knowlege based fit...')
        self.rank_recommender.fit(self.movies, self.reviews)

        self.set_status('Fitting finished.')
        

    def rating_exists(self, user, movie):
        return self.user_item_mat[user, movie] > 0
        
    def predict_rating(self, user_id, movie_id):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''

        try:
            user_row = self.find_index(user_id, self.user_ids_series)
            movie_column = self.find_index(movie_id, self.movie_ids_series)

            # Caculate prediction with dot product (U . V)
            prediction = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_column])

            movie_name = self.find_movie_name(movie_id)
            self.set_status('Prediction for user {} and movie {}: {:.0f}'.format(user_id, movie_name, prediction))

            return prediction
        
        except:
            self.set_status('Prediction not feasible. User {} or Movie {} not in database!'.format(user_id, movie_id))


    def find_movie_name(self, movie_id):
        ''' Extracts the movie name from the rather unclean data set '''

        movie_name = str(self.movies[self.movies['movie_id']== movie_id]['movie'])[5:]
        return movie_name.replace('\nName: movie, dtype: object', '')

    def find_index(self, serie, id):
        ''' Returns index of row or column from id in serie '''
        return np.where(serie == id)[0][0]

    def make_recommendations(self, _id, _id_type='movie', num_of_recs=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        num_of_recs - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                        given movie, or recs for a user_id given
        '''

        movie_ids, movie_names = None, None

        if _id_type == 'movie':
            return self.recommend_closest_movies(_id, num_of_recs)
        
        if _id in self.user_ids_series:
            user_index = self.find_index(self.user_ids_series, _id)

            # Calculate prediction with dot product (U . V)
            prediction = np.dot(self.user_mat[user_index, :], self.movie_mat)

            # Find closest movies
            movie_indices = prediction.argsort()[-num_of_recs:][::-1]
            movie_ids = self.movie_ids_series[movie_indices]
            movie_names = get_movie_names(movie_ids, self.movies)

        else:
            self.set_status('User not in database. Falling back to ranking based recommendation.')
            movie_names = self.rank_recommender.predict(num_of_recs)

        self.set_status('Movie recommendations for user {}: {}'.format(_id, movie_names))
        return movie_ids, movie_names

    def recommend_closest_movies(self, movie_id, num_of_recs):
        ''' Recommends the closest movies based on euclidean distance metric '''
        if movie_id in self.movie_ids_series:
            return list(find_similar_movies(movie_id, self.movies))[:num_of_recs]
        else:
            self.set_status('Movie not in database. Sorry, no recommendations for you!')

def get_movie_names(movie_ids, movies_df):
    '''
    INPUT
    movie_ids - a list of movie_ids
    movies_df - original movies dataframe
    OUTPUT
    movies - a list of movie names associated with the movie_ids
    '''

    # Find the movies by id and return their names
    movie_names = list(movies_df[movies_df['movie_id'].isin(movie_ids)]['movie'])
    
    return movie_names

def find_similar_movies(movie_id, movies_df):
    '''
    INPUT
    movie_id - a movie_id
    movies_df - original movies dataframe
    OUTPUT
    similar_movies - an array of the most similar movies by title
    '''

    # dot product to get similar movies
    movie_content = np.array(movies_df.iloc[:,4:])
    dot_prod_movies = movie_content.dot(np.transpose(movie_content))

    # find the row of each movie id
    movie_idx = np.where(movies_df['movie_id'] == movie_id)[0][0]

    # find the most similar movie indices - to start we take 
    # only movies with the exact same rating
    similar_idxs = np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]

    # pull the movie titles based on the indices
    similar_movies = np.array(movies_df.iloc[similar_idxs, ]['movie'])

    return similar_movies

if __name__ == '__main__':
    
    import recommender as r

    # Instantiate and fit recommender
    recommender = r.Recommender()
    recommender.fit(reviews_path='data/train_data.csv', movies_path= 'data/movies_clean.csv', 
        learning_rate=.01, iters=1)

    # Make various predictions
    print('Predict rating for known user and know movie:')
    recommender.predict_rating(user_id=8, movie_id=2844)

    print('Make recommendation for know user:')
    recommender.make_recommendations(8,'user')
    print('Make recommendation for unknow user:')
    recommender.make_recommendations(1,'user')
    print('Find neighbours for known movie:')
    recommender.make_recommendations(1853728)
    print('Find neighbours for unknown movie:')
    recommender.make_recommendations(1)
