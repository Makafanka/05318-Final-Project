import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVD
# from surprise.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
import warnings; warnings.simplefilter('ignore')


class Recommender:
    def __init__(self, md, links_small, creds, keywords, ratings, surveys):
        """
        Class used to create recommendations from 
        :param md: pd.dataframe, information about movies
        :param links_small: pd.dataframe, links between MovieLens movies and movies in other databases
        :param creds: pd.dataframe, information about the cast and crew of movies
        :param keywords: pd.dataframe, keywords associated with movies
        :param ratings: pd.dataframe, ratings for movies
        :param surveys: pd.dataframe, results of my surveys on CMU students about their movie tastes
        """
        print("Recommender Class")
        self.md = md
        self.links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        self.id_map = links_small[['movieId', 'tmdbId']]
        self.credits = creds
        self.keywords = keywords
        self.ratings = ratings
        self.surveys = surveys
        self.smd = None
        self.svd = None
        self.cosine_sim = None
        self.indices = None
        self.titles = None
        self.m = None
        self.C = None
        
    def weighted_rating(self, x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+self.m) * R) + (self.m/(self.m+v) * self.C)

    def get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    ### Metadata Based Recommender
    def metadata_rec(self):
        self.keywords['id'] = self.keywords['id'].astype('int')
        self.credits['id'] = self.credits['id'].astype('int')
        self.md['genres'] = self.md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.md['year'] = pd.to_datetime(self.md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
        self.md = self.md.drop([19730, 29503, 35587])
        self.md['id'] = self.md['id'].astype('int')
        self.md = self.md.merge(self.credits, on='id')
        self.md = self.md.merge(self.keywords, on='id')
        self.smd = self.md[self.md['id'].isin(self.links_small)]
        self.smd['cast'] = self.smd['cast'].apply(literal_eval)
        self.smd['crew'] = self.smd['crew'].apply(literal_eval)
        self.smd['keywords'] = self.smd['keywords'].apply(literal_eval)
        self.smd['cast_size'] = self.smd['cast'].apply(lambda x: len(x))
        self.smd['crew_size'] = self.smd['crew'].apply(lambda x: len(x))
        self.smd['director'] = self.smd['crew'].apply(self.get_director)
        self.smd['cast'] = self.smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.smd['cast'] = self.smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
        self.smd['keywords'] = self.smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.smd['cast'] = self.smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        self.smd['director'] = self.smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
        self.smd['director'] = self.smd['director'].apply(lambda x: [x,x])

        s = self.smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'keyword'
        s = s.value_counts()
        s = s[s > 1]

        def filter_keywords(x):
            words = []
            for i in x:
                if i in s:
                    words.append(i)
            return words

        stemmer = SnowballStemmer('english')
        self.smd['keywords'] = self.smd['keywords'].apply(filter_keywords)
        self.smd['keywords'] = self.smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
        self.smd['keywords'] = self.smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        self.smd['soup'] = self.smd['keywords'] + self.smd['cast'] + self.smd['director'] + self.smd['genres']
        self.smd['soup'] = self.smd['soup'].apply(lambda x: ' '.join(x))

        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
        count_matrix = count.fit_transform(self.smd['soup'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.smd = self.smd.reset_index()
        self.titles = self.smd['title']
        self.indices = pd.Series(self.smd.index, index=self.smd['title'])
        
    def get_movieId(self, title):
        ids = self.smd[self.smd['title'] == title]['id']
        movieId = 0
        if len(ids) > 0:
            movieId = ids.iloc[0]
        return movieId
    
    def calculate_weighted_score(self, row, weights=(0.6, 0.4, 1)):
        features = row[['majorId', 'schoolYear', 'rating']].values
        weighted_score = np.dot(features, weights)
        return weighted_score
    
    ## Collaborative Filtering (Changed & New)
    def collaborative_filter(self):
        reader = Reader()
        self.surveys['movieId'] = self.surveys['favoriteMovies'].apply(self.get_movieId)
        self.surveys['majorId'] = self.surveys.groupby("major").cumcount() + 1
        self.surveys = self.surveys.drop(columns=['major', 'favoriteMovies'])

        self.ratings['majorId'] = 0
        self.ratings['schoolYear'] = 0
        self.ratings = self.ratings.drop(columns=['timestamp'])
        self.surveys = self.surveys[self.ratings.columns]
        combined_ratings = pd.concat([self.ratings, self.surveys], ignore_index=True)

        scaler = MinMaxScaler()
        q = combined_ratings.apply(self.calculate_weighted_score, axis=1).values.reshape(-1, 1)
        q = scaler.fit_transform(q)
        combined_ratings['score'] = q * 5

        data = Dataset.load_from_df(combined_ratings[['userId', 'movieId', 'score']], reader)
        self.svd = SVD()
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)

    ## Combined Recommender (Changed & New)
    def convert_int(self, x):
        try:
            return int(x)
        except:
            return np.nan
        
    def rec(self, userId, title):
        self.metadata_rec()
        self.collaborative_filter()
        id_map = self.id_map
        id_map['tmdbId'] = id_map['tmdbId'].apply(self.convert_int)
        id_map.columns = ['movieId', 'id']
        id_map = id_map.merge(self.smd[['title', 'id']], on='id').set_index('title')
        indices_map = id_map.set_index('id')
        if title not in self.indices:
            message = "Sorry, but we know little about this movie :("
            return message, None
        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            message = "Sorry, but we know little about this movie :("
            return message, None
        sim_scores = list(enumerate(self.cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = self.smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: self.svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        message = "Here are the top ten recommendations :)"
        return message, movies.head(10)

# print(None)
# path = os.getcwd()
# data_path = path+"/data"
# # data_path = path
# md = pd.read_csv(data_path + "/movies_metadata.csv")
# print(md.shape)
# print("hello")
# links_small = pd.read_csv(data_path + "/links_small.csv")
# creds = pd.read_csv(data_path + "/credits.csv")
# print(creds.shape)
# keywords = pd.read_csv(data_path + "/keywords.csv")
# ratings = pd.read_csv(data_path + "/ratings_small.csv")
# surveys = pd.read_csv(data_path + "/surveys.csv")
#
# Rec = Recommender(md, links_small, creds, keywords, ratings, surveys)
# m, r = Rec.rec(703, 'Saw')
# print(m)
# print(r)
