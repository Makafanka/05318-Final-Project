import streamlit as st
import pandas as pd
import os
import numpy as np
from pathlib import Path
from recommender import Recommender
from PIL import Image

@st.cache_data
def read_data(path):
    """Load a csv object"""
    with open(path, "rb") as file:
        obj = pd.read_csv(file)
    return obj

def read_in_data(movie_path):
    md = read_data(os.path.join(movie_path, "movies_metadata.csv"))
    links_small = read_data(os.path.join(movie_path, "links_small.csv"))
    creds = read_data(os.path.join(movie_path, "credits.csv"))
    keywords = read_data(os.path.join(movie_path, "keywords.csv"))
    ratings = read_data(os.path.join(movie_path, "ratings_small.csv"))
    surveys = read_data(os.path.join(movie_path, "surveys.csv"))
    meetups = read_data(os.path.join(movie_path, "Movie_MeetUp.csv"))

    data = {"movies_metadata": md,
            "links": links_small,
            "credits": creds,
            "keywords": keywords,
            "ratings": ratings,
            "surveys": surveys,
            "meetups": meetups}
    return data

def get_input(data_path):
    """Collect the user information, including major, school year, favorite movies, and ratings of these movies"""
    with st.form("my_form"):
        # col1_1, col1_2 = st.columns(2)
        # with col1_1:
        st.header("Required Information: Your favorite movie")

        st.subheader("Favorite Movie Name")
        favoriteMovie1 = st.text_input("Favorite Movie Name", value="Put the movie name here")

        st.subheader("Rating for Favorite Movie")
        rating1 = st.slider("Rating:", 0.0, 5.0, 0.5, 0.5)

        st.write("\n")

        # st.subheader("Favorite Movie 2 (Optional)")
        # favoriteMovie2 = st.text_input("Favorite Movie 2", value="Put the movie name here")
        #
        # st.subheader("Rating for Movie 2")
        # rating2 = st.slider("Rating 2:", 0.0, 5.0, 0.5, 0.5)
        #
        # st.write("\n")
        #
        # st.subheader("Favorite Movie 3 (Optional)")
        # favoriteMovie3 = st.text_input("Favorite Movie 3", value="Put the movie name here")
        #
        # st.subheader("Rating for Movie 3")
        # rating3 = st.slider("Rating 3:", 0.0, 5.0, 0.5, 0.5)


        # with col1_2:
        st.header("Optional Information: Personal Info")
        st.subheader("(To specialize the recommendation for CMU)")
        checkbox_result = st.checkbox("Please check  this box if you allow this website to use your personal info")
        st.write("Q: How are we going to use the data?")
        st.write("A: We are going to compare the information with your peer CMU students and make recommendations "
                 "based on their preferences. You are going to know what movies are in trend at CMU!")
        st.write("We promise that the information will only be used for movie recommendations and will be kept "
                 "confidential.")

        st.subheader("Major")
        major = st.text_input("Major", value="Put your major here")
        st.write(
            "Please use the abbreviation or concise name (e.g. CS, ECE, Math, Stats&ML, CEE, IS, AI, Physics, "
            "Psychology, BHA, Business, etc.)")

        year_list = ["Freshman", "Sophomore", "Junior", "Senior", "Master", "PhD", "Other"]
        st.subheader("School Year")
        year = st.selectbox("School Year", year_list, index=0)
        years = {"Freshman":1, "Sophomore":2, "Junior":3, "Senior":4, "Master":5, "PhD":6, "Other":0}
        schoolYear = years[year]

        submitted = st.form_submit_button("Submit")

        input_dict = {"movie1": favoriteMovie1,
                      "rating1": rating1,
                      # "movie2": favoriteMovie2,
                      # "rating2": rating2,
                      # "movie3": favoriteMovie3,
                      # "rating3": rating3,
                      "major": major,
                      "schoolYear": schoolYear}

    return submitted, checkbox_result, input_dict

def get_meetup(meetup):
    st.write("Meetup Theme: " + str(meetup['activity']))
    st.write("Time: " + meetup['Time'])
    st.write("Location: " + str(meetup['Location']))

def get_rec(movie_data, input_dict):
    """ Get the recommendations and display them on the streamlit application"""
    st.markdown("""---""")
    st.header("Recommended Movies:")
    md = movie_data["movies_metadata"]
    links_small = movie_data["links"]
    creds = movie_data["credits"]
    keywords = movie_data["keywords"]
    ratings = movie_data["ratings"]
    surveys = movie_data["surveys"]
    meetups = movie_data["meetups"]
    rec_class = Recommender(md, links_small, creds, keywords, ratings, surveys)
    user_id = surveys['userId'].iloc[-1]
    print(user_id)
    favorite_movie1 = input_dict["movie1"]
    meetup_list = {"The Avengers":1, "Blade Runner":2, "Star Wars":3, "Titanic":4, "Spirited Away":5, "Leon: The Professional":6, "The Shining":7, "Zootopia":8}
    message, result = rec_class.rec(user_id, favorite_movie1)
    st.write(message)
    movies = []
    if result is not None:
        st.dataframe(result[['title', 'vote_average', 'year']])
        st.write("Meanings: title-movie name, vote_average-movie's averaged rating, year-movie's released year")
        movies = list(result['title'])
    st.subheader("Upcoming Movie Fans Meetup")

    flag = False
    if favorite_movie1 in meetup_list:
        flag = True
        get_meetup(meetups.iloc[meetup_list[favorite_movie1]-1])
    if len(movies) > 0:
        for movie in movies:
            if movie in meetup_list:
                flag = True
                get_meetup(meetups.iloc[meetup_list[movie]-1])
    if flag:
        st.write("Please feel free to join!")
    else:
        st.write("Sorry, there's no upcoming meetup :(")


def data_process(input_dict, movie_data):
    surveys = movie_data["surveys"]
    userId = surveys['userId'].iloc[-1]+1
    major = input_dict["major"]
    schoolYear = input_dict["schoolYear"]
    movie1 = input_dict["movie1"]
    rating1 = input_dict["rating1"]
    new_row_data = {'userId': userId, 'major': major, 'schoolYear': schoolYear, 'favoriteMovies': movie1, 'rating': rating1}
    surveys.loc[len(surveys)] = new_row_data
    movie_data["surveys"] = surveys
    return movie_data

def recommender_app():
    st.title("Carnegie Movie Universe - Movie Recommender for CMU community")
    # Set path to current directory:
    path = os.path.dirname(__file__)

    # Create and gather Parameters
    data_path = os.path.join(path, "data")

    movie_data = read_in_data(data_path)

    submitted, checkbox_result, input_dict = get_input(data_path)

    if submitted:
        st.write('Information has been collected. Generating Recommendations...')
        if checkbox_result:
            st.write("Your recommendations will be CMU specialized!")
            # Process data from parameters
            movie_data = data_process(input_dict, movie_data)

        # Build Recommendations
        get_rec(movie_data, input_dict)


# Main Interation function
def main():
    # Set path to current directory:
    path = os.path.dirname(__file__)

    # Set theme PresentationTools
    logo = Image.open(os.path.join(path, "cmu.jpg"))
    st.set_page_config(page_title='05318 Final Project - Yifan Sun', page_icon=logo, layout='wide')

    recommender_app()


if __name__ == '__main__':
    main()