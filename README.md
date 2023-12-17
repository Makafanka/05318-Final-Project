# 05318-Final-Project: 
A Unique Movie Recommender for CMU Community-Carnegie Movie Universe

## Description
This is a movie recommendation website specially designed for CMU community. Users put in their movie preferences and personal information, and then the system will provide recommended movies and related information. It also offers opportunities for on-campus movie fans meetup.

## To run the app
>streamlit run app.py
Note that the credit.csv is too large for github, so you need to unzip it at first.

## Open-source Code and My Updates
Open-source code: https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/notebook
I built a dynamic input system so that the user could put in their favorite movie, use a slider to rate the movie. And it would be totally optional for the user to put in their personal information. If the user would like to gain a CMU-specialized recommendation, then they could put in their major and school year. I combined the metadata-based recommender (use cosine-similarity to present the similarity between movies) and collaborative filtering (use SVD to present the similarity different users with ratings, majors, school years) in order to get recommendations. I conducted a survey among some CMU students about their movie preferences. I used the survey results to improve the collaborative filtering and make it CMU specific. I also simulated a system that would recommend movie fans meetup for the user based on their favoriate movie and the recommended movies. I also provided background information about how and why I would use the personal information and what the recommended information meant in order to help the user understand this system.
