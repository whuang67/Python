# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:03:38 2017

@author: whuang67
"""

### Sent Text
from twilio.rest import Client
# Your Account SID from twilio.com/console
account_sid = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Your Auth Token from twilio.com/console
auth_token  = "your_auth_token"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+15558675309", 
    from_="+15017250604",
    body="Hello from Python!")

print(message.sid)

### Profanity editor
import urllib

def read_text(path):
    quotes = open(path)
    contents_of_file = quotes.read()
    print(contents_of_file)
    quotes.close()
    check_profanity(contents_of_file)

def check_profanity(text_to_check):
    connection = urllib.urlopen("http://www.wdylike.appspot.com/?q="+text_to_check)
    output = connection.read()
    # print(output)
    connection.close()
    if "true" in output:
        print("Profanity Alert!!!")
    elif "false" in output:
        print("This document has no curse words!")
    else:
        print("Could not scan the document properly,")

check_profanity("shit")
read_text("C:/users/whuang67/downloads/movie_quotes.txt")


### Movie Website
import os
os.chdir("C:/users/whuang67/downloads")
import webbrowser
import fresh_tomatoes

class Movie():
    """This class provides a way to store movie related information."""
    
    VALID_RATINGS = ["G", "PG", "PG-13", "R"]
    
    def __init__(self, movie_title, movie_storyline, poster_image, trailer_youtube):
        self.title = movie_title
        self.storyline = movie_storyline
        self.poster_image_url = poster_image
        self.trailer_youtube_url = trailer_youtube
    
    def show_trailer(self):
        webbrowser.open(self.trailer_youtube_url)

# print Movie.__doc__
# print Movie.VALID_RATINGS
# print Movie.__name__
# print Movie.__module__
toy_story = Movie("Toy Story",
                  "A story of a boy and his toys that come to life",
                  "http://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg",
                  "https://www.youtube.com/watch?v=vwyZH85NQC4")
# print(toy_story.storyline)
# toy_story.show_trailer()
avatar = Movie("Avatar",
               "A marine on an alien planet",
               "http://upload.wikimedia.org/wikipedia/id/b/b0/Avatar-Teaser-Poster.jpg",
               "https://www.youtube.com/watch?v=cRdxXPV9GNQ")
# print(avatar.storyline)
# avatar.show_trailer()
school_of_rock = Movie("School of Rock", "Storyline",
                       "http://upload.wikimedia.org/wikipedia/en/1/11/School_of_Rock_Poster.jpg",
                       "https://www.youtube.com/watch?v=3PsUJFEBC74")
ratatouille = Movie("Ratatouille", "Storyline",
                    "http://upload.wikimedia.org/wikipedia/en/5/50/RatatouillePoster.jpg",
                    "https://www.youtube.com/watch?v=c3sBBRxDAqk")
midnight_in_paris = Movie("Midnight in Paris", "Storyline",
                          "http://upload.wikimedia.org/wikipedia/en/9//9f/Midnight_in_Paris_Poster.jpg",
                          "https://www.youtube.com/watch?v=atLg2wQQxvU")
hunger_games = Movie("Hunger Games", "Storyline",
                     "http://upload.wikimedia.org/wikipedia/en/4/42/HungerGamesPoster.jpg",
                     "https://www.youtube.com/watch?v=PbA63a7H0bo")

movies = [toy_story, avatar, school_of_rock, ratatouille, midnight_in_paris, hunger_games]
fresh_tomatoes.open_movies_page(movies)