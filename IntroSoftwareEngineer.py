# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:37:14 2017

@author: whuang67
"""

import time
import webbrowser
num = 1
while num <= 3:
    time.sleep(2)
    webbrowser.open("http://www.google.com")
    num += 1
    print("time is "+time.ctime())


import os
def rename_files(path):
    # (1) Get file names from a folder
    file_list = os.listdir(path)
    # (2) for each file, rename filename
    os.chdir(path)
    for file_name in file_list:
        os.rename(file_name,
                  file_name.translate(str.maketrans(dict.fromkeys("0123456789"))))
rename_files(path = "C:/users/whuang67/downloads/prank/prank")

### First Attempt
import turtle
def draw_square():
    window = turtle.Screen()
    window.bgcolor("red")
    
    brad = turtle.Turtle()
    # brad.shape("turtle")
    # brad.color("yellow")
    # brad.speed(2)
    
    brad.forward(100)
    brad.right(90)
    brad.forward(100)
    brad.right(90)
    brad.forward(100)
    brad.right(90)
    brad.forward(100)
    brad.right(90)
    
    angie = turtle.Turtle()
    angie.shape("arrow")
    angie.color("blue")
    angie.circle(100)
    # window.exitonclick()
draw_square()

### Second Attempt
def draw_square2(some_turtle):
    for i in range(4):
        some_turtle.forward(100)
        some_turtle.right(90)
def draw_art():
    window = turtle.Screen()
    window.bgcolor("red")
    # Create a turtle Brad - Draw a square
    brad = turtle.Turtle()
    brad.shape("turtle")
    brad.color("yellow")
    brad.speed(2)
    draw_square2(brad)
    # Create a turtle Angie - Draw a circle
    angie = turtle.Turtle()
    angie.shape("arrow")
    angie.color("blue")
    angie.circle(100)
    # window.exitonclick()
draw_art()

### Third Attempt
def draw_art2(degrees = 10):
    window = turtle.Screen()
    window.bgcolor("red")
    # Create a turtle Brad - Draw a square
    brad = turtle.Turtle()
    brad.shape("turtle")
    brad.color("yellow")
    brad.speed(2)
    for i in range(360//degrees):
        draw_square2(brad)
        brad.right(10)
draw_art2()
