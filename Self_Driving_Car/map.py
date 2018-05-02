#@Objective: To create a map
#imports of central libraries for support for environment
import matplotlib.pyplot as plt
import time
import numpy as np
from random import random, randint

#Use of kivy to create the map, template provided
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

#used ai.py : Deep Q Learning- Dqn class from AI 
from ai import Dqn


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
# starting coordinates for createing barriers
last_x = 0
last_y = 0
n_points = 0
length = 0

#this brain contains ANN : Brain:object-type, dqn:class-type(5 dimesions-states , 3-actions (left,straight, right),0.9-Gamma Lerning parameter)
brain = Dqn(5,3,0.9)

#vector of 3[0,1,2]-actions selected is 0 1 or 2 0 -go straight, 20-go left 20 degree and -20 go right 20 degree
action2rotation = [0,20,-20]

#iet reward
last_reward = 0

#initialize scores vector to store rewards assignments
scores = []

# Initializing the map
first_update = True
def init():
    
    #divider is the lines: divider for road 1 if sand else 0 if no sand
    global divider
    
    # point of map to set destination on map x,y
    global goal_x
    global goal_y
    
    #
    global first_update
    
    #initialize divider as we draw
    divider = np.zeros((longueur,largeur))
    
    #set x,y axis to set goal state
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# 0 reward once we reach goal
last_distance = 0


# car class
class Car(Widget):
    
    angle = NumericProperty(0)
    
    rotation = NumericProperty(0)
    
    #velocity from previous point to new point
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    #to detect divider on left or right
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    
    #density of dividers= signals
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

#to make the car move
    def move(self, rotation):
        #velocity updates position in the direction of velocity vector
        self.pos = Vector(*self.velocity) + self.pos

#0 - straight, 1- left, 2 right
        self.rotation = rotation
       
        #degree of rotation
        self.angle = self.angle + self.rotation
        
        #30 is the distance between the sensor-what the car detects
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
       
        #after sensor, update signals
        self.signal1 = int(np.sum(divider[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(divider[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(divider[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
       
        #to stop sensors from signals into 1(full density of divider) to move away from edges
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        
        
        #update function
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        
        #mean score of reward
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        #position of car updated
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

#if car is onto the divider , slow down the car
        if divider[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
            # and give a -1 reward
        else: # otherwise
            #nomal speed of 6 and reward assigment accordingly
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.05
            if distance < last_distance:
                last_reward = 0.02

#if car is edges of map!!!!
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1#left
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1#right
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1#bottom edge
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1#upper edge

        if distance < 100:#update coordinates of class
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools


#KIVY painting tools - not AI so 
        #Open Sourced
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            divider[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            divider[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent
    
#API buttons : clear save and load
    def clear_canvas(self, obj):
        global divider
        self.painter.canvas.clear()
        divider = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# main method to execute
if __name__ == '__main__':
    CarApp().run()