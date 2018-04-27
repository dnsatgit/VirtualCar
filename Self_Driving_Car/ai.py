# the artificial intelligence's architechture : ANN CLASS

import numpy as np
import random
import os
import torch
import torch.nn as nn# to use nn as torch nn module
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# ANN Network class

class Network(nn.Module):#netowrk inherits from Module
#init: initialize everytime: 
# hiddenlayers, output layers defined    
    def __init__(self, input_size, nb_action):#3 arguments: self-object type, no. of input neron: 5 vectors of encoded values, output neurons
        super(Network, self).__init__()#to use nn.Module's inheritance
        self.input_size = input_size#no. of input neorons
        self.nb_action=nb_action#
        self.fc1 = nn.Linear(input_size, 50)#30 hidden layer:full connection between ionput and hiddern
        self.fc2 = nn.Linear(50, nb_action)#full connection between hidden and input layer
     
    #forward: to activate the neurons rectifier activation fucntion
    def forward(self, state):
        x = F.relu(self.fc1(state))#relu : activation function
        q_values = self.fc2(x)#q values for output neurons
        return q_values# every forward function will  return q values 

#to implement experience replay: to consider past states 
class ReplayMemory(object):
    #to initialize variables for future instances of the class
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []#empty list initially
    #to limit capacity of memory
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]#if size goes over capacity, first object is removed
    #to create data sample of the made size of memory
    def sample(self, batch_size):#get samples from memory
        samples = zip(*random.sample(self.memory, batch_size))#zip to reshape list: one sample each for state, action and reward; so , zip reshapes accordingly
        return map(lambda x: Variable(torch.cat(x, 0)), samples)#samples to pyTorch variables
#concatinate into one dimesional torch variable AND apply lamda to all samples

class Dqn():#deepQ learning model 
    #
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)#model for softmax
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.005)#adam optimizer for good learnign car: 
        self.last_state = torch.Tensor(input_size).unsqueeze(0)#vector of 5 dimesions: 3 sensors, left orientation and right orientation( has to be a torch tensor)
        self.last_action = 0
        self.last_reward = 0
    
    #select function: use of softmax function 
    def select_action(self, state):#distribution of probabnility for 3 q values
        probs = F.softmax(self.model(Variable(state, volatile = True))*80)#volatile to save memory; 100: temp parameter
        action = probs.multinomial()#random draw: from distribution of probs
        return action.data[0,0]# return indeces of action
    
    #to trian forward propagation and backward using socastic gradient decent
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):#batches for markov decision process
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)#output of the batch state and gather to get action that was chosen unsqueeze to have same dimension of state and squueze to kill fake batch to get into simple tensor(vector)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]#detach all net states of the batch; then take max (1) specifies the action on index [0]
        target = self.gamma*next_outputs + batch_reward#to compute target
        td_loss = F.smooth_l1_loss(outputs, target)#td= loss function (hoover loss) with args(output , target)
        self.optimizer.zero_grad()#we must reinitialize optimizer on each loop
        td_loss.backward(retain_variables = True)#backprop to update weights with socastic gradient descent
        self.optimizer.step()#step to update the weight
    
    
    #to update when ai discoveres a new state
    def update(self, reward, new_signal):#last reward that was  acquired on map, and last signal 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)#new state to be updated first that the sensor jsut detected, signal is the state and convert it into a torch tensor
        #to update the memory next: self.memory from dqn, push function to append a new transition to the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))#long is 0 1 or 2; 
        action = self.select_action(new_state)# new transition: select new state as action
        if len(self.memory.memory) > 100:#if the memory batch is filled over 100, it has to learn 
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)# to get the batches : arguments from learn should be = to what sample function returns
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)#learn from 100 transitions; 100 of each
        self.last_action = action#update new action
        self.last_state = new_state#updating state 
        self.last_reward = reward#update new reward
        self.reward_window.append(reward)#update reward window/ append reward
        if len(self.reward_window) > 1000:
            del self.reward_window[0]#limit window to 1000
        return action#return action of update
    
    def score(self):#mean of all of rewards in the reward window
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    #takes the sum of of all the elements inside reward window divide over number of reward window+1 just so it is never equal to zero
    
    #fucntion to save the brain of the car; save all the models
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),#to add the self.model into python dictionary
                    'optimizer' : self.optimizer.state_dict(),#to save optimizer key into dictionary
                   }, 'last_brain.pth')#save as last_brain a.pth in the app folder
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> laoding!! ")
            checkpoint = torch.load('last_brain.pth')#totch library load: file name that contains 
            self.model.load_state_dict(checkpoint['state_dict'])# self.model loaded
            self.optimizer.load_state_dict(checkpoint['optimizer'])# self.optimizer loaded
            print("loaded model and optimizer successfully")#to print message upon successful completion
        else:
            print("directory does not exist!!!")
            #compeltion of ai.py brain=> execution of brain in map