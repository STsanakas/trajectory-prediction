from os import path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import time
import random
import statistics
import pandas
import math
import csv
import random
import logging
from functools import reduce
from operator import add
from tqdm import tqdm
import geopy.distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping


training_percentage=0.67
myverbose=0
mypath='/home/antjpgr/Documents/Ploia June19/December2019/'
dataset = mypath+'Datasets/Demo.csv'
EarlyStopper = EarlyStopping(patience=1, monitor='loss', mode='min')
filename2=mypath+'Results/GeneticAngles/Results.txt'
f= open(filename2,"w+")
best_score=10000000.0


def preprocessing(dataset):
  dataframe = pd.read_csv(dataset, usecols=[0,1], engine='python')    
  dataset_Y = dataframe.values
  dataset_Y = dataset_Y.astype('float32')
  dataset_X = np.roll(dataset_Y, 1, axis=0)  
  dataframe = pd.read_csv(dataset, usecols=[2,3,4], engine='python')    
  Coords = dataframe.values
  Coords = Coords.astype('float32')
  Coords=np.roll(Coords, 1, axis=0)  

  scaler_X = MinMaxScaler(feature_range=(0, 1))
  scaler_X.fit(dataset_X)
  dataset_X = scaler_X.transform(dataset_X)
  scaler_Y = MinMaxScaler(feature_range=(0, 1))
  scaler_Y.fit(dataset_Y)
  dataset_Y = scaler_Y.transform(dataset_Y)

  train_size = int(len(dataset_X) * training_percentage)
  test_size = len(dataset_X) - train_size
  trainX, testX = dataset_X[0:train_size,:], dataset_X[train_size:len(dataset_X),:]
  trainY, testY = dataset_Y[0:train_size,:], dataset_Y[train_size:len(dataset_X),:]
  Coords = Coords[train_size:len(dataset_X),:]
  

  return trainX, trainY, testX, testY, scaler_X, scaler_Y, Coords
  
def compile_model(network, trainX, trainY):
  nb_neurons=[]
  activation=[]
  # Get our network parameters.
  nb_layers = network['nb_layers']
  lstms=network['lstms']
  implementation1=network['implementation1']
  units1=network['units1']
  lstm_activation1=network['lstm_activation1']
  recurrent_activation1=network['recurrent_activation1']
  implementation2=network['implementation2']
  units2=network['units2']
  lstm_activation2=network['lstm_activation2']
  recurrent_activation2=network['recurrent_activation2']
  nb_neurons.append(network['nb_neurons1'])
  nb_neurons.append(network['nb_neurons2'])
  nb_neurons.append(network['nb_neurons3'])
  nb_neurons.append(network['nb_neurons4'])
  nb_neurons.append(network['nb_neurons5'])
  activation.append(network['activation1'])  
  activation.append(network['activation2'])
  activation.append(network['activation3'])
  activation.append(network['activation4'])
  activation.append(network['activation5'])
  optimizer = network['optimizer']
  model = Sequential()
  # Add each layer.
  

  
  if(lstms==1):
	  model.add(LSTM(units1, input_shape=(1, 2), activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1))
	  for i in range(nb_layers):
		  model.add(Dense(nb_neurons[i], activation=activation[i]))
  elif (lstms==2):
	  model.add(LSTM(units1, input_shape=(1, 2), activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1, return_sequences=True))
	  model.add(LSTM(units2, activation=lstm_activation2, recurrent_activation=recurrent_activation2, implementation=implementation2))
	  for i in range(nb_layers):
		  model.add(Dense(nb_neurons[i], activation=activation[i]))
  elif(lstms==0):
	  model.add(Dense(nb_neurons[0], input_shape=(2,), activation=activation[0]))
	  for i in range(nb_layers):
		  if i>0:
			  model.add(Dense(nb_neurons[i], activation=activation[i]))
  model.add(Dense(2))
  
  model.compile(loss='mean_squared_error', optimizer=optimizer)
  
  if(lstms>0):
    trainX=trainX.reshape(len(trainX),1,2)  
  model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=myverbose, callbacks=[EarlyStopper])
  return model, lstms

def evaluate(model, trainX, testX, trainY, testY, scaler_X, scaler_Y, Coords):
  global best_score
  if not isinstance(model.get_layer(index=0), keras.layers.Dense):
    testX=testX.reshape(len(testX),1,2)    
  testPredict = scaler_Y.inverse_transform(model.predict(testX))
  testY = scaler_Y.inverse_transform(testY)
  testScore=0
  predictions=0

  for i in range(len(Coords)-1):
    R = 6378.1 #Radius of the Earth
    brng = testY[i,1]*(math.pi/180) #Bearing is 90 degrees converted to radians.
    d = testY[i,0] #Distance in km
    brng3 = testPredict[i,1]*(math.pi/180) #Bearing is 90 degrees converted to radians.
    d3 = testPredict[i,0] #Distance in km
    lat1 = math.radians(Coords[i,0]) #Current lat point converted to radians
    lon1 = math.radians(Coords[i,1]) #Current long point converted to radians
    lat3 = math.asin( math.sin(lat1)*math.cos(d3/R) + math.cos(lat1)*math.sin(d3/R)*math.cos(brng3))
    lon3 = lon1 + math.atan2(math.sin(brng3)*math.sin(d3/R)*math.cos(lat1), math.cos(d3/R)-math.sin(lat1)*math.sin(lat3))
    true=(Coords[i+1,0],Coords[i+1,1])    
    mypred=(math.degrees(lat3),math.degrees(lon3))   
    testScore += geopy.distance.vincenty(true, mypred).km    
    predictions=predictions+1
  testScore=testScore/predictions  
  if(best_score>testScore):
    best_score=testScore
    model.save(mypath+'Models/best.h5')
  return testScore
  
def train_and_score(network, dataset):
  trainX, trainY, testX, testY, scalerX, scalerY, Coords = preprocessing(dataset)
  model, lstms = compile_model(network, trainX, trainY)
  error = evaluate(model, trainX, testX, trainY, testY, scalerX, scalerY, Coords)
  return error


#from train import train_and_score

class Network():


  def __init__(self, nn_param_choices=None):

	  self.accuracy = 0.
	  self.nn_param_choices = nn_param_choices
	  self.network = {}  # (dic): represents MLP network parameters

  def create_random(self):
	  for key in self.nn_param_choices:
		  self.network[key] = random.choice(self.nn_param_choices[key])

  def create_set(self, network):


	  self.network = network

  def train(self, dataset):

	  if self.accuracy == 0.:
		  self.accuracy = train_and_score(self.network, dataset)
	
  def print_network(self):
    print("Network error: %.5f" % (self.accuracy))
    print("%d LSTM layers" % self.network['lstms'])
    if (self.network['lstms']>0):
    	print("first LSTM layer: ", self.network['units1'], self.network['lstm_activation1'], self.network['recurrent_activation1'], self.network['implementation1'])
    	if (self.network['lstms']>1):	
    		print("second LSTM layer: ", self.network['units2'], self.network['lstm_activation2'], self.network['recurrent_activation2'], self.network['implementation2'])	
    print("First Hidden Layer: %d neurons, with" % self.network['nb_neurons1'], self.network['activation1'], "activation")
    if (self.network['nb_layers']>1):
    	print("Second Hidden Layer: %d neurons, with" % self.network['nb_neurons2'], self.network['activation2'], "activation")
    if (self.network['nb_layers']>2):
    	print("Third Hidden Layer: %d neurons, with" % self.network['nb_neurons3'], self.network['activation3'], "activation")
    if (self.network['nb_layers']>3):
    	print("Fourth Hidden Layer: %d neurons, with" % self.network['nb_neurons4'], self.network['activation4'], "activation")
    if (self.network['nb_layers']>4):
    	print("Fifth Hidden Layer: %d neurons, with" % self.network['nb_neurons5'], self.network['activation5'], "activation")
    print('Optimizer: ', self.network['optimizer'])
    print('-'*80)

  def print_tofile(self, filename):
    mytime=int(time.time() - start_time)/60
    if self.network['lstms']<2:
        self.network['units2']='N/A'
        self.network['implementation2']='N/A'
        self.network['lstm_activation2']='N/A'
        self.network['recurrent_activation2']='N/A'
        if self.network['lstms']<1:
            self.network['units1']='N/A'
            self.network['implementation1']='N/A'
            self.network['lstm_activation1']='N/A'
            self.network['recurrent_activation1']='N/A'
    if self.network['nb_layers']<5:       
        self.network['nb_neurons5']='N/A'
        self.network['activation5']='N/A' 
        if self.network['nb_layers']<4:
            self.network['nb_neurons4']='N/A'
            self.network['activation4']='N/A'               
            if self.network['nb_layers']<3:       
                self.network['nb_neurons3']='N/A'
                self.network['activation3']='N/A' 
                if self.network['nb_layers']<2:
                    self.network['nb_neurons2']='N/A'
                    self.network['activation2']='N/A'               
    print(vessel,',', int(1000*(self.accuracy)),',', self.network['lstms'],',', self.network['units1'],',', self.network['implementation1'],',', self.network['lstm_activation1'],',', self.network['recurrent_activation1'],',', self.network['units2'],',', self.network['implementation2'],',', self.network['lstm_activation2'],',', self.network['recurrent_activation2'],',', self.network['nb_layers'],',', self.network['nb_neurons1'],',', self.network['activation1'],',', self.network['nb_neurons2'],',', self.network['activation2'],',', self.network['nb_neurons3'],',', self.network['activation3'],',', self.network['nb_neurons4'],',', self.network['activation4'],',', self.network['nb_neurons5'],',', self.network['activation5'],',', self.network['optimizer'],',',mytime, file=open(filename,'a'))
    print(vessel,',', int(1000*(self.accuracy)),',', self.network['lstms'],',', self.network['units1'],',', self.network['implementation1'],',', self.network['lstm_activation1'],',', self.network['recurrent_activation1'],',', self.network['units2'],',', self.network['implementation2'],',', self.network['lstm_activation2'],',', self.network['recurrent_activation2'],',', self.network['nb_layers'],',', self.network['nb_neurons1'],',', self.network['activation1'],',', self.network['nb_neurons2'],',', self.network['activation2'],',', self.network['nb_neurons3'],',', self.network['activation3'],',', self.network['nb_neurons4'],',', self.network['activation4'],',', self.network['nb_neurons5'],',', self.network['activation5'],',', self.network['optimizer'],',',mytime, file=open(mypath+'Results/GeneticAngles/'+str(vessel)+'-'+str(sequence_begin_end[vessel][0])+'.txt','a'))



class Optimizer():

  def __init__(self, nn_param_choices, retain=0.4,
			   random_select=0.1, mutate_chance=0.2):

	  self.mutate_chance = mutate_chance
	  self.random_select = random_select
	  self.retain = retain
	  self.nn_param_choices = nn_param_choices

  def create_population(self, count):

	  pop = []
	  for _ in range(0, count):
		  # Create a random network.
		  network = Network(self.nn_param_choices)
		  network.create_random()

		  # Add the network to our population.
		  pop.append(network)

	  return pop

  @staticmethod
  def fitness(network):
	  return network.accuracy

  def grade(self, pop):

	  summed = reduce(add, (self.fitness(network) for network in pop))
	  return summed / float((len(pop)))

  def breed(self, mother, father):

	  children = []
	  for _ in range(2):

		  child = {}

		  # Loop through the parameters and pick params for the kid.
		  for param in self.nn_param_choices:
			  child[param] = random.choice(
				  [mother.network[param], father.network[param]]
			  )

		  # Now create a network object.
		  network = Network(self.nn_param_choices)
		  network.create_set(child)

		  # Randomly mutate some of the children.
		  if self.mutate_chance > random.random():
			  network = self.mutate(network)

		  children.append(network)

	  return children

  def mutate(self, network):

	  # Choose a random key.
	  mutation = random.choice(list(self.nn_param_choices.keys()))

	  # Mutate one of the params.
	  network.network[mutation] = random.choice(self.nn_param_choices[mutation])

	  return network

  def evolve(self, pop):

	  # Get scores for each network.
	  graded = [(self.fitness(network), network) for network in pop]

	  # Sort on the scores.
	  graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]

	  # Get the number we want to keep for the next gen.
	  retain_length = int(len(graded)*self.retain)

	  # The parents are every network we want to keep.
	  parents = graded[:retain_length]

	  # For those we aren't keeping, randomly keep some anyway.
	  for individual in graded[retain_length:]:
		  if self.random_select > random.random():
			  parents.append(individual)

	  # Now find out how many spots we have left to fill.
	  parents_length = len(parents)
	  desired_length = len(pop) - parents_length
	  children = []

	  # Add children, which are bred from two remaining networks.
	  while len(children) < desired_length:

		  # Get a random mom and dad.
		  male = random.randint(0, parents_length-1)
		  female = random.randint(0, parents_length-1)

		  # Assuming they aren't the same network...
		  if male != female:
			  male = parents[male]
			  female = parents[female]

			  # Breed them.
			  babies = self.breed(male, female)

			  # Add the children one at a time.
			  for baby in babies:
				  # Don't grow larger than desired length.
				  if len(children) < desired_length:
					  children.append(baby)

	  parents.extend(children)

	  return parents

#from optimizer import Optimizer


def train_networks(networks, dataset):

  pbar = tqdm(total=len(networks))
  for network in networks:
	  network.train(dataset)
	  pbar.update(1)
  pbar.close()

def get_average_accuracy(networks):

  total_accuracy = 0
  for network in networks:
	  total_accuracy += network.accuracy

  return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset):
  optimizer = Optimizer(nn_param_choices)
  networks = optimizer.create_population(population)

  # Evolve the generation.
  for i in range(generations):
	  print("***Doing generation %d of %d***" %
				   (i + 1, generations))

	  # Train and get accuracy for networks.
	  train_networks(networks, dataset)
	  networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
	  # Get the average accuracy for this generation.
	  average_accuracy = get_average_accuracy(networks)

	  # Print out the average accuracy each generation.
	  print('Best score up until now:', int(1000*best_score))
	  print('-'*80)

	  # Evolve, except on the last iteration.
	  if i != generations - 1:
		  # Do the evolution.
		  networks = optimizer.evolve(networks)

  # Sort our final population.
  networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

  # Print out the top 5 networks.
  print_networks(networks[:1])
  print_networks_file(networks[:1],sequence_begin_end[vessel])
  



def print_networks_file(networks, ID):
  filename=mypath+'Results/GeneticAngles/Results.csv'
  for network in networks:
	  network.print_tofile(filename)


def print_networks(networks):

  print('-'*80)
  for network in networks:
	  network.print_network()
  print('-'*80)

def main():
  generations = 7  # Number of times to evole the population.
  population = 14  # Number of networks in each generation.


  nn_param_choices = {
	'lstms':[0,1,2],
	'implementation1':[1,2],
	'units1':[2,8,16,32,64,128],
	'lstm_activation1':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'recurrent_activation1':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'implementation2':[1,2],
	'units2':[2,8,16,32,64,128],
	'lstm_activation2':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'recurrent_activation2':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'nb_layers': [1, 2, 3, 4, 5],
    'nb_neurons1': [2,8,16,32,64,128],    
	'activation1': ['tanh', 'sigmoid', 'linear', 'relu'],
    'nb_neurons2': [2,8,16,32,64,128],    
    'activation2': ['tanh', 'sigmoid', 'linear', 'relu'],
    'nb_neurons3': [2,8,16,32,64,128],    
    'activation3': ['tanh', 'sigmoid', 'linear', 'relu'],
    'nb_neurons4': [2,8,16,32,64,128],    
    'activation4': ['tanh', 'sigmoid', 'linear', 'relu'],
    'nb_neurons5': [2,8,16,32,64,128],
    'activation5': ['tanh', 'sigmoid', 'linear', 'relu'],
	  'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
					'adadelta', 'adamax', 'nadam'],
  }

  print("***Evolving %d generations with population %d***" %
			   (generations, population))

  generate(generations, population, nn_param_choices, dataset)


start_time = time.time()
main()
