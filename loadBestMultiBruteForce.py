
# Import libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# Connects Matlab to Pyhton
import transplant
# Imports Reinforcement learning class
from MultiAgentRLBruteForce import agent_class
import time
import timeit

# Starting Matlab
matlab = transplant.Matlab(arguments=['-desktop'])

matlab.addpath('tensegrityObjects')

#Define tspan=display time interval, wall position and wall height for Matlab simulation
tspan=0.05            # monitor refresh
wallPosition=0.76
wallHeight=5.0
delT=0.001             #time step

#Define how much the motors can spool in or out.
deltaSpool=0.001

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)

matlab.createGraph(env)

#BAD 0 8 and 17

#Define some useful variables
lr=0.02                     # Learning rate
H=4                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.99                  # Gamma used to decay the rewards, higher gamma values= future matters more
sessionTime=600           # Maximum number of cycles in each episode
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
RENDER=True

maxCoord=[]

f= open('MultiBruteForceResults/coordinates.txt','r')
coord=[]
position=[]
i=0
baseLine=0.8
for line in f:
    if float(line)>baseLine:
        coord=np.append(coord,float(line))
        position=np.append(position,i)
    i=i+1
f.close()
print("Center of mass Z coordinate of the best nets (CoM > 0.8)")
print(coord)
print(position)
print("Number of the best nets")
print(len(position))

agent={}
for i in range(len(position)):
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(i)+"_"+str(m)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=MOTOR_NUMBER,gamma=gamma,L=L,n=int(position[i]),m=m)

# Load session
for i in range(len(position)):
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(i)+"_"+str(m)].loadSession()

    print('Loaded session '+str(position[i]))

    #Observe initial rest lengths of the strings
    features = matlab.envReset(env,RENDER)

    features=np.reshape(features,(1,MOTOR_NUMBER))

    maxCoord=np.append(maxCoord,0)

    for j in range(sessionTime):

        action=[]

        for m in range(MOTOR_NUMBER):
            action=np.append(action,agent["A_"+str(i)+"_"+str(m)].pickActionHighestProb(features))
        observations= matlab.actionStep(env,action)

        # Assign the new features to the feature variable for the next cycle
        features=observations
        features=np.reshape(features,(1,MOTOR_NUMBER))

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            error_cnt +=1
            print("ERROR count ",error_cnt)
            for m in range(MOTOR_NUMBER):
                agent["A_"+str(i)+"_"+str(m)].cancel_transition()
            break

        if RENDER:
            matlab.updateGraph(env)

        coordinate=matlab.getCenterOfMass(env)
        if coordinate>maxCoord[i]:
            maxCoord[i]=coordinate
    print(maxCoord)

print("Matrix of the coordinates of this run")
print(maxCoord)
print(np.amax(maxCoord))
best=np.argmax(maxCoord)
print("Best performing net")
print(best)
