
# Import libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# Connects Matlab to Pyhton
import transplant
# Imports Reinforcement learning class
from RLBruteForce import agent_class
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

#Define how much the motors can spool in or out in one tspan.
deltaSpool=0.001

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)

matlab.createGraph(env)


#Define some useful variables
lr=0.02                     # Learning rate
H=48                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.5                  # Gamma used to decay the rewards, higher gamma values= future matters more
sessionTime=600           # Maximum number of cycles in each episode
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
RENDER=False
# Number of brute force approach cases
BRUTE_FORCE_CASES=500

maxCoord=[]


# Create 1000 neural networks with random weight initialization and biases initialized to zero
# Neural netwrks are not trained, but saved as is and then loaded. The heigth of the center of mass is
# used as an estimator of how well each network is doing.
# This brute force approach is used to figure out which set of parameters gives the
# best result.

# Generate one agent per motor
agent={}
for i in range(BRUTE_FORCE_CASES):
    agent=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=1,gamma=gamma,L=L,n=i)
    agent.saveSession(i)
print('Neural networks are ready')


print('Saved all the neural networks')
# Load each network and run it for the same time.
# Save the estimator of the netowork performance (height of center of mass)

for i in range(BRUTE_FORCE_CASES):
    # Load session
    agent.loadSession(i)
    print('Loaded session '+str(i))

    #Start Evaluation Timer
    start = timeit.default_timer()
    #Observe initial rest lengths of the strings
    features = matlab.envReset(env,RENDER)

    features=np.reshape(features,(24,1))

    #Initialize center of mass coordinates to 0
    centerMassCoord= []
    maxCoord=np.append(maxCoord,0)

    for j in range(sessionTime):

        action=agent.pick_action(features)
        observations= matlab.actionStep(env,action)

        # Assign the new features to the feature variable for the next cycle
        features=observations

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            error_cnt +=1
            print("ERROR count ",error_cnt)
            agent.cancel_transition()
            break

        if RENDER:
            matlab.updateGraph(env)

        coordinate=matlab.getCenterOfMass(env)
        centerMassCoord=np.append(centerMassCoord,coordinate)
        if coordinate>maxCoord[i]:
            maxCoord[i]=coordinate


    stop = timeit.default_timer()

    print ("Iteration time : ",stop - start )


print(maxCoord)
agent.saveCoordinates(maxCoord)
print(np.amax(maxCoord))
best=np.argmax(maxCoord)
print(best)

#Show the best neural networks
RENDER=True

# Load Best session
agent.loadSession(best)
print('Loaded session '+str(best))


#Observe initial rest lengths of the strings
features = matlab.envReset(env,RENDER)

features=np.reshape(features,(24,1))


for j in range(sessionTime):

    action=agent.pick_action(features)
    observations= matlab.actionStep(env,action)

    # Assign the new features to the feature variable for the next cycle
    features=observations

    if env.superBallDynamicsPlot.plotErrorFlag==1:
        error_cnt +=1
        print("ERROR count ",error_cnt)
        agent.cancel_transition()
        break

    if RENDER:
        matlab.updateGraph(env)


#Plot results
x=np.arange(0,len(maxCoord))
y1= maxCoord

fig= plt.figure(figsize=(20,10))
ax1=plt.subplot(211)
plt.scatter(x,y1)

plt.title('Maximum Z coordinate of the center of mass for each Neural Network run')

plt.show()
