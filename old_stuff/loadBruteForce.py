
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

#Define how much the motors can spool in or out.
deltaSpool=0.001

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)

matlab.createGraph(env)


#Define some useful variables
lr=0.02                     # Learning rate
H=8                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.5                  # Gamma used to decay the rewards, higher gamma values= future matters more
sessionTime=600           # Maximum number of cycles in each episode
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
RENDER=False
BRUTE_FORCE_CASES=200
maxCoord=[]

best=3
#agent=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=1,gamma=gamma,L=L,n=0)
agent={}
for i in range(BRUTE_FORCE_CASES):
    # Load session
    print(i)
    agent['A'+str(i)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=1,gamma=gamma,L=L,n=i)

    agent['A'+str(i)].loadSession(i)
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

        action=agent['A'+str(i)].pick_action(features)
        observations= matlab.actionStep(env,action)

        # Assign the new features to the feature variable for the next cycle
        features=observations

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            error_cnt +=1
            print("ERROR count ",error_cnt)
            agent['A'+str(i)].cancel_transition()
            break

        if RENDER:
            matlab.updateGraph(env)

        coordinate=matlab.getCenterOfMass(env)
        centerMassCoord=np.append(centerMassCoord,coordinate)
        if coordinate>maxCoord[i]:
            maxCoord[i]=coordinate


    stop = timeit.default_timer()
    print(maxCoord)
    print ("Iteration time : ",stop - start )
    agent['A'+str(i)].sessionClose()


print(maxCoord)
agent['A'+str(i)].saveCoordinates(maxCoord)
print(np.amax(maxCoord))
best=np.argmax(maxCoord)
print(best)




#Plot results
x=np.arange(0,len(maxCoord))
y1= maxCoord

fig= plt.figure(figsize=(20,10))
ax1=plt.subplot(211)
plt.scatter(x,y1)

plt.title('Maximum Z coordinate of the center of mass for each Neural Network run')

plt.show()
