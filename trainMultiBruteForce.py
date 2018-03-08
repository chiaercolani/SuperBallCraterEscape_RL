
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
#matlab = transplant.Matlab(arguments=['-desktop'])
matlab=transplant.Matlab(arguments=['desktop=True'])

matlab.addpath('tensegrityObjects')

#Define tspan=display time interval, wall position and wall height for Matlab simulation
tspan=0.05            # monitor refresh
wallPosition=0.8
wallHeight=5.0
delT=0.001             #time step

#Define how much the motors can spool in or out.
deltaSpool=0.0005

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)

matlab.createSuperBallGraph(env)


#BAD 0 8 and 17

#Define some useful variables
lr=0.02                     # Learning rate
H=4                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.99                  # Gamma used to decay the rewards, higher gamma values= future matters more
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
RENDER=False

maxCoord=[]

f= open('NNMassimo/Results-6Feb-11pm/coordinates.txt','r')
coord=[]
position=[]
i=0
minHeight=0.88
maxHeight=1
for line in f:
    if float(line)>minHeight and float(line)<maxHeight:
        coord=np.append(coord,float(line))
        position=np.append(position,i)
    i=i+1
f.close()
print("Center of mass Z coordinate of the best nets (CoM >"+str(minHeight))
print(coord)
print(position)
print("Number of the best nets")
print(len(position))

agent={}
# Manually add the number of the net that you would like to train

episode_number=300
max_ep_cycles=600
j_episode=0
totRewardArray=[]

agent={}
for i in range(len(position)):
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(position[i])+"_"+str(m)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=MOTOR_NUMBER,gamma=gamma,L=L,n=int(position[i]),m=m)

# Load session

for pos in range(len(position)):
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(position[pos])+"_"+str(m)].loadSession()

    print('Loaded session '+str(position[pos]))

    #strt training of the session
    while j_episode<episode_number:

        #Start Evaluation Timer
        start = timeit.default_timer()
        #Reset environment
        features = matlab.envReset(env,RENDER)
        #Observe initial length of the strings
        features=np.reshape(features,(1,MOTOR_NUMBER))

        centerMassCoord= []
        maxCoord=np.append(maxCoord,0)

        for i in range(max_ep_cycles):

            action=[]

            for m in range(MOTOR_NUMBER):
                action=np.append(action,agent["A_"+str(position[pos])+"_"+str(m)].pick_action(features))

            # Collect new features rewards and done signal from environment after having performed the action
            # Separated in three functions because transplant could not handle multiple output functions
            observations= matlab.actionStep(env,action)

            #Compute rewards for the cycle
            rewards = matlab.computeRewards(env)

            # Assign the new features to the feature variable for the next cycle
            features=observations
            features=np.reshape(features,(1,MOTOR_NUMBER))

            if env.superBallDynamicsPlot.plotErrorFlag==1:
                error_cnt +=1
                print("ERROR count ",error_cnt)
                # Cancel the records regarding this transition
                for tr in range(MOTOR_NUMBER):
                    agent["A_"+str(position[pos])+"_"+str(tr)].cancel_transition()

                #TODO add negative reward when error occurs
                break

            # Store the transition
            for tr in range(MOTOR_NUMBER):
                agent["A_"+str(position[pos])+"_"+str(tr)].store_transition(features,action[tr],rewards)


            if RENDER:
                matlab.updateGraph(env)
            # If the environment asserts the done signal, collect reward and start a new episode
            coordinate=matlab.getCenterOfMass(env)
            centerMassCoord=np.append(centerMassCoord,coordinate)
            if coordinate>maxCoord[j_episode]:
                maxCoord[j_episode]=coordinate

        if not(env.superBallDynamicsPlot.plotErrorFlag==1):
            #print(agent['A0'].ep_rewards)
            ep_rewards_sum=sum(agent["A_"+str(position[pos])+"_0"].ep_rewards)

            totRewardArray=np.append(totRewardArray,ep_rewards_sum)

            print("episode:", j_episode, " cycle ",i, "  reward:", int(ep_rewards_sum))
            print("Accumulated rewards for MultiAgent: ",totRewardArray)
            discounted_r= agent["A_"+str(position[pos])+"_0"].computeDiscountedRewards()

            for tr in range(MOTOR_NUMBER):
                agent["A_"+str(position[pos])+"_"+str(tr)].nn_learn(discounted_r)

            error_cnt=0
            j_episode += 1

            #Start rendering if the last 5 rewards are > 0
            #if len(totRewardArray)>5 and totRewardArray[len(totRewardArray)-1]>0 and totRewardArray[len(totRewardArray)-2]>0 and totRewardArray[len(totRewardArray)-3]>0 and totRewardArray[len(totRewardArray)-4]>0 and totRewardArray[len(totRewardArray)-5]>0:
            #    RENDER=True

            stop = timeit.default_timer()
            agent["A_"+str(position[pos])+"_0"].saveRunCoordinates(centerMassCoord,j_episode)
            print ("Iteration time MultiAgent: ",stop - start )

    #save each session
    agent["A_"+str(position[pos])+"_0"].saveRewards(totRewardArray)
    agent["A_"+str(position[pos])+"_0"].saveCoordinates(maxCoord)
    totRewardArray=0
    j_episode=0
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(position[pos])+"_"+str(m)].saveSession()
