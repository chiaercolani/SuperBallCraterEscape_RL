
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



agent={}
# Manually add the number of the net that you would like to train
best=3
episode_number=10
max_ep_cycles=600
j_episode=0
totRewardArray=[]

for m in range(MOTOR_NUMBER):
    agent["A_"+str(best)+"_"+str(m)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=MOTOR_NUMBER,gamma=gamma,L=L,n=best,m=m)

# Load session

for m in range(MOTOR_NUMBER):
    agent["A_"+str(best)+"_"+str(m)].loadSession()

print('Loaded session '+str(best))

while j_episode<episode_number:

    #Start Evaluation Timer
    start = timeit.default_timer()
    #Reset environment
    features = matlab.envReset(env,RENDER)
    #Observe initial length of the strings
    features=np.reshape(features,(1,MOTOR_NUMBER))


    for i in range(max_ep_cycles):

        action=[]

        for m in range(MOTOR_NUMBER):
            action=np.append(action,agent["A_"+str(best)+"_"+str(m)].pick_action(features))

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
                agent["A_"+str(best)+"_"+str(tr)].cancel_transition()

            #TODO add negative reward when error occurs
            break

        # Store the transition
        for tr in range(MOTOR_NUMBER):
            agent["A_"+str(best)+"_"+str(tr)].store_transition(features,action[tr],rewards)

        if RENDER:
            matlab.updateGraph(env)
        # If the environment asserts the done signal, collect reward and start a new episode


    if not(env.superBallDynamicsPlot.plotErrorFlag==1):
        #print(agent['A0'].ep_rewards)
        ep_rewards_sum=sum(agent["A_"+str(best)+"_0"].ep_rewards)

        totRewardArray=np.append(totRewardArray,ep_rewards_sum)

        print("episode:", j_episode, " cycle ",i, "  reward:", int(ep_rewards_sum))
        print("Accumulated rewards for MultiAgent: ",totRewardArray)
        discounted_r= agent["A_"+str(best)+"_0"].computeDiscountedRewards()

        for m in range(MOTOR_NUMBER):
            agent["A_"+str(best)+"_"+str(m)].nn_learn(discounted_r)

        error_cnt=0
        j_episode += 1

        #Start rendering if the last 5 rewards are > 0
        #if len(totRewardArray)>5 and totRewardArray[len(totRewardArray)-1]>0 and totRewardArray[len(totRewardArray)-2]>0 and totRewardArray[len(totRewardArray)-3]>0 and totRewardArray[len(totRewardArray)-4]>0 and totRewardArray[len(totRewardArray)-5]>0:
        #    RENDER=True

        stop = timeit.default_timer()

        print ("Iteration time MultiAgent: ",stop - start )
