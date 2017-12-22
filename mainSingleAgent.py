
# Import libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# Connects Matlab to Pyhton
import transplant
# Imports Reinforcement learning class
from RLSingleAgent import agent_class
import time
import timeit

# Starting Matlab
matlab = transplant.Matlab(arguments=['-desktop'])

matlab.addpath('tensegrityObjects')

#Define tspan=display time interval, wall position and wall height for Matlab simulation
tspan=0.01            # monitor refresh
wallPosition=0.76
wallHeight=5.0
delT=0.001             #time step
graphRefresh=1000 #used to refresh every second

#Define how much the motors can spool in or out.
deltaSpool=0.001

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)

matlab.createGraph(env)


#Define some useful variables
lr=0.02                     # Learning rate
H=8                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.99                  # Gamma used to decay the rewards, higher gamma values= future matters more
episode_number=3000         # Number of episodes used to train the NN
max_ep_cycles=300           # Maximum number of cycles in each episode
j_episode=0
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
RENDER=False

totRewardArray=[]

# Create the "brain" of the agent by calling the agent_class
# feature size is 1 because there is only one value considered for each motor
# TODO add new features!!!
agent=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=1,gamma=gamma,L=L)
print('Agents are ready')

while j_episode<episode_number:

    #Start Evaluation Timer
    start = timeit.default_timer()
    #Observe initial rest lengths of the strings
    features = matlab.envReset(env,RENDER)

    features=np.reshape(features,(24,1))

    for i in range(max_ep_cycles):

        action=agent.pick_action(features)
        # Collect new features rewards and done signal from environment after having performed the action
        # Separated in three functions because transplant could not handle multiple output functions
        observations= matlab.actionStep(env,action)
        rewards = matlab.computeRewards(env)


        # Assign the new features to the feature variable for the next cycle
        features=observations

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            error_cnt +=1
            print("ERROR count ",error_cnt)
            agent.cancel_transition()
            break

        # Store the transition
        agent.store_transition(features,action,rewards)

        if RENDER:
            matlab.updateGraph(env)
        # If the environment asserts the done signal, collect reward and start a new episode

    if not(env.superBallDynamicsPlot.plotErrorFlag==1):
        ep_rewards_sum=sum(agent.ep_rewards)
        totRewardArray=np.append(totRewardArray,ep_rewards_sum)

        print("episode:", j_episode, "  episode reward:", int(ep_rewards_sum))
        print("Accumulated rewards for SingleAgent: ",totRewardArray)
        discounted_r= agent.computeDiscountedRewards()
        agent.nn_learn(discounted_r)

        error_cnt=0
        j_episode += 1

        #Start rendering if the last 5 rewards are > 0
        #if len(totRewardArray)>5 and totRewardArray[len(totRewardArray)-1]>0 and totRewardArray[len(totRewardArray)-2]>0 and totRewardArray[len(totRewardArray)-3]>0 and totRewardArray[len(totRewardArray)-4]>0 and totRewardArray[len(totRewardArray)-5]>0:
        #    RENDER=True
        #else:
        #    RENDER=False

        stop = timeit.default_timer()

        print ("Iteration time SingleAgent: ",stop - start )
