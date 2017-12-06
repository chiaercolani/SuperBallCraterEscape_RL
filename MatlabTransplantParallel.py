
# Import libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# Connects Matlab to Pyhton
import transplant
# Imports Reinforcement learning class
from RL_agentParallel import agent_class
import time

# Starting Matlab
matlab = transplant.Matlab(arguments=['-desktop'])

matlab.addpath('tensegrityObjects')

#Define temporal step, wall position and wall height for MAtlab simulation
tspan=0.1
wallPosition=0.8
wallHeight=5.0

#Define how much the motors can spool in or out.
deltaSpool=0.004

env=matlab.myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool)

matlab.createGraph(env)


#Define some useful variables
lr=0.02                     # Learning rate
H=8                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.99                  # Gamma used to decay the rewards
episode_number=3000         # Number of episodes used to train the NN
max_ep_cycles=200           # Maximum number of cycles in each episode
j_episode=0
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24

# Create the "brain" of the agent by calling the agent_class
# feature size is 1 because there is only one value considered for each motor
# TODO add new features!!!
#Generate one agent per motor
agent={}
for i in range(MOTOR_NUMBER):
    agent['A'+str(i)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=MOTOR_NUMBER,gamma=gamma,L=L,n=i)
print('Agents are ready')

while j_episode<episode_number:

    #Observe initial rest lengths of the strings
    features = matlab.envReset(env)
    features=np.reshape(features,(1,MOTOR_NUMBER))


    for i in range(max_ep_cycles):

        action=[]

        for m in range(MOTOR_NUMBER):
            action=np.append(action,agent['A'+str(m)].pick_action(features))

        # Collect new features rewards and done signal from environment after having performed the action
        # Separated in three functions because transplant could not handle multiple output functions
        observations= matlab.actionStep(env,action)

        rewards = matlab.computeRewards(env)
        #done= matlab.computeDone(env)

        # Assign the new features to the feature variable for the next cycle
        features=observations
        features=np.reshape(features,(1,MOTOR_NUMBER))

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            print("transition caused an error, I won't count it")
            error_cnt +=1
            print("error count ",error_cnt)
            # Cancel the records regarding this transition
            for tr in range(MOTOR_NUMBER):
                agent['A'+str(tr)].cancel_transition()

            #TODO add negative reward when error occurs
            break

        # Store the transition
        for tr in range(MOTOR_NUMBER):
            agent['A'+str(tr)].store_transition(features,action[tr],rewards)

        # If the environment asserts the done signal, collect reward and start a new episode
        #Done if 300 iterations are completed

    if not(env.superBallDynamicsPlot.plotErrorFlag==1):
        print(agent['A0'].ep_rewards)
        ep_rewards_sum=sum(agent['A0'].ep_rewards)


        print("episode:", j_episode, " cycle ",i, "  reward:", int(ep_rewards_sum))

        discounted_r= agent['A0'].computeDiscountedRewards()

        for i in range(MOTOR_NUMBER):
            agent['A'+str(i)].nn_learn(discounted_r)

        error_cnt=0
        j_episode += 1
