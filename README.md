# SuperBallCraterEscape_RL

This project consists of a Matlab simulation environment that simulates the behaviour of SuperBall and of a Python based neural network used to train SuperBall to escape a square chimney. 

# Matlab simulation

The Matlab simulation is based on this https://github.com/chiaercolani/Tensegrity_MATLAB_Objects repository, which was forked from Jeffrey Friesen and Alex Popescu's initial simuation environment. This simulation environment uses the SuperBall dynamic properties that were already modeled and adapts them to an environment where four walls are present.

# Python simulation

The Python simulation uses the Tensorflow library to create a Neural Network used for the Reinforcement Learning process. The system is treated like a multi-agent one, where each motor is an agent. Since the motors are working towards a single goal, the system is fully cooperative and only one reward is given.
