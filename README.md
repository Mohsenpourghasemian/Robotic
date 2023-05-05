# Robotic_Task_Offloading_Paper
Based on Conference Paper: Cooperative task offloading and path planning for industrial robotic using spectral and energy efficient
federated learning

# Requirements
-Python >= 3.7 (tested with python 3.7)\\
-Tensorflow = 1.15\\
-Keras = 2.0.4\\
-Numpy\\

# How to run
In each directory, for example "Centralized", there is a "mainRobotic_DDPG.py" file, where you can modify the simulation parameters and/or also change the number of robots and access points. THe default numbers of robots and access points are 10 and 4, respectively. Please note that if you increase the number of agents (robots and access points), then you need to increase the input state of each agent. Don't worry, if you forget to do so and run, the program tells you hou much you should increase the nmber of state.
-Simply run "mainRobot_DDPG.py".
-Multiple .txt files are created (such as reward robots and reward access point). Then, you can simply plot them using excel, matplotlib, or matlab.

# Please note
I also put Decentralized MADDPG for the interested individuals. It is not in the paper. :D.


Feel free to go to "robotic_env" and change the reward function and see exciting performance.

I would be happy to report any bug for this code.
# One result

![](https://github.com/Mohsenpourghasemian/Robotic/blob/main/Reward_smooth_all.png)

