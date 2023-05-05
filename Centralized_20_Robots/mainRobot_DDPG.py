from DDPG_network import Agent
from robotic_env import RobotEnv
import random
import numpy as np

if __name__ == '__main__':
    LR_A = 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    numberOfRobots = 20
    numberOfAPs = 4
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP

    Robot0 = Agent('actor0', 'critic0', 'actor0_target', 'critic0_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=numberOfRobots * 8)
  

    Robot0_mover = Agent('actor0_mover', 'critic0_mover', 'actor0_mover_target', 'critic0_mover_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=numberOfRobots * 2)


    AP1 = Agent('actor1_AP', 'critic1_AP', 'actor1_AP_target', 'critic1_AP_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=numberOfAPs * 8)


    score_Robot = 0
    score_AP = 0
    score1_history = []
    score0_history = []
    delay_history = []
    energy_history = []
    reject_history = []
    steps = 100
    Explore = 100000.
    epsilon = 1
    epsilon_move = 1
    i = 0
    warmed = 0
    new_state = 0
    for i in range(1000):
        obs = env.reset()
        obs_move = obs
        a = 0
        test = 0  # this is for debugging, and when it is '1', we can choose action to debug
        for move in range(1000):
            epsilon_move -= 1 / Explore
            if np.random.random() < epsilon_move:
                act0_robot_move = np.random.uniform(-1.0, 1.0, 2 * numberOfRobots)
           

            else:
                act0_robot_move = Robot0_mover.choose_action(obs_move)
              

            for j in range(100):
                # while not done:
                epsilon -= 1 / Explore
                a += 1
                if np.random.random() <= epsilon:
                    act0_robot = np.random.uniform(-1.0, 1.0, 8 * numberOfRobots)
                    act0_AP = np.random.uniform(-1.0, 1.0, 8 * numberOfAPs)
                else:
                    act0_robot = Robot0.choose_action(obs)
                 
                    act0_AP = AP1.choose_action(obs)

                new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy, posX0, posY0, posX1, posY1 = env.step_task_offloading(act0_AP, act0_robot, act0_robot_move)

                Robot0.remember(obs, act0_robot, reward_Robot, new_state, done)
            
                AP1.remember(obs, act0_AP, reward_AP, new_state, done)

                Robot0.learn()
               
                AP1.learn()

                score_Robot += np.average(reward_Robot)
                score_AP += np.average(reward_AP)
                print("Robots reward = ", score_Robot)
                print("APs reward = ", score_AP)
                if i %10 == 0:
                    with open("01-accept_LR_vhigh_big_size.txt", 'a') as reward_APs:
                        reward_APs.write(str(accept) + '\n')
                    with open("01-AoI_LR_vhigh_big_size.txt", 'a') as AoI_file:
                        AoI_file.write(str(AoI) + '\n')
                    with open("01-energy_LR_vhigh_big_size.txt", 'a') as energy_file:
                        energy_file.write(str(energy) + '\n')
                obs = new_state
                obs_move = new_state

            with open("01-reward_robot_LR_vhigh_big_size.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("01-reward_AP_LR_vhigh_big_size.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            with open("01-posLR_vhigh_big_size.txt", 'a') as pos0:
                pos0.write(str(posX0) + ', ' + str(posY0) + '\n')
            with open("01-posLR_vhigh_big_size.txt", 'a') as pos1:
                pos1.write(str(posX1) + ', ' + str(posY1) + '\n')                 

            Robot0_mover.remember(obs, act0_robot_move, score_Robot, new_state, False)

            Robot0_mover.learn()

            score_Robot = 0
            score_AP = 0        