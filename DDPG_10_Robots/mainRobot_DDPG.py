from DDPG_network import Agent
from robotic_env import RobotEnv
import random
import numpy as np

if __name__ == '__main__':
    LR_A = 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    numberOfRobots = 10
    numberOfAPs = 4
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP

    Robot0 = Agent('actor0', 'critic0', 'actor0_target', 'critic0_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot1 = Agent('actor1', 'critic1', 'actor1_target', 'critic1_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot2 = Agent('actor2', 'critic2', 'actor2_target', 'critic2_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot3 = Agent('actor3', 'critic3', 'actor3_target', 'critic3_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot4 = Agent('actor4', 'critic4', 'actor4_target', 'critic4_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot5 = Agent('actor5', 'critic5', 'actor5_target', 'critic5_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot6 = Agent('actor6', 'critic6', 'actor6_target', 'critic6_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot7 = Agent('actor7', 'critic7', 'actor7_target', 'critic7_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot8 = Agent('actor8', 'critic8', 'actor8_target', 'critic8_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)
    Robot9 = Agent('actor9', 'critic9', 'actor9_target', 'critic9_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=7)

    Robot0_mover = Agent('actor0_mover', 'critic0_mover', 'actor0_mover_target', 'critic0_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot1_mover = Agent('actor1_mover', 'critic1_mover', 'actor1_mover_target', 'critic1_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot2_mover = Agent('actor2_mover', 'critic2_mover', 'actor2_mover_target', 'critic2_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot3_mover = Agent('actor3_mover', 'critic3_mover', 'actor3_mover_target', 'critic3_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot4_mover = Agent('actor4_mover', 'critic4_mover', 'actor4_mover_target', 'critic4_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot5_mover = Agent('actor5_mover', 'critic5_mover', 'actor5_mover_target', 'critic5_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot6_mover = Agent('actor6_mover', 'critic6_mover', 'actor6_mover_target', 'critic6_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot7_mover = Agent('actor7_mover', 'critic7_mover', 'actor7_mover_target', 'critic7_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot8_mover = Agent('actor8_mover', 'critic8_mover', 'actor8_mover_target', 'critic8_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    Robot9_mover = Agent('actor9_mover', 'critic9_mover', 'actor9_mover_target', 'critic9_mover_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=2)
    AP1 = Agent('actor1_AP', 'critic1_AP', 'actor1_AP_target', 'critic1_AP_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=8)
    AP2 = Agent('actor2_AP', 'critic2_AP', 'actor2_AP_target', 'critic2_AP_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=8)
    AP3 = Agent('actor3_AP', 'critic3_AP', 'actor3_AP_target', 'critic3_AP_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=8)
    AP4 = Agent('actor4_AP', 'critic4_AP', 'actor4_AP_target', 'critic4_AP_target', alpha=LR_A, beta=LR_C, input_dims=[122],
                   tau=0.001, n_actions=8)

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
                act0_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act1_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act2_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act3_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act4_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act5_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act6_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act7_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act8_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
                act9_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)  # based on calculation, 10 will have 36 km/h movemenr speed
            else:
                act0_robot_move = Robot0_mover.choose_action(obs_move)
                act1_robot_move = Robot1_mover.choose_action(obs_move)
                act2_robot_move = Robot2_mover.choose_action(obs_move)
                act3_robot_move = Robot3_mover.choose_action(obs_move)
                act4_robot_move = Robot4_mover.choose_action(obs_move)
                act5_robot_move = Robot5_mover.choose_action(obs_move)
                act6_robot_move = Robot6_mover.choose_action(obs_move)
                act7_robot_move = Robot7_mover.choose_action(obs_move)
                act8_robot_move = Robot8_mover.choose_action(obs_move)
                act9_robot_move = Robot9_mover.choose_action(obs_move)
            for j in range(100):
                # while not done:
                epsilon -= 1 / Explore
                a += 1
                if np.random.random() <= epsilon:
                    act0_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act1_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act2_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act3_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act4_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act5_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act6_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act7_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act8_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act9_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act0_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act1_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act2_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act3_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

                else:
                    act0_robot = Robot0.choose_action(obs)
                    act1_robot = Robot1.choose_action(obs)
                    act2_robot = Robot2.choose_action(obs)
                    act3_robot = Robot3.choose_action(obs)
                    act4_robot = Robot4.choose_action(obs)
                    act5_robot = Robot5.choose_action(obs)
                    act6_robot = Robot6.choose_action(obs)
                    act7_robot = Robot7.choose_action(obs)
                    act8_robot = Robot8.choose_action(obs)
                    act9_robot = Robot9.choose_action(obs)
                    act0_AP = AP1.choose_action(obs)
                    act1_AP = AP2.choose_action(obs)
                    act2_AP = AP3.choose_action(obs)
                    act3_AP = AP4.choose_action(obs)
                    # print("AP0  >", act0_AP, "    AP1  >", act1_AP, "    AP2 > ", act2_AP, "   AP3 >", act3_AP)
                act_robots = np.concatenate([act0_robot, act1_robot, act2_robot, act3_robot, act4_robot, act5_robot, act6_robot, act7_robot, act8_robot, act9_robot])
                act_robo_move = np.concatenate([act0_robot_move, act1_robot_move, act2_robot_move, act3_robot_move, act4_robot_move, act5_robot_move, act6_robot_move, act7_robot_move, act8_robot_move, act9_robot_move])
                act_APs = np.concatenate([act0_AP, act1_AP, act2_AP, act3_AP])
                # act_robots = np.concatenate([act0_robot, act1_robot])
                # act_robo_move = np.concatenate([act0_robot_move, act1_robot_move])
                new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy = env.step_task_offloading(act_APs, act_robots, act_robo_move)

                Robot0.remember(obs, act0_robot, reward_Robot, new_state, done)
                Robot1.remember(obs, act1_robot, reward_Robot, new_state, done)
                Robot2.remember(obs, act2_robot, reward_Robot, new_state, done)
                Robot3.remember(obs, act3_robot, reward_Robot, new_state, done)
                Robot4.remember(obs, act4_robot, reward_Robot, new_state, done)
                Robot5.remember(obs, act5_robot, reward_Robot, new_state, done)
                Robot6.remember(obs, act6_robot, reward_Robot, new_state, done)
                Robot7.remember(obs, act7_robot, reward_Robot, new_state, done)
                Robot8.remember(obs, act8_robot, reward_Robot, new_state, done)
                Robot9.remember(obs, act9_robot, reward_Robot, new_state, done)

                AP1.remember(obs, act0_AP, reward_AP, new_state, done)
                AP2.remember(obs, act1_AP, reward_AP, new_state, done)
                AP3.remember(obs, act2_AP, reward_AP, new_state, done)
                AP4.remember(obs, act3_AP, reward_AP, new_state, done)

                Robot0.learn()
                Robot1.learn()
                Robot2.learn()
                Robot3.learn()
                Robot4.learn()
                Robot5.learn()
                Robot6.learn()
                Robot7.learn()
                Robot8.learn()
                Robot9.learn()

                AP1.learn()
                AP2.learn()
                AP3.learn()
                AP4.learn()

                score_Robot += reward_Robot
                score_AP += reward_AP

                print("Robots reward = ", reward_Robot)
                print("APs reward = ", reward_AP)
                if i %10 == 0:
                    with open("04-accept_Corrected_RW_LR_high_Increase_punish.txt", 'a') as reward_APs:
                        reward_APs.write(str(accept) + '\n')
                    with open("04-AoI_LR_Corrected_RW_LR_high_Increase_punish.txt", 'a') as AoI_file:
                        AoI_file.write(str(AoI) + '\n')
                    with open("04-energy_Corrected_RW_LR_high_Increase_punish.txt", 'a') as energy_file:
                        energy_file.write(str(energy) + '\n')                    

                obs = new_state
                obs_move = new_state

            with open("04-reward_robot_Corrected_RW_LR_high_Increase_punish.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("04-reward_AP_Corrected_RW_LR_high_Increase_punish.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            score_Robot = 0
            score_AP = 0

            Robot0_mover.remember(obs, act0_robot_move, score_Robot, new_state, False)
            Robot1_mover.remember(obs, act1_robot_move, score_Robot, new_state, False)
            Robot2_mover.remember(obs, act2_robot_move, score_Robot, new_state, False)
            Robot3_mover.remember(obs, act3_robot_move, score_Robot, new_state, False)
            Robot4_mover.remember(obs, act4_robot_move, score_Robot, new_state, False)
            Robot5_mover.remember(obs, act5_robot_move, score_Robot, new_state, False)
            Robot6_mover.remember(obs, act6_robot_move, score_Robot, new_state, False)
            Robot7_mover.remember(obs, act7_robot_move, score_Robot, new_state, False)
            Robot8_mover.remember(obs, act8_robot_move, score_Robot, new_state, False)
            Robot9_mover.remember(obs, act9_robot_move, score_Robot, new_state, False)

            Robot0_mover.learn()
            Robot1_mover.learn()
            Robot2_mover.learn()
            Robot3_mover.learn()
            Robot4_mover.learn()
            Robot5_mover.learn()
            Robot6_mover.learn()
            Robot7_mover.learn()
            Robot8_mover.learn()
            Robot9_mover.learn()
