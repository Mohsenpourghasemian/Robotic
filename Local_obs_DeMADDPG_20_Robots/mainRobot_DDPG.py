from DDPG_network import Agent, Federated_Server, Federated_Server_AP
from robotic_env import RobotEnv
import random
import numpy as np

if __name__ == '__main__':
    LR_A = 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    numberOfRobots = 20
    numberOfAPs = 4
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP

    Robot0 = Agent('actor0', 'critic0', 'actor0_target', 'critic0_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot1 = Agent('actor1', 'critic1', 'actor1_target', 'critic1_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot2 = Agent('actor2', 'critic2', 'actor2_target', 'critic2_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot3 = Agent('actor3', 'critic3', 'actor3_target', 'critic3_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot4 = Agent('actor4', 'critic4', 'actor4_target', 'critic4_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot5 = Agent('actor5', 'critic5', 'actor5_target', 'critic5_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot6 = Agent('actor6', 'critic6', 'actor6_target', 'critic6_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot7 = Agent('actor7', 'critic7', 'actor7_target', 'critic7_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot8 = Agent('actor8', 'critic8', 'actor8_target', 'critic8_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot9 = Agent('actor9', 'critic9', 'actor9_target', 'critic9_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot10 = Agent('actor10', 'critic10', 'actor10_target', 'critic10_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot11 = Agent('actor11', 'critic11', 'actor11_target', 'critic11_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot12 = Agent('actor12', 'critic12', 'actor12_target', 'critic12_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot13 = Agent('actor13', 'critic13', 'actor13_target', 'critic13_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot14 = Agent('actor14', 'critic14', 'actor14_target', 'critic14_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot15 = Agent('actor15', 'critic15', 'actor15_target', 'critic15_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot16 = Agent('actor16', 'critic16', 'actor16_target', 'critic16_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot17 = Agent('actor17', 'critic17', 'actor17_target', 'critic17_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot18 = Agent('actor18', 'critic18', 'actor18_target', 'critic18_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)
    Robot19 = Agent('actor19', 'critic19', 'actor19_target', 'critic19_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=8)

    Robot0_mover = Agent('actor0_mover', 'critic0_mover', 'actor0_mover_target', 'critic0_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot1_mover = Agent('actor1_mover', 'critic1_mover', 'actor1_mover_target', 'critic1_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot2_mover = Agent('actor2_mover', 'critic2_mover', 'actor2_mover_target', 'critic2_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot3_mover = Agent('actor3_mover', 'critic3_mover', 'actor3_mover_target', 'critic3_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot4_mover = Agent('actor4_mover', 'critic4_mover', 'actor4_mover_target', 'critic4_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot5_mover = Agent('actor5_mover', 'critic5_mover', 'actor5_mover_target', 'critic5_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot6_mover = Agent('actor6_mover', 'critic6_mover', 'actor6_mover_target', 'critic6_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot7_mover = Agent('actor7_mover', 'critic7_mover', 'actor7_mover_target', 'critic7_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot8_mover = Agent('actor8_mover', 'critic8_mover', 'actor8_mover_target', 'critic8_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot9_mover = Agent('actor9_mover', 'critic9_mover', 'actor9_mover_target', 'critic9_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot10_mover = Agent('actor10_mover', 'critic10_mover', 'actor10_mover_target', 'critic10_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot11_mover = Agent('actor11_mover', 'critic11_mover', 'actor11_mover_target', 'critic11_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot12_mover = Agent('actor12_mover', 'critic12_mover', 'actor12_mover_target', 'critic12_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot13_mover = Agent('actor13_mover', 'critic13_mover', 'actor13_mover_target', 'critic13_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot14_mover = Agent('actor14_mover', 'critic14_mover', 'actor14_mover_target', 'critic14_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot15_mover = Agent('actor15_mover', 'critic15_mover', 'actor15_mover_target', 'critic15_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot16_mover = Agent('actor16_mover', 'critic16_mover', 'actor16_mover_target', 'critic16_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot17_mover = Agent('actor17_mover', 'critic17_mover', 'actor17_mover_target', 'critic17_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot18_mover = Agent('actor18_mover', 'critic18_mover', 'actor18_mover_target', 'critic18_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
    Robot19_mover = Agent('actor19_mover', 'critic19_mover', 'actor19_mover_target', 'critic19_mover_target', alpha=LR_A, beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)    

    AP1 = Agent('actor1_AP', 'critic1_AP', 'actor1_AP_target', 'critic1_AP_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=8)
    AP2 = Agent('actor2_AP', 'critic2_AP', 'actor2_AP_target', 'critic2_AP_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=8)
    AP3 = Agent('actor3_AP', 'critic3_AP', 'actor3_AP_target', 'critic3_AP_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=8)
    AP4 = Agent('actor4_AP', 'critic4_AP', 'actor4_AP_target', 'critic4_AP_target', alpha=LR_A, beta=LR_C, input_dims=[357], tau=0.001, n_actions=8)

    Server_robot = Federated_Server(name_actor='server_actor', name_critic='server_critic', input_dims=[148], n_actions=8, layer1_size=32, layer2_size=32)
    Server_AP = Federated_Server_AP(name_actor='server_actor_AP', name_critic='server_critic_AP', input_dims=[357], n_actions=8, layer1_size=32, layer2_size=32)

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
            Local_Observaion = []
            AP_observation = obs[numberOfRobots]  # the last observation is for access points
            for R in range(numberOfRobots):
                Local_Observaion.append(obs[R])            
            epsilon_move -= 1 / Explore
            if np.random.random() < epsilon_move:
                act0_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act1_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act2_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act3_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act4_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act5_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act6_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act7_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act8_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act9_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act10_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act11_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act12_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act13_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act14_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act15_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act16_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act17_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act18_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act19_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)                

            else:
                act0_robot_move = Robot0_mover.choose_action(Local_Observaion[0])
                act1_robot_move = Robot1_mover.choose_action(Local_Observaion[1])
                act2_robot_move = Robot2_mover.choose_action(Local_Observaion[2])
                act3_robot_move = Robot3_mover.choose_action(Local_Observaion[3])
                act4_robot_move = Robot4_mover.choose_action(Local_Observaion[4])
                act5_robot_move = Robot5_mover.choose_action(Local_Observaion[5])
                act6_robot_move = Robot6_mover.choose_action(Local_Observaion[6])
                act7_robot_move = Robot7_mover.choose_action(Local_Observaion[7])
                act8_robot_move = Robot8_mover.choose_action(Local_Observaion[8])
                act9_robot_move = Robot9_mover.choose_action(Local_Observaion[9])
                act10_robot_move = Robot10_mover.choose_action(Local_Observaion[10])
                act11_robot_move = Robot11_mover.choose_action(Local_Observaion[11])
                act12_robot_move = Robot12_mover.choose_action(Local_Observaion[12])
                act13_robot_move = Robot13_mover.choose_action(Local_Observaion[13])
                act14_robot_move = Robot14_mover.choose_action(Local_Observaion[14])
                act15_robot_move = Robot15_mover.choose_action(Local_Observaion[15])
                act16_robot_move = Robot16_mover.choose_action(Local_Observaion[16])
                act17_robot_move = Robot17_mover.choose_action(Local_Observaion[17])
                act18_robot_move = Robot18_mover.choose_action(Local_Observaion[18])
                act19_robot_move = Robot19_mover.choose_action(Local_Observaion[19])                

            for j in range(100):
                # while not done:
                Local_Observaion = []
                AP_observation = obs[numberOfRobots]  # the last observation is for access points
                for R in range(numberOfRobots):
                    Local_Observaion.append(obs[R])   
                epsilon -= 1 / Explore
                a += 1
                if np.random.random() <= epsilon:
                    act0_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act1_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act2_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act3_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act4_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act5_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act6_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act7_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act8_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act9_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act10_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act11_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act12_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act13_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act14_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act15_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act16_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act17_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act18_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act19_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)                        
                    act0_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act1_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act2_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act3_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                else:
                    act0_robot = Robot0.choose_action(Local_Observaion[0])
                    act1_robot = Robot1.choose_action(Local_Observaion[1])
                    act2_robot = Robot2.choose_action(Local_Observaion[2])
                    act3_robot = Robot3.choose_action(Local_Observaion[3])
                    act4_robot = Robot4.choose_action(Local_Observaion[4])
                    act5_robot = Robot5.choose_action(Local_Observaion[5])
                    act6_robot = Robot6.choose_action(Local_Observaion[6])
                    act7_robot = Robot7.choose_action(Local_Observaion[7])
                    act8_robot = Robot8.choose_action(Local_Observaion[8])
                    act9_robot = Robot9.choose_action(Local_Observaion[9])
                    act10_robot = Robot10.choose_action(Local_Observaion[10])
                    act11_robot = Robot11.choose_action(Local_Observaion[11])
                    act12_robot = Robot12.choose_action(Local_Observaion[12])
                    act13_robot = Robot13.choose_action(Local_Observaion[13])
                    act14_robot = Robot14.choose_action(Local_Observaion[14])
                    act15_robot = Robot15.choose_action(Local_Observaion[15])
                    act16_robot = Robot16.choose_action(Local_Observaion[16])
                    act17_robot = Robot17.choose_action(Local_Observaion[17])
                    act18_robot = Robot18.choose_action(Local_Observaion[18])
                    act19_robot = Robot19.choose_action(Local_Observaion[19])                        
                    act0_AP = AP1.choose_action(AP_observation)
                    act1_AP = AP2.choose_action(AP_observation)
                    act2_AP = AP3.choose_action(AP_observation)
                    act3_AP = AP4.choose_action(AP_observation)
                act_robots = np.concatenate([act0_robot, act1_robot, act2_robot, act3_robot, act4_robot, act5_robot, act6_robot, act7_robot, act8_robot, act9_robot,
                                             act10_robot, act11_robot, act12_robot, act13_robot, act14_robot, act15_robot, act16_robot, act17_robot, act18_robot, act19_robot])
                act_robo_move = np.concatenate([act0_robot_move, act1_robot_move, act2_robot_move, act3_robot_move, act4_robot_move, act5_robot_move, act6_robot_move, act7_robot_move, act8_robot_move, act9_robot_move,
                                                act10_robot_move, act11_robot_move, act12_robot_move, act13_robot_move, act14_robot_move, act15_robot_move, act16_robot_move, act17_robot_move, act18_robot_move, act19_robot_move])
                act_APs = np.concatenate([act0_AP, act1_AP, act2_AP, act3_AP])
                new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy, posX0, posY0, posX1, posY1 = env.step_task_offloading(act_APs, act_robots, act_robo_move)
                Robot0.remember(Local_Observaion[0], act0_robot, reward_Robot[0], new_state[0], done)
                Robot1.remember(Local_Observaion[1], act1_robot, reward_Robot[1], new_state[1], done)
                Robot2.remember(Local_Observaion[2], act2_robot, reward_Robot[2], new_state[2], done)
                Robot3.remember(Local_Observaion[3], act3_robot, reward_Robot[3], new_state[3], done)
                Robot4.remember(Local_Observaion[4], act4_robot, reward_Robot[4], new_state[4], done)
                Robot5.remember(Local_Observaion[5], act5_robot, reward_Robot[5], new_state[5], done)
                Robot6.remember(Local_Observaion[6], act6_robot, reward_Robot[6], new_state[6], done)
                Robot7.remember(Local_Observaion[7], act7_robot, reward_Robot[7], new_state[7], done)
                Robot8.remember(Local_Observaion[8], act8_robot, reward_Robot[8], new_state[8], done)
                Robot9.remember(Local_Observaion[9], act9_robot, reward_Robot[9], new_state[9], done)
                Robot10.remember(Local_Observaion[10], act10_robot, reward_Robot[10], new_state[10], done)
                Robot11.remember(Local_Observaion[11], act11_robot, reward_Robot[11], new_state[11], done)
                Robot12.remember(Local_Observaion[12], act12_robot, reward_Robot[12], new_state[12], done)
                Robot13.remember(Local_Observaion[13], act13_robot, reward_Robot[13], new_state[13], done)
                Robot14.remember(Local_Observaion[14], act14_robot, reward_Robot[14], new_state[14], done)
                Robot15.remember(Local_Observaion[15], act15_robot, reward_Robot[15], new_state[15], done)
                Robot16.remember(Local_Observaion[16], act16_robot, reward_Robot[16], new_state[16], done)
                Robot17.remember(Local_Observaion[17], act17_robot, reward_Robot[17], new_state[17], done)
                Robot18.remember(Local_Observaion[18], act18_robot, reward_Robot[18], new_state[18], done)
                Robot19.remember(Local_Observaion[19], act19_robot, reward_Robot[19], new_state[19], done)                    
                AP1.remember(AP_observation, act0_AP, reward_AP[0], new_state[numberOfRobots], done)
                AP2.remember(AP_observation, act1_AP, reward_AP[1], new_state[numberOfRobots], done)
                AP3.remember(AP_observation, act2_AP, reward_AP[2], new_state[numberOfRobots], done)
                AP4.remember(AP_observation, act3_AP, reward_AP[3], new_state[numberOfRobots], done)
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
                Robot10.learn()
                Robot11.learn()
                Robot12.learn()
                Robot13.learn()
                Robot14.learn()
                Robot15.learn()
                Robot16.learn()
                Robot17.learn()
                Robot18.learn()
                Robot19.learn()                    
                AP1.learn()
                AP2.learn()
                AP3.learn()
                AP4.learn()
                score_Robot += np.average(reward_Robot)
                score_AP += np.average(reward_AP)
                print("Robots reward = ", score_Robot)
                print("APs reward = ", score_AP)
                if i %10 == 0:
                    with open("01-accept_LR_high_Sensor_served.txt", 'a') as reward_APs:
                        reward_APs.write(str(accept) + '\n')
                    with open("01-AoI_LR_high_Sensor_served.txt", 'a') as AoI_file:
                        AoI_file.write(str(AoI) + '\n')
                    with open("01-energy_LR_high_Sensor_served.txt", 'a') as energy_file:
                        energy_file.write(str(energy) + '\n')
                obs = new_state
                obs_move = new_state

            with open("01-reward_robot_LR_high_Sensor_served_Reject.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("01-reward_AP_LR_high_Sensor_served_Reject.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            with open("01-pos_0_high_Energy_Sensor_served.txt", 'a') as pos0:
                pos0.write(str(posX0) + ', ' + str(posY0) + '\n')
            with open("01-pos_1_high_Energy_Sensor_served.txt", 'a') as pos1:
                pos1.write(str(posX1) + ', ' + str(posY1) + '\n')                 

            Robot0_mover.remember(Local_Observaion[0], act0_robot_move, score_Robot, new_state[0], False)
            Robot1_mover.remember(Local_Observaion[1], act1_robot_move, score_Robot, new_state[1], False)
            Robot2_mover.remember(Local_Observaion[2], act2_robot_move, score_Robot, new_state[2], False)
            Robot3_mover.remember(Local_Observaion[3], act3_robot_move, score_Robot, new_state[3], False)
            Robot4_mover.remember(Local_Observaion[4], act4_robot_move, score_Robot, new_state[4], False)
            Robot5_mover.remember(Local_Observaion[5], act5_robot_move, score_Robot, new_state[5], False)
            Robot6_mover.remember(Local_Observaion[6], act6_robot_move, score_Robot, new_state[6], False)
            Robot7_mover.remember(Local_Observaion[7], act7_robot_move, score_Robot, new_state[7], False)
            Robot8_mover.remember(Local_Observaion[8], act8_robot_move, score_Robot, new_state[8], False)
            Robot9_mover.remember(Local_Observaion[9], act9_robot_move, score_Robot, new_state[9], False)
            Robot10_mover.remember(Local_Observaion[10], act10_robot_move, score_Robot, new_state[10], False)
            Robot11_mover.remember(Local_Observaion[11], act11_robot_move, score_Robot, new_state[11], False)
            Robot12_mover.remember(Local_Observaion[12], act12_robot_move, score_Robot, new_state[12], False)
            Robot13_mover.remember(Local_Observaion[13], act13_robot_move, score_Robot, new_state[13], False)
            Robot14_mover.remember(Local_Observaion[14], act14_robot_move, score_Robot, new_state[14], False)
            Robot15_mover.remember(Local_Observaion[15], act15_robot_move, score_Robot, new_state[15], False)
            Robot16_mover.remember(Local_Observaion[16], act16_robot_move, score_Robot, new_state[16], False)
            Robot17_mover.remember(Local_Observaion[17], act17_robot_move, score_Robot, new_state[17], False)
            Robot18_mover.remember(Local_Observaion[18], act18_robot_move, score_Robot, new_state[18], False)
            Robot19_mover.remember(Local_Observaion[19], act19_robot_move, score_Robot, new_state[19], False)            
          

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
            Robot10_mover.learn()
            Robot11_mover.learn()
            Robot12_mover.learn()
            Robot13_mover.learn()
            Robot14_mover.learn()
            Robot15_mover.learn()
            Robot16_mover.learn()
            Robot17_mover.learn()
            Robot18_mover.learn()
            Robot19_mover.learn()    

            score_Robot = 0
            score_AP = 0        