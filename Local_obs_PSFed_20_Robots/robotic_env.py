from typing import DefaultDict
import numpy as np
import math
from math import sqrt
from gym.utils import seeding
from devices import AP, Robot, Sensor
import itertools


class RobotEnv(object):

    def __init__(self, numOfRobo, numOfAPs):
        self.numOfAction_Robots = 8
        self.numOfAction_APs = 8
        self.numOfSensors = 25
        self.numOfRobots = numOfRobo
        self.numOfAPs = numOfAPs
        self.seed()
        self.AP = []
        self.robot = []
        self.Sensor = []
        self.np_random = 0
        self.APs, self.Robots, self.Sensors = self.build()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build(self):
        self.AP.append(AP(posX=800, posY=100, compResource=np.random.uniform(10e9, 12e9), costForProcess=np.random.uniform(5, 10), currentTask=dict([("type", "Null"), ("size", 0), ("source", "Null")]), XIaccess=0.0078125, PHIaccess=0.01))
        self.AP.append(AP(posX=800, posY=100, compResource=np.random.uniform(10e9, 12e9), costForProcess=np.random.uniform(5, 10), currentTask=dict([("type", "Null"), ("size", 0), ("source", "Null")]), XIaccess=0.0078125, PHIaccess=0.01))
        self.AP.append(AP(posX=100, posY=800, compResource=np.random.uniform(10e9, 12e9), costForProcess=np.random.uniform(5, 10), currentTask=dict([("type", "Null"), ("size", 0), ("source", "Null")]), XIaccess=0.0078125, PHIaccess=0.01))
        self.AP.append(AP(posX=100, posY=800, compResource=np.random.uniform(10e9, 12e9), costForProcess=np.random.uniform(5, 10), currentTask=dict([("type", "Null"), ("size", 0), ("source", "Null")]), XIaccess=0.0078125, PHIaccess=0.01))
        for i in range(self.numOfRobots):
            self.robot.append(Robot(
                posX=np.random.uniform(10, 900),  # m
                posY=np.random.uniform(10, 900),  # m
                speedX=np.random.uniform(1, 10),  # m/s
                speedY=np.random.uniform(1, 10),  # m/s
                battery=np.random.uniform(300, 500),  # Wh
                cpu=np.random.uniform(100.e6, 200e6),  # Hz
                XI=np.random.uniform(0.01, 0.03),  # no unit ---> as maximum for 64 bit CPU 1.5 GHZ x 2 cores x 8 bytes = 24 GB/S
                PHI=np.random.uniform(0.1, .2)))  # no unit
        for s in range(self.numOfSensors):
            self.Sensor.append(Sensor(s))
        return self.AP, self.robot, self.Sensor

    def ChGains(self):
        channel_gains = [[0 for m in range(self.numOfRobots + self.numOfAPs)] for n in range(self.numOfRobots)]
        for i in range(self.numOfRobots):
            x, y = self.Robots[i].posRobot()
            for j in range(self.numOfRobots):
                if i is not j:
                    x_dest, y_dest = self.Robots[j].posRobot()
                    L = 5.76 * np.log2(sqrt(pow((x - x_dest), 2) + pow((y - y_dest), 2)))
                    c = -1 * L / 50
                    antenna_gain = 0.9
                    s = 0.8
                    channel_gains[i][j] = pow(10, c) * math.sqrt((antenna_gain * s))  # * np.random.rayleigh(scale=1.0, size=1)
            for k in range(self.numOfAPs):
                x_dest_AP, y_dest_AP = self.APs[k].posAP()
                L = 5.76 * np.log2(sqrt(pow((x - x_dest_AP), 2) + pow((y - y_dest_AP), 2)))
                c = -1 * L / 50
                antenna_gain = 0.9
                s = 0.8
                channel_gains[i][k + self.numOfRobots] = pow(10, c) * math.sqrt(
                    (antenna_gain * s))  # * np.random.rayleigh(scale=1.0, size=1)
        return channel_gains

    def step_task_offloading(self, action_APs, action_Robots, action_move):
        reward_test = 0
        reward_test_Ap = 0
        """
        Here, I take care of robots movements
        """
        path_planing_obs_next = []
        energy_move_all = []
        for i in range(self.numOfRobots):
            # I considered at each 1 millisecond the task offloading happens, so I pass "1.0e-3" as the last argument as time duration
            posX, posY, energy_move = self.Robots[i].move(int(10 * action_move[2 * i + 0]), int(10 * action_move[2 * i + 1]), 1.0e-3)
            path_planing_obs_next.append(posX)
            path_planing_obs_next.append(posY)
            path_planing_obs_next.append(energy_move)
            energy_move_all.append(energy_move)

        gainRo = self.ChGains()
        gainAP_AP = 0.3  # just considered a fixed value
        Noise = 1e-20  # -170 dBm/Hz
        RB_BW = 180.0e3 * 2  # For simplicity, I considered big RBs
        reward_Robots = np.zeros(self.numOfRobots)
        reward_APs = np.zeros(self.numOfAPs)
        delay_reward_robot = np.zeros(self.numOfRobots)
        energy_reward_robot = np.zeros(self.numOfRobots)
        CPU_reward_robot = np.zeros(self.numOfRobots)
        delay_reward_AP = np.zeros(self.numOfAPs)
        CPU_reward_AP = np.zeros(self.numOfAPs)
        txDelay_rbt = np.zeros(self.numOfRobots)
        type_of_comm_flag = 0  # 1 for robot_only, 2 for AP only, and 3 for both
        howManyPortion = 0
        Destination2 = 0
        RB2 = 0
        PW2 = 0
        Destination3 = 0
        RB3 = 0
        PW3 = 0
        taskCPU = 0
        taskSize = 0
        taskTargetDelay = 0
        accept = 0
        averageAoI = 0

        '''
        Here, I reserve the actions of all robots to some buffer for further evaluation
        the action for each robot is: - Which sensors data
                                      - howManyPortions,
                                      - destination2 for offloading,
                                      - Power2 for transmission,
                                      - RB2 for transmission,
                                      - destination3 for offloading,
                                      - Power3 for transmission,
                                      - RB3 for transmission)
        Also, in the following I take care of flags and RB reservation for uplink transmission
        '''
        temp_served_sensor = 0
        served_sensors = 0
        for i in range(self.numOfRobots):
            picked_sensor = int(self.numOfSensors / 2 * (np.clip(action_Robots[0 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))
            if np.clip(action_Robots[1 + i * self.numOfAction_Robots], -1.0, 1.0) < -0.7:
                howManyPortion = 1
            elif -0.7 <= np.clip(action_Robots[1 + i * self.numOfAction_Robots], -1.0, 1.0) < 0.3:
                howManyPortion = 2
            else:
                howManyPortion = 3
                # howManyPortion = int(action_Robots[0 + i * self.numOfAction_Robots] + 1)
            if int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) == i:  # 3 is chosed to cover 2 robots and 4 APs which are 6 in total
                if i < self.numOfRobots - 1:
                    Destination2 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) + 1
                else:
                    Destination2 = 0
            else:
                Destination2 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))

            PW2 = 10 * (np.clip(action_Robots[3 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)
            RB2 = int((self.numOfRobots / 2) * (np.clip(action_Robots[4 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))  # 1 is considered since there are only 2 robots
            if int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) == i:
                if i < self.numOfRobots - 1:
                    Destination3 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) + 1
                else:
                    Destination3 = 0
            else:
                Destination3 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))
            # this part is only for correcting the action problem to not go beyond number of all destinations
            if Destination2 == self.numOfRobots + self.numOfAPs:
                Destination2 = self.numOfRobots + self.numOfAPs - 1
            if Destination3 == self.numOfRobots + self.numOfAPs:
                Destination3 = self.numOfRobots + self.numOfAPs - 1
            PW3 = 10 * (np.clip(action_Robots[6 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)
            RB3 = int((self.numOfRobots / 2) * (np.clip(action_Robots[7 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))

            '''
            Here the robots catch the sensors data and create task for themselves. each robot has only one task
            '''
            self.Robots[i].getTask(self.Sensors[picked_sensor].reqCPU,
                                   self.Sensors[picked_sensor].dataSize,
                                   self.Sensors[picked_sensor].delay,
                                   self.Sensors[picked_sensor].VoI,
                                   self.Sensors[picked_sensor].posX,
                                   self.Sensors[picked_sensor].posY)
            if temp_served_sensor == picked_sensor:
                pass
            else:
                served_sensors += 1
                temp_served_sensor = picked_sensor

            taskCPU, taskSize, taskTargetDelay, taskVoI, taskposX, taskposY = self.Robots[i].taskSpec()
            if howManyPortion == 1:
                type_of_comm_flag = "Local"
                self.Robots[i].setFlag(howManyPortion, type_of_comm_flag)
                self.Robots[i].successfulUpload()
                self.Robots[i].reward_reject += 1
                accept += 1
            elif howManyPortion == 2:
                # Reserve the chosen RB ...
                if Destination2 < self.numOfRobots:  # destination is another robot
                    type_of_comm_flag = "2_Robot"
                    self.Robots[i].reserveRB_Ro_Ro_1(RB2, PW2, taskSize / howManyPortion, Destination2)
                else:  # destination is AP
                    type_of_comm_flag = "2_AP"
                    self.Robots[i].reserveRB_Ro_AP_1(RB2, PW2, taskSize / howManyPortion, Destination2)
                self.Robots[i].setFlag(howManyPortion, type_of_comm_flag)

            elif howManyPortion == 3:

                # Reserve the chosen RB and buffer task to another robots...
                '''
                I consider 100 MHz BW; half of it is for Robot-Robot and another half is for Robot-AP.
                Also, each half is divided into 10 subcarriers with 5 MHz BW, each!
                '''
                if Destination2 < self.numOfRobots and Destination3 < self.numOfRobots:  # destinations are other robots
                    type_of_comm_flag = "3_Only_Robot"
                    self.Robots[i].reserveRB_Ro_Ro_2(RB2, PW2, taskSize / howManyPortion, Destination2, RB3, PW3, taskSize / howManyPortion, Destination3)
                elif Destination2 >= self.numOfRobots and Destination3 >= self.numOfRobots:  # destinations are APs
                    type_of_comm_flag = "3_Only_AP"
                    self.Robots[i].reserveRB_Ro_AP_2(RB2, PW2, taskSize / howManyPortion, Destination2, RB3, PW3, taskSize / howManyPortion, Destination3)
                elif Destination2 < self.numOfRobots <= Destination3:
                    type_of_comm_flag = "3_Robot&AP"
                    self.Robots[i].reserveRB_Ro_Ro_1(RB2, PW2, taskSize / howManyPortion, Destination2)
                    self.Robots[i].reserveRB_Ro_AP_1(RB3, PW3, taskSize / howManyPortion, Destination3)
                elif Destination2 >= self.numOfRobots > Destination3:
                    type_of_comm_flag = "3_AP&Robot"
                    self.Robots[i].reserveRB_Ro_AP_1(RB2, PW2, taskSize / howManyPortion, Destination2)
                    self.Robots[i].reserveRB_Ro_Ro_1(RB3, PW3, taskSize / howManyPortion, Destination3)
                else:
                    print("Error in Destination selection  8888888888888888888888888888 Destination2  ", Destination2, Destination3)
                self.Robots[i].setFlag(howManyPortion, type_of_comm_flag)
            else:  # Das ist fur debuggen
                print("------------------------------\n\n")
                print("howManyPortion", howManyPortion)
                print("000000000000000000000000000000\n\n")

        # .......................................
        # .....Now, I start task offloading .....
        # .......................................
        '''
        Calculating data rates for uplink and transmission delays
        '''
        for i in range(self.numOfRobots):
            RBnumber = -1
            PW = 0
            size = 0
            dest = -1
            RBnumber1 = -1
            PW1 = 0
            size1 = 0
            dest1 = -1
            RB_I = -1
            RB_I1 = -1
            RB_I2 = -1
            PW_I = 0
            PW_I1 = 0
            PW_I2 = 0
            Interference = 0
            Interference1 = 0
            numPortion, CommType = self.Robots[i].numOfPortions()
            if numPortion == 2:  # the destination can be a robot or an AP
                if CommType == "2_Robot":  # destination is only one robot
                    RBnumber, PW, size, dest = self.Robots[i].recallRB_Ro_Ro_1()
                if CommType == "2_AP":  # destination is only one AP
                    RBnumber, PW, size, dest = self.Robots[i].recallRB_Ro_AP_1()
                '''
                In the following for loop, I calculate the interferences from other communications
                to device with index "dest"
                '''
                for r in range(self.numOfRobots):
                    if self.Robots[r].local_flag == 1:
                        RB_I, PW_I = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 2:
                        RB_I, PW_I = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 3:
                        RB_I1, PW_I1, RB_I2, PW_I2 = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 4:
                        RB_I1, PW_I1, RB_I2, PW_I2 = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    else:
                        pass
                    if RB_I == RBnumber:
                        Interference += gainRo[r][dest] * PW_I
                    if RB_I1 == RBnumber:
                        Interference += gainRo[r][dest] * PW_I1
                    if RB_I2 == RBnumber:
                        Interference += gainRo[r][dest] * PW_I2
                # print("\n\n\n")
                # print(" interference in Portion 2   ------------------------- ", Interference)
                # print(" gainRo[i][dest] in Portion 2*** between ",i, dest,"  is  ", gainRo[i][dest])
                # print(" PW  and RB in Portion 2--------------------------------", PW, RB_BW)
                txRate = RB_BW * math.log2(1 + (PW * gainRo[i][dest]) / (Noise + Interference))
                if txRate != 0:
                    txDelay_rbt[i] = size / txRate  # Note that the 'size' here is based on millisecond
                else:
                    txDelay_rbt[i] = 10000  # This is for punishment, and of course is not real
                # print(" txRate    in Portion 2--------------------------------", txRate)
                if txRate >= self.Robots[i].taskSize / numPortion:
                    self.Robots[i].successfulUpload()
                    self.Robots[i].reward_reject += 1
                    accept += 1
                else:
                    self.Robots[i].reward_reject -= 1000
                    averageAoI -= 1000
            elif numPortion == 3:  # the destinations can be robots AND APs
                if CommType == "3_Only_Robot":  # both destinations are robots
                    RBnumber, PW, size, dest, RBnumber1, PW1, size1, dest1 = self.Robots[i].recallRB_Ro_Ro_2()
                elif CommType == "3_Only_AP":  # both destinations are APs
                    RBnumber, PW, size, dest, RBnumber1, PW1, size1, dest1 = self.Robots[i].recallRB_Ro_AP_2()
                elif CommType == "3_Robot&AP" or CommType == "3_AP&Robot":  # one destination is robot and the other one is AP
                    RBnumber, PW, size, dest = self.Robots[i].recallRB_Ro_Ro_1()
                    RBnumber1, PW1, size1, dest1 = self.Robots[i].recallRB_Ro_AP_1()
                else:
                    print("Error --------XXXXXXXXXXXXXXXXXXXXX \n\n\n CommType", CommType, "numPortion  ", numPortion)
                '''
                In the following for loop, I calculate the interferences from other communications
                to device with index "dest"
                '''
                for r in range(self.numOfRobots):
                    if self.Robots[r].local_flag == 1:
                        RB_I, PW_I = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 2:
                        RB_I, PW_I = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 3:
                        RB_I1, PW_I1, RB_I2, PW_I2 = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    elif self.Robots[r].local_flag == 4:
                        RB_I1, PW_I1, RB_I2, PW_I2 = self.Robots[r].chosenRBPW(self.Robots[r].local_flag)
                    else:
                        pass
                    if RB_I == RBnumber:
                        Interference += gainRo[r][dest] * PW_I
                    if RB_I1 == RBnumber:
                        Interference += gainRo[r][dest] * PW_I1
                    if RB_I2 == RBnumber:
                        Interference += gainRo[r][dest] * PW_I2
                    if RB_I == RBnumber1:
                        Interference1 += gainRo[r][dest1] * PW_I
                    if RB_I1 == RBnumber1:
                        Interference1 += gainRo[r][dest1] * PW_I1
                    if RB_I2 == RBnumber1:
                        Interference1 += gainRo[r][dest1] * PW_I2
                # print(" interference in Portion 3   ---------------------------", Interference)
                # print(" interference1 in Portion 3   --------------------------", Interference1)
                # print(" gainRo[i][dest] in Portion 2*** between  ",i, dest,"  is  ", gainRo[i][dest])
                # print(" gainRo[i][dest1] in Portion 2*** between ", i, dest1, "  is  ", gainRo[i][dest1])
                # print(" PW  and RB in Portion 2----------------------------------", PW, RB_BW)
                # print(" PW1  and RB1 in Portion 2--------------------------------", PW1, RB_BW)
                # Calculate the rate for the first destination
                txRate = RB_BW * math.log2(1 + (PW * gainRo[i][dest]) / (Noise + Interference))
                # Calculate the rate for the second destination
                txRate1 = RB_BW * math.log2(1 + (PW1 * gainRo[i][dest1]) / (Noise + Interference1))
                if txRate != 0 and txRate1 != 0:
                    txDelay_rbt[i] = (size / txRate) + (size1 / txRate1)
                else:
                    txDelay_rbt[i] = 10000  # This is for punishment, and of course is not real
                # print(" txRate----------------------------------------------------", txRate)
                # print(" txRate1---------------------------------------------------", txRate1)
                # print("\n\n\n")
                if txRate >= self.Robots[i].taskSize / numPortion and txRate1 >= self.Robots[i].taskSize / numPortion:
                    self.Robots[i].successfulUpload()
                    self.Robots[i].reward_reject += 1
                    accept += 1
                else:
                    self.Robots[i].reward_reject -= 1000
                    averageAoI -= 1000
            else:
                pass

        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        for i in range(self.numOfRobots):
            if np.clip(action_Robots[1 + i * self.numOfAction_Robots], -1.0, 1.0) < -0.7:
                howManyPortion = 1
            elif -0.7 <= np.clip(action_Robots[1 + i * self.numOfAction_Robots], -1.0, 1.0) < 0.3:
                howManyPortion = 2
            else:
                howManyPortion = 3
            # howManyPortion = int(action_Robots[1 + i * self.numOfAction_Robots] + 1)
            if int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2+ i * self.numOfAction_Robots], -1.0, 1.0) + 1)) == i:
                if i < self.numOfRobots - 1:
                    Destination2 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) + 1
                else:
                    Destination2 = 0
            else:
                Destination2 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[2 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))

            PW2 = 10 * (np.clip(action_Robots[3 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)
            RB2 = int((self.numOfRobots / 2) * (np.clip(action_Robots[4 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))
            if int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) == i:
                if i < self.numOfRobots - 1:
                    Destination3 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)) + 1
                else:
                    Destination3 = 0
            else:
                Destination3 = int(((self.numOfRobots + self.numOfAPs) / 2) * (np.clip(action_Robots[5 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))
            if Destination2 == self.numOfRobots + self.numOfAPs:
                Destination2 = self.numOfRobots + self.numOfAPs - 1
            if Destination3 == self.numOfRobots + self.numOfAPs:
                Destination3 = self.numOfRobots + self.numOfAPs - 1
            PW3 = 10 * (np.clip(action_Robots[6 + i * self.numOfAction_Robots], -1.0, 1.0) + 1)
            RB3 = int((self.numOfRobots / 2) * (np.clip(action_Robots[7 + i * self.numOfAction_Robots], -1.0, 1.0) + 1))
            taskCPU, taskSize, taskTargetDelay, taskVoI, taskposX, taskposY = self.Robots[i].taskSpec()
            if self.Robots[i].uploadOK():
                if howManyPortion == 1:
                    if self.Robots[i].robotCPU > taskCPU:
                        # buffer the task to robots
                        self.Robots[i].bufferTask(taskCPU, taskSize)
                        # self.Robots[i].taskAoI = 1
                        self.Robots[i].reward_VoI += taskVoI
                    else:
                        self.Robots[i].reward_VoI -= taskVoI
                        self.Robots[i].taskAoI += 1
                elif howManyPortion == 2:
                    local_Flag_for_AoI = 0
                    if self.Robots[i].robotCPU > taskCPU / howManyPortion:
                        # buffer the task to robots
                        self.Robots[i].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                        local_Flag_for_AoI += 0.5
                    if Destination2 < self.numOfRobots:  # destination is another robot
                        if self.Robots[Destination2].robotCPU > taskCPU / howManyPortion:
                            # Buffer task to the destination robot
                            self.Robots[Destination2].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                            local_Flag_for_AoI += 0.5
                    else:  # destination is an AP
                        # I assume APs always have enough CPU for process task, but this imposes more cost for robots
                        # Buffer task to the destination AP
                        self.APs[Destination2 - self.numOfRobots].bufferTask_AP(taskCPU / howManyPortion, taskSize / howManyPortion)
                        local_Flag_for_AoI += 0.5
                    if local_Flag_for_AoI >= 1:
                        # self.Robots[i].taskAoI = 1
                        self.Robots[i].reward_VoI += taskVoI
                    else:
                        # self.Robots[i].taskAoI += 1
                        self.Robots[i].reward_VoI -= taskVoI
                elif howManyPortion == 3:
                    local_Flag_for_AoI = 0
                    if self.Robots[i].robotCPU > taskCPU / howManyPortion:
                        # buffer the task to robot
                        self.Robots[i].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                        local_Flag_for_AoI += 0.5
                    if Destination2 < self.numOfRobots and Destination3 < self.numOfRobots:  # destinations are other robots
                        if self.Robots[Destination2].robotCPU > (taskCPU / howManyPortion) and self.Robots[Destination3].robotCPU > (taskCPU / howManyPortion):
                            # buffering ...
                            self.Robots[Destination2].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                            self.Robots[Destination3].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                            local_Flag_for_AoI += 0.5

                    elif Destination2 >= self.numOfRobots and Destination3 >= self.numOfRobots:  # destinations are APs
                        # Here, again I assume APs have enough CPU capacity
                        # buffering
                        self.APs[Destination2 - self.numOfRobots].bufferTask_AP(taskCPU / howManyPortion, taskSize / howManyPortion)
                        self.APs[Destination3 - self.numOfRobots].bufferTask_AP(taskCPU / howManyPortion, taskSize / howManyPortion)
                    elif Destination2 < self.numOfRobots <= Destination3:
                        if self.Robots[Destination2].robotCPU > taskCPU / howManyPortion:
                            # buffering
                            self.Robots[Destination2].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                            self.APs[Destination3 - self.numOfRobots].bufferTask_AP(taskCPU / howManyPortion, taskSize / howManyPortion)
                            local_Flag_for_AoI += 0.5
                    elif Destination2 >= self.numOfRobots > Destination3:
                        if self.Robots[Destination3].robotCPU > taskCPU / howManyPortion:
                            # buffering
                            self.APs[Destination2 - self.numOfRobots].bufferTask_AP(taskCPU / howManyPortion, taskSize / howManyPortion)
                            self.Robots[Destination3].bufferTask(taskCPU / howManyPortion, taskSize / howManyPortion)
                            local_Flag_for_AoI += 0.5
                    else:
                        print("Error in Destination selection  8888888888888888888888888888 Destination2  ", Destination2, Destination3)
                    if local_Flag_for_AoI >= 1:
                        # self.Robots[i].taskAoI = 1
                        self.Robots[i].reward_VoI += taskVoI
                    else:
                        # self.Robots[i].taskAoI += 1
                        self.Robots[i].reward_VoI -= taskVoI
                else:  # Das ist fur debuggen
                    print("------------------------------\n\n")
                    print("howManyPortion", howManyPortion)
                    print("000000000000000000000000000000\n\n")

        '''
        Here, I reserve the actions of all APs
        the action for each AP is: -size of migration1
                                   -destination1
                                   -PW1
                                   -RB1
                                   -size of migration2
                                   -destination2 
                                   -PW2
                                   -RB2
        '''
        for j in range(self.numOfAPs):
            sizeOfMigration1 = int((np.clip(action_APs[0 + self.numOfAction_APs], -1.0, 1.0) + 1) / 2)
            DestinationAP1 = int((np.clip(action_APs[1 + self.numOfAction_APs], -1.0, 1.0) + 1) * 3 / 2)
            PW_AP1 = 10 * (np.clip(action_APs[2 + self.numOfAction_APs], -1.0, 1.0) + 1)
            RB_AP1 = int(5 * (np.clip(action_APs[3 + self.numOfAction_APs], -1.0, 1.0) + 1))
            sizeOfMigration2 = int((np.clip(action_APs[4 + self.numOfAction_APs], -1.0, 1.0) + 1) / 2)
            DestinationAP2 = int((np.clip(action_APs[5 + self.numOfAction_APs], -1.0, 1.0) + 1) * 3 / 2)
            PW_AP2 = 10 * (np.clip(action_APs[6 + self.numOfAction_APs], -1.0, 1.0) + 1)
            RB_AP2 = int(5 * (np.clip(action_APs[7 + self.numOfAction_APs], -1.0, 1.0) + 1))
            self.APs[j].resCommResrc(RB_AP1, PW_AP1, RB_AP2, PW_AP2)
            currentAP_taskCPU, currentAP_taskSize = self.APs[j].recallAlltasks()
            # here I should write the code about transmission between APs due to migration
            if DestinationAP1 is not j:
                self.APs[j].releaseTask(sizeOfMigration1 * currentAP_taskCPU, sizeOfMigration1 * currentAP_taskSize)
                txRateAP = RB_BW * math.log2(1 + (PW_AP1 * gainAP_AP) / Noise)
                print("\n\n", "tx rate AP  1 = ", txRateAP, "\n\n")
                if txRateAP >= sizeOfMigration1 * currentAP_taskSize:
                    self.APs[DestinationAP1].bufferTask_AP(sizeOfMigration1 * currentAP_taskCPU, sizeOfMigration1 * currentAP_taskSize)
                    self.APs[j].reward_reject += 1
                else:
                    self.APs[j].reward_reject -= 1000

            if DestinationAP2 is not j:
                self.APs[j].releaseTask(sizeOfMigration2 * currentAP_taskCPU, sizeOfMigration2 * currentAP_taskSize)
                txRateAP = RB_BW * math.log2(1 + (PW_AP2 * gainAP_AP) / Noise)
                print("\n\n", "tx rate AP  2 = ", txRateAP, "\n\n")
                if txRateAP > sizeOfMigration2 * currentAP_taskSize:
                    self.APs[DestinationAP2].bufferTask_AP(sizeOfMigration2 * currentAP_taskCPU, sizeOfMigration2 * currentAP_taskSize)
                    self.APs[j].reward_reject += 1
                else:
                    self.APs[j].reward_reject -= 1000
        '''
        Calculating CPU, delay, and energy processes
        '''
        AoI = 0
        Energy = 0
        for i in range(self.numOfRobots):
            alloCPU, portionSize = self.Robots[i].recallAlltasks()
            procDelay, procEnergy, residualrobotCPU, residualrobotBattery = self.Robots[i].perform_task_Robot(alloCPU, portionSize)
            delay_reward_robot[i] = procDelay
            AoI += self.Robots[i].taskAoI
            self.Robots[i].reward_energy += procEnergy

        '''
        From here, the rewards are calculated
        '''
        reward_VoI = []
        reward_energy = []
        for i in range(self.numOfRobots):
            if self.Robots[i].reward_VoI < 50:
                reward_Robots[i] -= 2000
            else:
                reward_Robots[i] += 10 * served_sensors
            if self.Robots[i].reward_energy < 1000000:
                reward_Robots[i] += 10 * served_sensors
            else:
                reward_Robots[i] -= 1000
            if accept < self.numOfRobots:
                reward_Robots[i] -= 1000
            else:
                reward_Robots[i] += 10 * served_sensors
            # reward_Robots[i] = ((self.Robots[i].reward_VoI - self.Robots[i].reward_energy/1000000) * self.Robots[i].reward_reject)/1000
            reward_VoI = self.Robots[i].reward_VoI
            reward_energy = self.Robots[i].reward_energy
            self.Robots[i].reward_energy = 0
            self.Robots[i].reward_VoI = 0
        for j in range(self.numOfAPs):
            reward_APs[j] = self.APs[j].reward_reject
            self.APs[j].reward_reject = 0

        observation_next = self.NextState()
        self.createTaskRobot()
        done = False
        info = {}
        return observation_next, reward_Robots, reward_APs, done, info, accept, np.average(reward_VoI), np.average(reward_energy), self.Robots[0].posX, self.Robots[0].posY, self.Robots[1].posX, self.Robots[1].posY,

    def reset(self):
        _ = self.ChGains()
        _, __, ___ = self.build()
        posX_all_robots = []
        posY_all_robots = []
        speedX_all_robots = []
        speedY_all_robots = []
        battery_all_robots = []
        cpu_all_robots = []
        tasksize_all_robots = []
        taskcpu_all_robots = []
        taskdelay_all_robots = []
        taskGamma_all_robtos = []
        taskAoI_all_robtos = []

        posX_all_APs = []
        posY_all_APs = []
        compRsrc_all_APs = []

        posX_all_sensors = []
        posY_all_sensors = []
        VoI_all_sensors = []
        cpu_all_sensors = []
        taskSize_all_sensors = []

        for i in range(self.numOfRobots):
            posX_all_robots.append(self.Robots[i].posX / 1000)
            posY_all_robots.append(self.Robots[i].posY / 1000)
            speedX_all_robots.append(self.Robots[i].speedX / 10)
            speedY_all_robots.append(self.Robots[i].speedY / 10)
            battery_all_robots.append(self.Robots[i].robotBattery / 500)
            cpu_all_robots.append(self.Robots[i].robotCPU / 1.5e9)
            tasksize_all_robots.append(self.Robots[i].taskSize / 1.e6)
            taskcpu_all_robots.append(self.Robots[i].taskCPU / 200.e6)
            taskdelay_all_robots.append(self.Robots[i].taskTargetDelay / 0.05)
            taskGamma_all_robtos.append(self.Robots[i].taskGamma)
            taskAoI_all_robtos.append(self.Robots[i].taskAoI)
        for j in range(self.numOfAPs):
            posX_all_APs.append(self.APs[j].posX / 1000)
            posY_all_APs.append(self.APs[j].posY / 1000)
            compRsrc_all_APs.append(self.APs[j].compResource / 12.0e9)
        for s in range(self.numOfSensors):
            posX_all_sensors.append(self.Sensors[s].posX / 1000)
            posY_all_sensors.append(self.Sensors[s].posY / 1000)
            VoI_all_sensors.append(self.Sensors[s].VoI)
            cpu_all_sensors.append(self.Sensors[s].reqCPU / 200.0e6)
            taskSize_all_sensors.append(self.Sensors[s].dataSize / 1.0e6)
        out = []
        for R in range(self.numOfRobots + 1):
            if R < self.numOfRobots:
                out.append(np.concatenate([[posX_all_robots[R], posY_all_robots[R], speedX_all_robots[R], speedY_all_robots[R],
                                            battery_all_robots[R], cpu_all_robots[R], tasksize_all_robots[R], taskcpu_all_robots[R],
                                            taskdelay_all_robots[R], taskGamma_all_robtos[R], taskAoI_all_robtos[R]], posX_all_APs, posY_all_APs, compRsrc_all_APs,
                                           posX_all_sensors, posY_all_sensors, VoI_all_sensors, cpu_all_sensors, taskSize_all_sensors]))
            else:
                out.append(np.concatenate([posX_all_robots, posY_all_robots, speedX_all_robots, speedY_all_robots,
                                           battery_all_robots, cpu_all_robots, tasksize_all_robots, taskcpu_all_robots,
                                           taskdelay_all_robots, taskGamma_all_robtos, taskAoI_all_robtos, posX_all_APs, posY_all_APs, compRsrc_all_APs,
                                           posX_all_sensors, posY_all_sensors, VoI_all_sensors, cpu_all_sensors, taskSize_all_sensors]))
        return out

    def NextState(self):
        _ = self.ChGains()
        posX_all_robots = []
        posY_all_robots = []
        speedX_all_robots = []
        speedY_all_robots = []
        battery_all_robots = []
        cpu_all_robots = []
        tasksize_all_robots = []
        taskcpu_all_robots = []
        taskdelay_all_robots = []
        taskGamma_all_robtos = []
        taskAoI_all_robtos = []

        posX_all_APs = []
        posY_all_APs = []
        compRsrc_all_APs = []

        posX_all_sensors = []
        posY_all_sensors = []
        VoI_all_sensors = []
        cpu_all_sensors = []
        taskSize_all_sensors = []

        for i in range(self.numOfRobots):
            posX_all_robots.append(self.Robots[i].posX / 1000)
            posY_all_robots.append(self.Robots[i].posY / 1000)
            speedX_all_robots.append(self.Robots[i].speedX / 10)
            speedY_all_robots.append(self.Robots[i].speedY / 10)
            battery_all_robots.append(self.Robots[i].robotBattery / 500)
            cpu_all_robots.append(self.Robots[i].robotCPU / 1.5e9)
            tasksize_all_robots.append(self.Robots[i].taskSize / 1.0e6)
            taskcpu_all_robots.append(self.Robots[i].taskCPU / 200.0e6)
            taskdelay_all_robots.append(self.Robots[i].taskTargetDelay / 0.05)
            taskGamma_all_robtos.append(self.Robots[i].taskGamma)
            taskAoI_all_robtos.append(self.Robots[i].taskAoI)
        for j in range(self.numOfAPs):
            posX_all_APs.append(self.APs[j].posX / 1000)
            posY_all_APs.append(self.APs[j].posY / 1000)
            compRsrc_all_APs.append(self.APs[j].compResource / 12.0e9)
        for s in range(self.numOfSensors):
            posX_all_sensors.append(self.Sensors[s].posX / 1000)
            posY_all_sensors.append(self.Sensors[s].posY / 1000)
            VoI_all_sensors.append(self.Sensors[s].VoI)
            cpu_all_sensors.append(self.Sensors[s].reqCPU / 200.0e6)
            taskSize_all_sensors.append(self.Sensors[s].dataSize / 1.0e6)
        out = []
        for R in range(self.numOfRobots + 1):
            if R < self.numOfRobots:
                out.append(np.concatenate([[posX_all_robots[R], posY_all_robots[R], speedX_all_robots[R], speedY_all_robots[R],
                                            battery_all_robots[R], cpu_all_robots[R], tasksize_all_robots[R], taskcpu_all_robots[R],
                                            taskdelay_all_robots[R], taskGamma_all_robtos[R], taskAoI_all_robtos[R]], posX_all_APs, posY_all_APs, compRsrc_all_APs,
                                           posX_all_sensors, posY_all_sensors, VoI_all_sensors, cpu_all_sensors, taskSize_all_sensors]))
            else:
                out.append(np.concatenate([posX_all_robots, posY_all_robots, speedX_all_robots, speedY_all_robots,
                                           battery_all_robots, cpu_all_robots, tasksize_all_robots, taskcpu_all_robots,
                                           taskdelay_all_robots, taskGamma_all_robtos, taskAoI_all_robtos, posX_all_APs, posY_all_APs, compRsrc_all_APs,
                                           posX_all_sensors, posY_all_sensors, VoI_all_sensors, cpu_all_sensors, taskSize_all_sensors]))
        return out

    def createTaskRobot(self):
        for i in range(self.numOfRobots):
            self.Robots[i].genTask_Robot()
        for s in range(self.numOfSensors):
            self.Sensor.append(Sensor(s))