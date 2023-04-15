import numpy as np
import math


class AP(object):
    def __init__(self, posX, posY, compResource, costForProcess, currentTask, XIaccess=0.0078125, PHIaccess=0.01):
        self.compResource = compResource  # 64-bit system
        self.costForProcess = costForProcess
        self.posX = posX
        self.posY = posY
        self.XIaccess = XIaccess  # 0.0078125  # required cpu cycle for processing 1 bit (with 9 GHz 64-bit dual-core cpu)
        self.PHIaccess = PHIaccess  # 0.01  # required PW for process one bit of task
        self.currentTaskType = currentTask["type"]
        self.currentTaskSource = currentTask["source"]
        self.currentTaskSize = currentTask["size"]
        self.storedTasks = 0
        self.migrationSize1 = 0
        self.migrationDest1 = 0
        self.migrationSize2 = 0
        self.migrationDest2 = 0

    def consumeCPU(self, taskSize):
        self.compResource -= taskSize
        return self.costForProcess * taskSize / 1.e6, taskSize / self.compResource

    def processCPU(self, taskSize):
        return taskSize / 1.e6 * self.costForProcess, taskSize / self.compResource

    def relaseCPU(self, taskSize):
        self.compResource += taskSize

    def posAP(self):
        return self.posX, self.posY

    def giveMeStatus(self):
        return self.compResource / 10.e9, self.costForProcess / 10, self.posX / 10.e3, self.posY / 1.e3

    def perform_task_AP(self, portionSize):
        procDelay = portionSize * 1000 * self.XIaccess / self.compResource  # here I multiplied by 1000 since I consider the task size with mili second basis
        self.compResource -= portionSize * 1000 * self.XIaccess  # update the cpu,  here I multiplied by 1000 since I consider the task size with mili second basis
        procEnergy = portionSize * 1000 * self.PHIaccess  # here I multiplied by 1000 since I consider the task size with mili second basis
        return procDelay, self.compResource, procEnergy

    def migrate_task(self, alloCPU, task):
        self.compResource += alloCPU  # update the cpu

    def bufferTask_AP(self, portionSize):
        self.storedTasks += portionSize

    def releaseTask(self, portionSize):
        self.storedTasks -= portionSize

    def recallAlltasks(self):
        return self.storedTasks

    def resCommResrc(self, RB_AP1, PW_AP1, RB_AP2, PW_AP2):
        self.migrationSize1 = RB_AP1
        self.migrationDest1 = PW_AP1
        self.migrationSize2 = RB_AP2
        self.migrationDest2 = PW_AP2

    def genCompRsrc(self):
        return np.random.uniform(10e9, 12e9)


class Robot(object):
    def __init__(self, posX, posY, speedX, speedY, battery, cpu, taskSize, taskCPU, taskTargetDelay, XI, PHI, taskGamma):
        numOfRobots = 10
        self.endX = 1000
        self.startX = 0
        self.endY = 1000
        self.startY = 0
        self.XI = XI  # 0.015625  # required cpu cycle for processing 1 bit (with 1.5 GHz 64-bit cpu)
        self.PHI = PHI  # .01  # watt  (The future Internet-An energy consumption perspective, 2009)
        self.energyPermeter = 1
        self.posX = posX
        self.posY = posY
        self.speedX = speedX
        self.speedY = speedY
        self.robotBattery = battery
        self.robotCPU = cpu
        self.taskSize = taskSize
        self.taskCPU = taskCPU
        self.taskTargetDelay = taskTargetDelay
        self.taskGamma = taskGamma
        self.storedTaskSizes = 0
        self.storedTaskCPUs = 0
        self.taskGamma = 0
        self.taskAoI = 10
        self.storedTask = []
        self.chosenRB_Ro_Ro_1 = []
        self.chosenRB_Ro_AP_1 = []
        self.chosenRB_Ro_Ro_2 = []
        self.chosenRB_Ro_AP_2 = []
        self.nOfPortions = 0
        self.typeOfcomm = 0
        self.local_flag = 0  # this flag is defined to reserve the RB for interference calculations
        self.UploadOK = 0  # this flag is for checking for successful upload

    def taskSpec(self):
        return self.taskCPU, self.taskSize, self.taskTargetDelay

    def posRobot(self):
        return self.posX, self.posY

    def move(self, speedX, speedY, deltaT):
        moveX = speedX * deltaT
        moveY = speedY * deltaT
        self.posX += moveX
        self.posY += moveY
        if self.posX > self.endX:
            self.posX = self.endX
        elif self.posX < self.startX:
            self.posX = self.startX
        else:
            pass
        if self.posY > self.endY:
            self.posY = self.endY
        elif self.posY < self.startY:
            self.posY = self.startY
        else:
            pass
        self.robotBattery -= (abs(moveX) + abs(moveY)) * self.energyPermeter
        return self.posX, self.posY, ((moveX + moveY) * self.energyPermeter)

    def bufferTask(self, poritionCPU, portionSize):
        self.storedTaskCPUs += poritionCPU
        self.storedTaskSizes += portionSize

    def recallAlltasks(self):
        return self.storedTaskCPUs, self.storedTaskSizes

    def reserveRB_Ro_Ro_1(self, RBnumber1, PW1, size1, dest1):
        self.local_flag = 1
        self.chosenRB_Ro_Ro_1 = np.array([RBnumber1, PW1, size1, dest1])

    def reserveRB_Ro_AP_1(self, RBnumber1, PW1, size1, dest1):
        self.local_flag = 2
        self.chosenRB_Ro_AP_1 = np.array([RBnumber1, PW1, size1, dest1])

    def reserveRB_Ro_Ro_2(self, RBnumber1, PW1, size1, dest1, RBnumber2, PW2, size2, dest2):
        self.local_flag = 3
        self.chosenRB_Ro_Ro_2 = np.array([RBnumber1, PW1, size1, dest1, RBnumber2, PW2, size2, dest2])

    def reserveRB_Ro_AP_2(self, RBnumber1, PW1, size1, dest1, RBnumber2, PW2, size2, dest2):
        self.local_flag = 4
        self.chosenRB_Ro_AP_2 = np.array([RBnumber1, PW1, size1, dest1, RBnumber2, PW2, size2, dest2])

    def recallRB_Ro_Ro_1(self):
        return int(self.chosenRB_Ro_Ro_1[0]), int(self.chosenRB_Ro_Ro_1[1]), int(self.chosenRB_Ro_Ro_1[2]), int(self.chosenRB_Ro_Ro_1[3])

    def recallRB_Ro_Ro_2(self):
        return int(self.chosenRB_Ro_Ro_2[0]), int(self.chosenRB_Ro_Ro_2[1]), int(self.chosenRB_Ro_Ro_2[2]), int(self.chosenRB_Ro_Ro_2[3]), int(self.chosenRB_Ro_Ro_2[4]), int(self.chosenRB_Ro_Ro_2[5]), int(self.chosenRB_Ro_Ro_2[6]), int(self.chosenRB_Ro_Ro_2[7])

    def recallRB_Ro_AP_1(self):
        return int(self.chosenRB_Ro_AP_1[0]), int(self.chosenRB_Ro_AP_1[1]), int(self.chosenRB_Ro_AP_1[2]), int(self.chosenRB_Ro_AP_1[3])

    def recallRB_Ro_AP_2(self):
        return int(self.chosenRB_Ro_AP_2[0]), int(self.chosenRB_Ro_AP_2[1]), int(self.chosenRB_Ro_AP_2[2]), int(self.chosenRB_Ro_AP_2[3]), int(self.chosenRB_Ro_AP_2[4]), int(self.chosenRB_Ro_AP_2[5]), int(self.chosenRB_Ro_AP_2[6]), int(self.chosenRB_Ro_AP_2[7])

    def setFlag(self, Portions, typeOfComm):
        self.nOfPortions = Portions
        self.typeOfcomm = typeOfComm

    def numOfPortions(self):
        return self.nOfPortions, self.typeOfcomm

    def perform_task_Robot(self, alloCPU, portionSize):
        procDelay = portionSize * 1000 * self.XI/ self.robotCPU  # here I multiplied by 1000 since I consider the task size with mili second basis
        procEnergy = portionSize * 1000 * self.PHI  # here I multiplied by 1000 since I consider the task size with mili second basis
        self.robotBattery -= procEnergy * procDelay  # update the battery value
        self.robotCPU -= alloCPU  # update the cpu
        return procDelay, procEnergy, self.robotCPU, self.robotBattery

    def local_flag(self):
        return self.local_flag

    def chosenRBPW(self, local_flag):
        if local_flag == 1:
            return self.chosenRB_Ro_Ro_1[0], self.chosenRB_Ro_Ro_1[1]
        elif local_flag == 2:
            return self.chosenRB_Ro_AP_1[0], self.chosenRB_Ro_AP_1[1]
        elif local_flag == 3:
            return self.chosenRB_Ro_Ro_2[0], self.chosenRB_Ro_Ro_2[1], self.chosenRB_Ro_Ro_2[4], self.chosenRB_Ro_Ro_2[5]
        elif local_flag == 4:
            return self.chosenRB_Ro_AP_2[0], self.chosenRB_Ro_AP_2[1], self.chosenRB_Ro_AP_2[4], self.chosenRB_Ro_AP_2[5]
        else:
            pass
        return

    def successfulUpload(self):
        self.UploadOK = 1

    def uploadOK(self):
        return self.UploadOK

    def genTask_Robot(self):
        self.taskAoI = 10
        self.storedTaskCPUs = 0
        self.storedTaskSizes = 0
        self.robotBattery = np.random.uniform(100, 600)  # Wh  for more realistic scenario I consider very low battery as well
        self.robotCPU = np.random.uniform(100.e6, 200e6) # Hz I consider very low CPU
        self.taskSize = np.random.uniform(10.e3, 150.e3)  # Bytes  Achtung!! be careful about these task sizes, since they are considered based on time slot (ms).
        self.taskCPU = np.random.uniform(100.e6, 200.e6)  # Hz
        self.taskTargetDelay = np.random.uniform(0.01, 0.05)  # ms
        self.taskGamma = np.random.uniform(0, 1)  # Gamma is the trade-off between time-sensitivity and data importance




