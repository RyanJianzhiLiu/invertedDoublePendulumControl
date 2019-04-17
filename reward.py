import numpy as np
import mujoco_py as mujoco

class RewardFunctionTerminalPenalty():
    def __init__(self, aliveBouns, deathPenalty, isTerminal):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        reward = self.aliveBouns 
        if self.isTerminal(state):
            reward = self.deathPenalty
        return reward

class RewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        reward = self.aliveBouns 
        return reward

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class RewardFunctionCompete():
    def __init__(self, aliveBouns, catchReward, disDiscountFactor, minXDis):
        self.aliveBouns = aliveBouns
        self.catchReward = catchReward
        self.disDiscountFactor = disDiscountFactor
        self.minXDis = minXDis
    def __call__(self, state, action):
        pos0 = state[0][2:4]
        pos1 = state[1][2:4]
        distance = euclideanDistance(pos0, pos1)

        if distance <= 2 * self.minXDis:
            catchReward = self.catchReward
        else:
            catchReward = 0

        distanceReward = self.disDiscountFactor * distance

        reward = np.array([distanceReward - catchReward, -distanceReward + catchReward])
        # print("reward", reward)
        return reward

class CartpoleRewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        distanceBonus = (0.21 - abs(state[2])) / 0.21 + (2.4 - abs(state[0])) / 2.4  
        reward = self.aliveBouns + distanceBonus
        return reward


class InvDblPendulumRewardFunction:
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
        modelName = 'inverted_double_pendulum'
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward = self.deathPenalty

        oldQPos = state[0: self.numQPos]
        oldQVel = state[self.numQPos: self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = action
        self.simulation.forward()

        x, _, y = self.simulation.data.site_xpos[0]
        distPenalty = 0.01 * x ** 2 + (y - 2) ** 2

        v1, v2 = self.simulation.data.qvel[1:3]
        velPenalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2

        return reward - distPenalty - velPenalty