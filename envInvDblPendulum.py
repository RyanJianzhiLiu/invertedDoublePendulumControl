import mujoco_py as mujoco
import numpy as np
import reward

#np.random.seed(123)
class Reset():
    def __init__(self, modelName, qPosInitNoise, qVelInitNoise):
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
    def __call__(self):
        qPos = self.simulation.data.qpos + np.random.uniform(low = -self.qPosInitNoise, high = self.qPosInitNoise, size = len(self.simulation.data.qpos))
        qVel = self.simulation.data.qvel + np.random.uniform(low = -self.qVelInitNoise, high = self.qVelInitNoise, size = len(self.simulation.data.qvel))
        startState = np.concatenate([qPos, qVel])
        return startState

class TransitionFunction():
    def __init__(self, modelName, renderOn):
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
        self.renderOn = renderOn
        if self.renderOn:
            self.viewer = mujoco.MjViewer(self.simulation)
    def __call__(self, oldState, action, numSimulationFrames = 1):
        oldQPos = oldState[0 : self.numQPos]
        oldQVel = oldState[self.numQPos : self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = action

        for i in range(numSimulationFrames):
            self.simulation.step()
            if self.renderOn:
                self.viewer.render()
        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newState = np.concatenate([newQPos, newQVel])
        #print("From\n\t qpos: {}; qvel: {}\nTo\n\t qpos: {}; qvel: {}\n".format(oldQPos, oldQVel, newQPos, newQVel))
        return newState

class InvDblPendulumIsTerminal():
    def __init__(self, minHeight):
        self.minHeight = minHeight
        self.model = mujoco.load_model_from_path('xmls/inverted_double_pendulum.xml')
        self.simulation = mujoco.MjSim(self.model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
    def __call__(self, state):
        oldQPos = state[0: self.numQPos]
        oldQVel = state[self.numQPos: self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.forward()

        x, _, y = self.simulation.data.site_xpos[0]
        terminal = bool(y <= self.minHeight)
        return terminal


if __name__ == '__main__':
    modelName = "inverted_double_pendulum"
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001
    minHeight = -100
    reset = Reset(modelName, qPosInitNoise, qVelInitNoise)
    transition = TransitionFunction(modelName, renderOn=True)
    isTerminal = InvDblPendulumIsTerminal(minHeight)

    aliveBonus = 10
    deathPenalty = 10
    rewardFunc = reward.InvDblPendulumRewardFunction(aliveBonus, deathPenalty, isTerminal)

    episodeLen = 1000
    state = reset()
    for i in range(episodeLen):
        if isTerminal(state):
            break
        action = 0.05*(np.random.randn() - 0.5)
        if i % 50 == 0:
            r = rewardFunc(state, action)
            print(r)
        nextState = transition(state, action)
        state = nextState
