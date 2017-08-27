import chainer
from chainer import cuda,Function,Variable
from chainer import optimizers,serializers
from chainer import Chain,ChainList
from chainer.initializers import *
import chainer.functions as F
import chainer.links as L
import numpy as np
import gym
from copy import deepcopy
import random

class Model(Chain):
    def __init__(self):
        super(Model,self).__init__(
            l1 = L.Linear(4,50,initialW=HeNormal()),
            l2 = L.Linear(50,2,initialW=Zero()),
        )

    def __call__(self,x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y

class ReplayMemory:
    def __init__(self,size):
        self.size = size
        self.count = 0
        self.ReplayMemory = [None for i in range(size)]
    def add(self,s,a,r,s_dash,done):
        self.ReplayMemory[self.count%self.size] = [s,a,r,s_dash,done]
        self.count += 1
    def sample(self,num):
        return random.sample(self.ReplayMemory[0:min(self.size,self.count)],num)

env = gym.make("CartPole-v0")
epsilon = 0.3
gamma = 0.95
time = 0
step = 0
memory_size = 10**4
replay_size = 32
initial_exploration = 500

model=Model()
optimizer=optimizers.Adam()
optimizer.setup(model)
target_model = deepcopy(model)
Memory = ReplayMemory(memory_size)

for i in range(1000):
    obs = env.reset()
    obs = obs.astype(np.float32).reshape((1,4))
    last_obs = deepcopy(obs)
    done = False
    time2 = 0
    while not done:
        env.render()
        time += 1
        time2 += 1
        step += 1
        Q = model(Variable(obs))
        if epsilon > np.random.rand() or step < initial_exploration:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.data[0])
        epsilon -= 1e-4
        if epsilon < 0.:
            epsilon = 0.
        obs, reward, done, _ = env.step(action)
        obs = obs.astype(np.float32).reshape((1,4))
        reward = 0.
        if done and not time2 >= 200:
            reward = -1.
        Memory.add(last_obs,action,reward,obs,done)

        if step < initial_exploration:
            continue
        sample = Memory.sample(replay_size)
        t = np.ndarray(shape=(replay_size,2), dtype=np.float32)
        state = np.ndarray(shape=(replay_size,4), dtype=np.float32)
        for j,s in enumerate(sample):
            state[j] = s[0]

        model.cleargrads()
        q = model(state)
        t = deepcopy(q.data)
        for j,s in enumerate(sample):
            if s[4]:
                t[j][s[1]] = s[2]
            else:
                Q_dash = target_model(s[3])
                t[j][s[1]] = s[2] + gamma * np.max(Q_dash.data[0])

        td = Variable(t) - q
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        zero_val = Variable(np.zeros((replay_size, 2), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)
        loss.backward()
        optimizer.update()

        last_obs = deepcopy(obs)
        if time % 10 == 0:
            target_model = deepcopy(model)

    if i % 10 == 0:
        print 'episode:',i,'step:',step,'epsilon:',epsilon,'ave:',time/10.,'Q:',Q.data
        time = 0.
