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
from annoy import AnnoyIndex

class Model(Chain):
    def __init__(self):
        super(Model,self).__init__(
            l1 = L.Linear(4,50,initialW=HeNormal()),
            l2 = L.Linear(50,50),
        )
    def __call__(self,x):
        h = F.relu(self.l1(x))
        embedding = self.l2(h)
        return embedding

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

class DND_:
    def __init__(self,memory_size,embedding_size,search_n):
        self.size = memory_size
        self.embedding_size = embedding_size
        self.count = 0
        self.n = search_n
        self.dnd = AnnoyIndex(embedding_size)
        self.x = None
        self.embeddings = [None for i in range(self.size)]
        self.values = [None for i in range(self.size)]
        self.delta = 10e-3
    def add(self,embedding,value):
        self.dnd.add_item(self.count%self.size,embedding)
        self.embeddings[self.count%self.size] = embedding
        self.values[self.count%self.size] = value
        self.count += 1
    def q_func(self,embedding):
        self.x = embedding
        index = self.dnd.get_nns_by_vector(embedding,self.n,include_distances=False)
        k = np.array([None for i in range(self.n)],dtype=np.float32)
        for i,ind in enumerate(index):
            k[i] = 1.0 / (np.sqrt(np.linalg.norm(embedding-self.embeddings[ind]))+self.delta)
        k_sum = np.sum(k)
        Q = 0.
        for i in range(self.n):
            Q += self.values[index[i]] * (k[i] / k_sum)
        return Q
    def loss(self,loss):
        grad = np.dot(self.x.reshape(self.embedding_size,1), loss.reshape(1,2))
        loss = np.dot(grad, np.ones((grad.shape[1],1))).T
        return loss
    def rebuild(self):
        self.dnd.build(-1)

class DND:
    def __init__(self,memory_size=10**4,embedding_size=50,search_n=10,action_n=2):
        self.action_n = action_n
        self.dnd = [DND_(memory_size,embedding_size,search_n) for i in range(action_n)]
    def add(self,action,embedding,value):
        self.dnd[action].add(embedding,value)
    def q_func(self,embedding):
        Q = np.ndarray(shape=(self.action_n),dtype=np.float32)
        for i in range(self.action_n):
            Q[i] = self.dnd[i].q_func(embedding)
        return Q
    def loss(self,action,loss):
        return self.dnd[action].loss(loss)
    def rebuild(self):
        for i in range(self.action_n):
            self.dnd[i].rebuild()

env = gym.make("CartPole-v0")
epsilon = 0.3
gamma = 0.95
memory_size = 10**4
replay_size = 32
time = 0
step = 0
initial_exploration = 300

model=Model()
optimizer=optimizers.Adam()
optimizer.setup(model)
target_model = deepcopy(model)
Memory = ReplayMemory(memory_size)
DND = DND()

for i in range(1000):
    obs = env.reset()
    obs = obs.astype(np.float32).reshape((1,4))
    last_obs = deepcopy(obs)
    done = False
    time2 = 0
    while not done:
        env.render()
        time += 1
        step += 1
        embedding = model(Variable(obs))
        embedding = embedding.data[0]
        if step > initial_exploration:
            Q = DND.q_func(embedding)
        else:
            Q = 0
        if epsilon > np.random.rand() or step < initial_exploration:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)
        epsilon -= 1e-4
        if epsilon < 0.:
            epsilon = 0.
        obs, reward, done, _ = env.step(action)
        obs = obs.astype(np.float32).reshape((1,4))
        reward = 0.
        if done:
            reward = -1.
        Memory.add(last_obs,action,reward,obs,done)
        if step < initial_exploration or done:
            DND.add(action,embedding,reward)
        else:
            embedding_dash = target_model(obs)
            embedding_dash = embedding_dash.data[0]
            DND.add(action,embedding,reward + gamma * np.max(DND.q_func(embedding_dash)))
        if step % 100 == 0:
            DND.rebuild()

        if step < initial_exploration:
            continue
        sample = Memory.sample(replay_size)
        q = np.ndarray(shape=(replay_size,2), dtype=np.float32)
        t = np.ndarray(shape=(replay_size,2), dtype=np.float32)
        state = np.ndarray(shape=(replay_size,4), dtype=np.float32)
        loss = np.ndarray(shape=(replay_size,50), dtype=np.float32)
        for j,s in enumerate(sample):
            state[j] = s[0]
        model.cleargrads()
        embeddings = model(state)
        embeddings = embeddings.data
        for j,e in enumerate(embeddings):
            q[j] = DND.q_func(e)
        for j,s in enumerate(sample):
            if s[4]:
                t[j][s[1]] = s[2]
            else:
                embedding_dash = target_model(s[3])
                embedding_dash = embedding_dash.data[0]
                t[j][s[1]] = s[2] + gamma * np.max(DND.q_func(embedding_dash))
            loss[j] = DND.loss(s[1],t[j]-q[j])
        zero_val = Variable(np.zeros((replay_size, 50), dtype=np.float32))
        loss = F.mean_squared_error(loss, zero_val)
        loss.backward()
        optimizer.update()
        last_obs = deepcopy(obs)
        if time % 10 == 0:
            target_model = deepcopy(model)

    if i % 10 == 0:
        print ('episode:',i,'step:',step,'epsilon:',epsilon,'ave:',time/10.,'Q:',Q)
        time = 0.
