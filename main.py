import gym
import numpy as np
import random, json, sys, time
from itertools import count
from wrappers import make_env
from network import QNetwork
from memory import ReplayMemory



class DQNAgent():
    def __init__(self, game_name = "pong"):
        self.EPS = 1
        self.EPS_MIN = 0.01
        self.EPS_DECAY = 0.995
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.TARGET_UPDATE_C = 1000
        self.MEMORY_CAPACITY_N = 1000
        self.episode_M = 5000
        self.GAME_ENV = {
            "pong" : "PongNoFrameskip-v4",
            "cartpole" : "CartPole-v0",
        }
        self.game_name = game_name
        env_name = self.GAME_ENV[self.game_name]
        self.env = make_env(gym.make(env_name))
        self.reward_list = []
        self.Qnetwork = QNetwork(self.env.action_space.n)
        self.Qnetwork.summary()

    def selectAction(self, eval_Qnetwork, state):
        """ 使用epsilon-greedy策略选择动作 
            使用神经网络近似值函数
        """
        if random.random() <= self.EPS :
            action = self.env.action_space.sample()
        else :
            action = np.argmax(eval_Qnetwork.predict([np.ones((1, self.env.action_space.n)), np.expand_dims(state, 0)]))    # (1, 84, 84, 4) 增加一个维度 - batch_size
        return action

    def updateQNetwork(self, eval_Qnetwork, target_Qnetwork, sample_batch, double=True):
        """ DDQN/DQN 梯度下降更新网络 """
        # states = np.array([a[0] for a in sample_batch])
        # actions = np.array([a[1] for a in sample_batch])
        # rewards = np.array([a[2] for a in sample_batch])
        # next_states = np.array([a[3] for a in sample_batch])
        # dones = np.array([a[4] for a in sample_batch])
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(len(sample_batch)):
            states.append(sample_batch[i][0])
            actions.append(sample_batch[i][1])
            rewards.append(sample_batch[i][2])
            next_states.append(sample_batch[i][3])
            dones.append(sample_batch[i][4])
        states = np.array(states, copy=False)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states, copy=False)
        dones = np.array(dones)
        ones_mat = np.ones((len(sample_batch), self.env.action_space.n))
        if double == True :
            eval_actions = np.argmax(eval_Qnetwork.predict([ones_mat, next_states]), axis=1)
            target_action_Qvalue = target_Qnetwork.predict([ones_mat, next_states])[range(len(sample_batch)), eval_actions]
        else :
            target_action_Qvalue = np.max(target_Qnetwork.predict([ones_mat, next_states]), axis=1)
        # y_true = eval_Qnetwork.predict(states)
        # y_true[range(len(y_true)), actions] = rewards + (1-dones)*self.GAMMA * target_action_Qvalue
        select_actions = np.zeros((len(sample_batch), self.env.action_space.n))
        select_actions[range(len(sample_batch)), actions] = 1
        y_true = rewards + (1-dones)*self.GAMMA * target_action_Qvalue
        eval_Qnetwork.fit(
            x=[select_actions, states], y=select_actions * np.expand_dims(y_true, axis=1), 
            epochs=1, batch_size=len(sample_batch), verbose=0
        )

    def dqnTrain(self, double=True):
        step = 0
        memory = ReplayMemory(self.MEMORY_CAPACITY_N)
        eval_Qnetwork = QNetwork(self.env.action_space.n)
        target_Qnetwork = QNetwork(self.env.action_space.n)
        eval_Qnetwork.set_weights(self.Qnetwork.get_weights())
        target_Qnetwork.set_weights(eval_Qnetwork.get_weights())
        reward_list = self.reward_list
        time_start = time.time()
        for episode in range(1, self.episode_M+1):
            episode_reward = 0
            obs = self.env.reset()
            state = np.array(obs)
            for t in count():
                step += 1
                action = self.selectAction(eval_Qnetwork, state)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                next_state = np.array(obs)
                memory.add((state, action, reward, next_state, done))
                state = next_state
                if len(memory) > self.BATCH_SIZE:
                    sample_batch = memory.sample(self.BATCH_SIZE)
                    self.updateQNetwork(eval_Qnetwork, target_Qnetwork, sample_batch, double)
                    self.EPS = self.EPS*self.EPS_DECAY if self.EPS > self.EPS_MIN else self.EPS_MIN
                if step % self.TARGET_UPDATE_C == 0:
                    target_Qnetwork.set_weights(eval_Qnetwork.get_weights())
                if done :
                    break
            reward_list.append(episode_reward)
            if episode % 5 == 0:
                print("episode {}. recent 5 episode_reward:{}. using {} min. total step: {}. ".format(episode, self.reward_list[-5:], (time.time()-time_start)/60, step))
            if episode % 100 == 0:
                self.save(target_Qnetwork, reward_list)
        self.Qnetwork.set_weights(target_Qnetwork.get_weights())
        self.reward_list = reward_list
        return target_Qnetwork, reward_list

    def load(self, filename_prefix=None):
        if filename_prefix == None :
            filename_prefix="pong/data/ddqn_bs"+str(self.BATCH_SIZE)
        self.Qnetwork = keras.models.load_model(filename_prefix+"network.h5")
        with open(filename_prefix+"reward.json", 'r') as file_obj:
            self.reward_list = json.loads(file_obj.read())

    def save(self, Qnetwork=None, reward_list=None, filename_prefix=None):
        if Qnetwork == None :
            Qnetwork=self.Qnetwork
        if reward_list == None :
            reward_list=self.reward_list
        if filename_prefix == None :
            filename_prefix="pong/data/ddqn_bs_"+str(self.BATCH_SIZE)
        Qnetwork.save(filename_prefix+"network.h5")
        with open(filename_prefix+"reward.json", 'w') as file_obj:
            file_obj.write(json.dumps(reward_list)) 

    def playByQv(self, Qnetwork=None, episode_num=1):
        if Qnetwork == None :
            Qnetwork=self.Qnetwork
        for episode in range(1, episode_num+1):
            obs = self.env.reset()
            state = np.array(obs)
            while True:
                self.env.render()
                action = np.argmax(eval_Qnetwork(np.expand_dims(state, 0)))
                obs, reward, done, _ = self.env.step(action)
                state = np.array(obs)
                time.sleep(0.02)
                if done :
                    break

    def plotReward(self, reward_list=None):
        if reward_list == None :
            reward_list=self.reward_list


def main():
    agent = DQNAgent()
    agent.dqnTrain()
    # agent.play(network)

if __name__ == "__main__":
    main()