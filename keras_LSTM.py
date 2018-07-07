# Developed by Nathan Shepherd

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('ggplot')

class DQN:
    def __init__(self, obs_space, num_actions, observation):

        self.prev_action = 0
        self.obs_space = obs_space + 1
        self.num_actions = num_actions

        self.define_model()
        

    def define_model(self, hidden=[128, 64], batch_size=1, look_back=1):
        #Source: https://bit.ly/2ElxXHE
        self.model = Sequential()
        self.model.add(LSTM(hidden[0],
                            stateful=True, return_sequences=True,
                            batch_input_shape=(batch_size,look_back,self.obs_space)))
        self.model.add(LSTM(hidden[1],
                            stateful=True, # State must be reset after every epoch
                            batch_input_shape=(batch_size,look_back,self.obs_space)))
        self.model.add(Dense(1, activation='tanh'))
        self.model.compile(loss='mse', optimizer='adam')

    def reshape(self, state):
        s = np.append(state, self.prev_action)
        s = np.reshape(s, (1, 1, self.obs_space))
        return s
        
    def get_action(self, state):
        actions = []
        for a in range(self.num_actions):
            self.prev_action = a
            _state = self.reshape(state)
            actions.append(self.model.predict(_state))
        action = np.argmax(actions)
        self.prev_action = action
        return action

    def evaluate_utility(self, state):
        actions = []
        for a in range(self.num_actions):
            self.prev_action = a
            _state = self.reshape(state)
            actions.append(self.model.predict(_state))
        value = max(actions)
        return value

    def update_policy(self, state, state_next, action, reward):
        state_value = self.model.predict(self.reshape(state))
        
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA*(reward + GAMMA * reward_next - state_value)

        self.prev_action = action
        state = self.reshape(state)
        
        self.model.fit(state, state_value, epochs=1,
                       batch_size=1, verbose=0, shuffle=False)
        

        #state = ''.join(str(int(elem)) for elem in self.digitize(state))
        #self.Q[state][action] = state_value
        #%#%#%#%#%
    
def play_episode(agent, act_space, epsilon=.2, viz=False):
    state = env.reset()
    total_reward = 0
    terminal = False
    num_frames = 0

    max_rwd = -200
    while not terminal:
        if viz: env.render()
        #if num_frames > 300: epsilon = 0.1

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)
        
        state_next, reward, terminal, info = env.step(action)

        total_reward += reward
        
        if terminal:
            #if num_frames > 150:
            #    reward += np.log(num_frames)
        
            if  num_frames < 200:
                reward = -300
        
        agent.update_policy(state, state_next, action, reward)
              
        state = state_next
        num_frames += 1
        
    agent.model.reset_states()
    
    return total_reward, num_frames

def train(obs_space, act_space=None,epochs=2000, obs=False, agent=False):
    if not agent: agent = DQN(obs_space, act_space, env.reset())

    stacked_frames = []
    #TODO: Plot reward averages
    rewards = [0]
    for ep in range(epochs):
        epsilon = max(EPSILON_MIN, np.tanh(-ep/(epochs/2))+ 1)

        ep_reward, num_frames = play_episode(agent, act_space, epsilon, viz=obs)
        if ep % 5 == 0:
            print("Ep: {} | {}".format(ep, epochs),
                  "%:", round(ep*100/epochs, 2),
                  "Epsilon:", round(epsilon, 4),
                  "Avg rwd:", round(np.mean(rewards),3),
                  "Ep rwd:", round(ep_reward, 3))

        stacked_frames.append(num_frames)
        rewards.append(ep_reward)

    return rewards, stacked_frames, agent

def observe(agent, N=15):
    [play_episode(agent, -1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def play_random(viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward

        #if terminal and num_frames < 200:
         #   reward = -300
        
    return total_reward

gym.envs.register(
    id='CartPoleExtraLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=250,
    reward_threshold=-110.0,
)
env = gym.make('CartPoleExtraLong-v0')
#env = gym.make('CartPole-v0')

EPSILON_MIN = 0.1
ALPHA = 0.01
GAMMA = 0.9

EPOCHS = 2000

obs_space = 4
observe_training = False
action_space = env.action_space.n

'''
TODO:
    Add frame buffer for sequence prediction
'''


if __name__ == "__main__":
    episode_rewards, _, Agent = train(obs_space,
                                      act_space = action_space,
                                      obs = observe_training,
                                      epochs = EPOCHS)
    
    random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()

    observe(Agent)


