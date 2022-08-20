
# Reference: https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
import os
import datetime

import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from gym.wrappers import GrayScaleObservation
from matplotlib.animation import FuncAnimation

import gym

from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, Embedding, Reshape, Activation, Flatten
from tensorflow.keras.optimizers import Adam

dt_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
performance_dir = f"performance/{dt_now}/"

batch_size = 32
num_of_episodes = 5000
timesteps_per_episode = 1000000

class Create_recording:
    def __init__(self, interval=40):
        self.interval = interval

    def update_scene(self, num, frames, patch):
        patch.set_data(frames[num])
        return (patch,)

    def plot_animation(self, frames, repeat=False):
        plt.close()
        fig = plt.figure()
        patch = plt.imshow(frames[0])
        plt.axis("off")
        return FuncAnimation(
            fig,
            self.update_scene,
            fargs=(frames, patch),
            frames=len(frames),
            repeat=repeat,
            interval=self.interval,
        )

    def save_animation(self, anim, filename):
        writergif = animation.PillowWriter(fps=30)
        anim.save(filename, writer=writergif)

    def record_agent_playing(self, agent, progress):

        frames = []
        max_game_moves = 1000

        game = gym.make("Boxing-v4")
        game = GrayScaleObservation(env=game, keep_dim=True)
        current_state = game.reset()

        current_state = preprocess_observation(current_state)

        for step in range(max_game_moves):
            # Take best action recommended by online DQN
            action = agent.act(current_state)

            new_state, reward, done, _ = game.step(action)
            new_state = preprocess_observation(new_state)

            frame = game.render(mode="rgb_array")
            frames.append(frame)

            if done:
                break
        video = self.plot_animation(frames)
        name = f"dqn_learning_at_{progress}_episodes.gif"
        self.save_animation(video, name)

def plot(
    indx,
    typ,
    rewards, 
    # losses: List[float], 
    # epsilons: List[float],
):
    """Plot the training progresses."""
    if not os.path.isdir(performance_dir):
        os.makedirs(performance_dir)

    # clear_output(True)
    
    title = f"{typ} {indx} score"
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(title)
    plt.plot(rewards)
    # plt.subplot(132)
    # plt.title('loss')
    # plt.plot(losses)
    # plt.subplot(133)
    # plt.title('epsilons')
    # plt.plot(epsilons)

    # plt.show()
    plt.savefig(f"{performance_dir}{title}.png")


def preprocess_observation(observation):
    img = observation[30:-30:2, ::2]
    # img = img.mean(axis=2) #to grayscale
    # img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(75, 80, 1)


class Agent:
    def __init__(self, enviroment, optimizer):
        
        self.enviroment = enviroment

        # Initialize atributes
        #self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model((75, 80, 1))
        self.target_network = self._build_compile_model((75, 80, 1))
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self, input_shape):        
        
        input = Input(shape=input_shape)
        x = Dense(128)(input)
        x = Activation("relu")(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = Dense(128)(x)
        x = Flatten()(x)
        x = Dense(self._action_size)(x)

        model = Model(inputs=input, outputs=x)

        model.compile(loss='mse', optimizer=self._optimizer)
        return model
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.enviroment.action_space.sample()
        
        q_values = self.q_network.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            next_state = preprocess_observation(next_state)
            
            target = self.q_network.predict(np.array([state]), verbose=0)     
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(np.array([next_state]), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(np.array([state]), target, epochs=1, verbose=0)


def undertake_learning(agent):
    rewards_per_episode_array = []

    for e in range(0, num_of_episodes):
        if (e == 0 or e % 10 == 0):
            Create_recording().record_agent_playing(agent, e)

        # Reset the enviroment
        state = agent.enviroment.reset()
        state = np.reshape(state, [210, 160, 1])
        
        # Initialize variables
        reward = 0
        terminated = False
        
        episode_rewards = 0
        for timestep in range(timesteps_per_episode):
            state = preprocess_observation(state)
            # Run Action
            action = agent.act(state)
            
            # Take action    
            next_state, reward, terminated, info = agent.enviroment.step(action) 
            # next_state = np.reshape(next_state, [1, 1])
            agent.store(state, action, reward, next_state, terminated)
            
            state = next_state

            episode_rewards += reward
            
            if terminated:
                agent.align_target_model()
                break
            
            # original logic: if len(agent.expirience_replay) > batch_size:
            if len(agent.expirience_replay) > batch_size and timestep % 10 == 0 or terminated:
                agent.retrain(batch_size)
            
            if timestep%10 == 0:
                print(f"episode {e} @ {timestep/timesteps_per_episode * 100} %")

        rewards_per_episode_array.append(episode_rewards)
        if e != 0:
            plot(e, "episodes", rewards_per_episode_array)
        
        if (e + 1) % 10 == 0:
            print("**********************************")
            print("Episode: {}".format(e + 1))
            agent.enviroment.render()
            print("**********************************")


if __name__ == "__main__":
    enviroment = gym.make("Boxing-v4").env
    enviroment = GrayScaleObservation(env=enviroment, keep_dim=True)
    enviroment.reset()
    # enviroment.render()

    print('Number of states: {}'.format(enviroment.observation_space.shape))
    print('Number of actions: {}'.format(enviroment.action_space.n))

    optimizer = Adam(learning_rate=0.01)
    agent = Agent(enviroment, optimizer)

    batch_size = 32
    num_of_episodes = 5000
    timesteps_per_episode = 1000000
    agent.q_network.summary()
    
    undertake_learning(agent)