import random
import numpy as np
import copy
import pylab
from time import sleep
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from GameBoard import Runner, GridCell, Base, Player, GameScene

GAMES = 1000
FRAMES = 500


class DQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.env = env
        print(self.state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            idx = random.randrange(self.action_size)
            action = [0, 0, 0, 0]
            action[idx] = 1
            return np.array([action]), 'random'
        act_values = self.model.predict(state)
        # print(act_values)
        # return act_values[0][0], 'predicted'
        return act_values[0], "predicted"

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:

            # state = np.reshape(state, (1,self.env.max_x+1,self.env.max_y+1))
            # next_state = np.reshape(next_state, (1,self.env.max_x+1,self.env.max_y+1))
            target = self.model.predict(state)
            if done:
                target[0][np.argmax(action)] = reward
            else:
                a_vals = self.model.predict(next_state)

                a = a_vals[0]  # [0]

                t_vals = self.target_model.predict(next_state)
                t = t_vals[0]  # [0]

                target[0][np.argmax(action)] = reward + self.gamma * t[np.argmax(a)]

            self.model.fit(state, target, epochs=10, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def game_done(game, GAMES, frame, agent, ep_rew, cum_rew, all_r):
    all_r.append(ep_rew)
    best = max(all_r)
    agent.update_target_model()
    print("Game: {}/{}, Frame: {}, Epsilon: {:.2} Game Reward: {} Highest: {}, Game {}"
          .format(game, GAMES, frame, agent.epsilon, ep_rew, best, all_r.index(best) + 1))
    # print("Cumulative Episode Reward: {0}".format(ep_rew)

    cum_rew += ep_rew


if __name__ == "__main__":
    env = Runner(static=False, nb_frames=FRAMES, player_count=1)

    state_size =   (env.max_x+1)*(env.max_y+1)#np.zeros(1260).reshape((env.max_x+1, env.max_y+1))


    action_size = 4
    agent = DQNAgent(state_size, action_size, env)
    try:
        agent.load("./saved-models/my-ddqn-new3.h5")
        print("loaded model")
    except:
        pass
    all_r = []
    done = False
    batch_size = 32
    cum_rew = 0

    for game in range(1,GAMES+1):
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if game % 100 == 0:
                print('Agent Model saved')
                agent.save("./saved-models/my-ddqn-new3.h5")
        breaker = False
        #print("Game: {0} Cumulative Reward: {1}".format(game, cum_rew))
        ep_rew = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        #state = np.reshape(state, (env.max_x+1, env.max_y+1))
        for frame in range(1,FRAMES+1):
            env.render()
            action, method = agent.act(state)
            env.last_action = method
            if method == 'predicted':
                #print(str( action))
                env.pred_list.append(str(action))

            env.action_history.append(env.last_action)
            reward, next_state, done = env.frame_step(action)
            reward = np.clip(reward, -1, 1)

            ep_rew += reward
            next_state = np.reshape(next_state, [1, state_size])
            #next_state = np.reshape(next_state, (env.max_x+1, env.max_y+1))

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('Game done')
                game_done(game, GAMES, frame, agent, ep_rew, cum_rew, all_r)
                breaker = True
                break
        if breaker: continue


    print("Total Cumulative Reward: {0}".format(cum_rew))







