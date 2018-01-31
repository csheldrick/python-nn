import random
import numpy as np
import copy
import pylab

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from GameBoard import Runner

EPISODES = 1000


class DeepSARSAgent:
    def __init__(self, load=False, nb_goals=5, gs=None):
        self.load_model = load
        # actions which agent can do
        self.action_space = [0, 1, 2, 3]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.number_goals = nb_goals
        self.state_size = gs  # ( diff of goal and agent x, diff of goal and agent y, amount)
        self.layer_size = 512#28 * 9  # int(self.state_size * 0.625)
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            try:

                self.model.load_weights('./saved-models/deep_sarsa_{}.h5'.format(self.layer_size))
                print('model loaded')
                self.epsilon = 0.05
            except:
                pass

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.layer_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)


def set_player_score(env, score, steps):
    env.scene.players[0].score = score
    env.scene.tick_count = steps


def main():
    nb_goals = 4
    env = Runner(nb_goals=nb_goals, player_count=1, wrap=False, rows=10, mode="halite")
    gs = 28
    agent = DeepSARSAgent(load=True, nb_goals=nb_goals, gs=gs)

    global_step = 0
    scores, episodes = [], []
    action_space = ['u', 'd', 'l', 'r']
    total_score = 0
    last_best = 0
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        if e == 1:
            print(state)
        state = np.pad(state, [0, gs - (len(state))], mode='constant')
        state = np.reshape(state, [1, len(state)])
        steps = 0
        while not done:
            # fresh env
            global_step += 1
            steps += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.pad(next_state, [0, gs - (len(next_state))], mode='constant')
            next_state = np.reshape(next_state, [1, len(next_state)])
            next_action = agent.get_action(next_state)
            state = next_state
            if steps > 1000:
                reward -= 1
                done = True
            score += reward
            set_player_score(env, score, steps)
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = copy.deepcopy(next_state)
            if done:
                scores.append(score)
                total_score += score
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./results/deep_sarsa_.png")
                best = max(scores)
                if best > last_best:
                    last_best = best
                    print("saving model - new best")
                    agent.model.save_weights("./saved-models/deep_sarsa_{}.h5".format(agent.layer_size))

                print("episode: {}   epsilon: {:.4}  score: {}  steps: {}  best:{}  total_steps: {} ".format(e,
                                                                                                             agent.epsilon,
                                                                                                             score,
                                                                                                             steps,
                                                                                                             best,
                                                                                                             global_step))
        if e % 100 == 0 or e == EPISODES:
            print("saving model")
            agent.model.save_weights("./saved-models/deep_sarsa_{}.h5".format(agent.layer_size))
    main()


if __name__ == "__main__":
    main()
