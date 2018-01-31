import copy
import pylab
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
from GameBoard import Runner, GridCell, Base, Player, GameScene

EPISODES = 250000


# this is REINFORCE Agent for GridWorld
class ReinforceAgent:
    def __init__(self, load=False, nb_goals=5, gs=None):
        self.load_model = load
        # actions which agent can do
        self.action_space = [0, 1, 2, 3]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.number_goals = nb_goals
        self.state_size = gs
        self.layer_size = 33#int(self.state_size * 0.625)
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            try:
                self.model.load_weights('./saved-models/reinforce_trained_{}.h5'.format(self.layer_size))
                print("model loaded")
            except:
                pass

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.layer_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.layer_size*2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # get action from policy network
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


def set_player_score(env, score, steps):
    env.scene.players[0].score = score
    env.scene.tick_count = steps


if __name__ == "__main__":
    nb_goals = 4
    env = Runner(static=False, nb_goals=nb_goals, player_count=1, wrap=False)
    gs = 28#len(env.scene.grid) * 3
    agent = ReinforceAgent(load=True, nb_goals=nb_goals, gs=gs)

    global_step = 0
    scores, episodes = [], []
    total_score = 0

    for e in range(EPISODES):
        done = False
        score = 0
        # fresh env
        state = env.reset()
        state = np.pad(state, [0, gs - (len(state))], mode='constant')
        state = np.reshape(state, [1, len(state)])
        steps = 0
        pos = env.scene.players[0].coords

        while not done:
            global_step += 1
            steps += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            if pos == env.scene.players[0].coords:
                #reward -= 1
                pass
            pos = env.scene.players[0].coords
            next_state = np.pad(next_state, [0, gs - (len(next_state))], mode='constant')
            next_state = np.reshape(next_state, [1, len(next_state)])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)
            set_player_score(env, score, steps)
            if score < -100 or steps > 1000:
                done = True
            if done:
                # update policy neural network for each episode
                agent.train_model()
                total_score += score
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                best = max(scores)
                print("episode: {}  score: {}  steps: {}  best:{}  total_steps: {} ".format(e,
                                                                                            score,
                                                                                            steps,
                                                                                            best,
                                                                                            global_step))

        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./results/reinforce.png")
            agent.model.save_weights("./saved-models/reinforce_trained_{}.h5".format(agent.layer_size))
            print("model saved...")
