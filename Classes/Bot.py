import random
import numpy as np
import torch
from collections import deque 
import copy
import tensorflow as tf
from kaggle_environments import evaluate, make

class Player:
    def __init__(self, name:str, turn:int, player_number:int) -> None:
        self.name = name
        self.turn = turn
        self.player_number = player_number

    def move(self, board_arr) -> None:
        raise NotImplementedError("Implement by child class [Player].")

class HumanPlayer(Player):
    def __init__(self, name: str, turn: int, player_number:int) -> None:
        super().__init__(name, turn, player_number)

    def move(self, board_arr) -> None:
        raise NotImplementedError("Implement by child class [Player].")


class BotPlayer(Player):
    def __init__(self, name: str, turn: int, player_number:int) -> None:
        super().__init__(name, turn, player_number)

    '''
    return the an empty column that the bot player randomly chooses
    '''
    def move(self, board_arr) -> int:
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        available_moves = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        return random.choice(available_moves) 
    

class DeepModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='sigmoid', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

#     @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


# num_states = env.observation_space.n + 1 # 43
# num_actions = env.action_space.n # 7
hidden_units = [100, 200, 200, 100]
gamma = 0.99
lr = 1e-2
batch_size = 32
max_experiences = 10000
min_experiences = 100

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = DeepModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # The buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

#     @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            # Only start the training process when we have enough experiences in the buffer
            return 0

        # Randomly select n experience in the buffer, n is batch-size
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # Prepare labels for training process
        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    # Get an action by using epsilon-greedy
    def get_action(self, state, epsilon):
        # if np.random.random() < epsilon:
        #     return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))
        # else:
        #     prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()
        #     for i in range(self.num_actions):
        #         if state.board[i] != 0:
        #             prediction[i] = -1e7
        #     return int(np.argmax(prediction))

        prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()
        for i in range(self.num_actions):
            if state.board[i] != 0:
                prediction[i] = -1e7
        return int(np.argmax(prediction))

    # Method used to manage the buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        ref_model = tf.keras.Sequential()

        ref_model.add(self.model.input_layer)
        for layer in self.model.hidden_layers:
            ref_model.add(layer)
        ref_model.add(self.model.output_layer)

        ref_model.load_weights(path)
    
    # Each state will consist of the board and the mark
    # in the observations
    def preprocess(self, state):
        result = state.board[:]
        result.append(state.mark)

        return result

def create_model(input_shape, num_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')  # Assuming a direct output to action space
    ])
    return model


class DQLearning:
    def __init__(self, name, turn, player_number, filepath, model):
        self.name = name
        self.turn = turn
        self.player_number = player_number

        self.model = model
        self.model.load_weights(filepath)  # Load the pre-trained model

    def predict_move(self, board_state):
        print("Original board state:", board_state)
        board_state = board_state.flatten()
        board_state = np.expand_dims(board_state, axis=0)  # Ensure input shape is correct
        print("Processed board state for prediction:", board_state)

        predictions = self.model.predict(board_state)[0]
        print("Model predictions:", predictions)

        legal_moves = (board_state[0] == 0)  # Check which columns are available for a move
        predictions[legal_moves == False] = -1e7  # Invalidate moves in full columns
        chosen_move = np.argmax(predictions)
        print("Chosen column:", chosen_move)
        return chosen_move
    
    def load_weights(self, path):
        ref_model = tf.keras.Sequential()

        ref_model.add(self.model.input_layer)
        for layer in self.model.hidden_layers:
            ref_model.add(layer)
        ref_model.add(self.model.output_layer)

        ref_model.load_weights(path)
