import random
import numpy as np
from collections import deque 
import tensorflow as tf
from kaggle_environments import evaluate, make
from hiddens import *  # This should include definitions for hl1_w, hl1_b, etc.
from logger import load_tup

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

class DQNAgent(Player):
    def __init__(self, name: str, turn: int, player_number:int) -> None:
        super().__init__(name, turn, player_number)
        self.hiddens = load_tup('arrays.npz')
        
    def predict(self, observation, configuration):
        state = list(observation['board'])  # This should be the flattened game board
        state.append(observation['mark'])   # This should be the player's mark
        out = np.array(state, dtype=np.float32)

        hl1_w, hl1_b, hl2_w, hl2_b, hl3_w, hl3_b, hl4_w, hl4_b, ol_w, ol_b = self.hiddens

        # Matrix multiplication with weights and addition of biases, followed by activation
        out = np.matmul(out, hl1_w) + hl1_b
        out = 1 / (1 + np.exp(-out))  # Sigmoid activation
        out = np.matmul(out, hl2_w) + hl2_b
        out = 1 / (1 + np.exp(-out))
        out = np.matmul(out, hl3_w) + hl3_b
        out = 1 / (1 + np.exp(-out))
        out = np.matmul(out, hl4_w) + hl4_b
        out = 1 / (1 + np.exp(-out))
        out = np.matmul(out, ol_w) + ol_b

        # Prevent moves in filled columns
        for i in range(configuration['columns']):
            if observation['board'][i] != 0:
                out[i] = -1e7

        return int(np.argmax(out))