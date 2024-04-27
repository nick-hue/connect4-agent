import random
import numpy as np
from collections import deque 
import tensorflow as tf
from kaggle_environments import evaluate, make
#from hiddens import *  # This should include definitions for hl1_w, hl1_b, etc.
#from logger import load_tup

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