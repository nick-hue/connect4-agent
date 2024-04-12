import random

class Player:
    def __init__(self, name:str, turn:int) -> None:
        self.name = name
        self.turn = turn

    def move(self, board_arr) -> None:
        raise NotImplementedError("Implement by child class [Player].")

class BotPlayer(Player):
    def __init__(self, name: str, turn: int) -> None:
        super().__init__(name, turn)

    '''
    return the column where the bot player 
    '''
    def move(self, board_arr) -> int:
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        available_moves = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        return random.choice(available_moves) 