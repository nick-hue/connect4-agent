import random
import numpy as np
import torch
from collections import deque 
import copy

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
    return the column where the bot player 
    '''
    def move(self, board_arr) -> int:
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        available_moves = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        return random.choice(available_moves) 
    
class RLBotPlayer(Player):
    def __init__(self, name: str, turn: int, player_number:int) -> None:    
        super().__init__(name, turn, player_number)

        self.activation = torch.nn.ReLU
        self.model = self.initialize_model()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.gamma = 0.9
        self.epsilon = 0.2

        self.losses = []
        self.current_sqars = [None, None, None, None, None]

        self.rewards_values = {
            'draw': 5,
            'win': 10,
            'loss': -10,
            'move': 0
        }

    def initialize_model(self):
        input_n = 84
        hidden_n = 150
        hidden_n_2 = 100
        output_n = 7

        model = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n),
            self.activation,
            torch.nn.Linear(hidden_n, hidden_n_2),
            self.activation,
            torch.nn.Linear(hidden_n_2, output_n)
        )

        return model

    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)

        reward = self.get_reward(result)

        with torch.no_grad():
            new_q = self.model(new_state)
        max_q = torch.max(new_q)

        Y = reward if result is not None else reward + (self.gamma*max_q)
        X = current_state_q_val 
        loss = self.loss_fn(X,Y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def get_state_array(self, piece_arrays: dict):
        state_array = np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]])
        return state_array
    
    def process_state(self, curr_state:np.array):
        num_cells = curr_state[0].shape[0]*curr_state[0].shape[1]
        curr_state = curr_state.reshape(1, num_cells*2)
        if isinstance(self.activation, torch.nn.ReLU):
            curr_state += np.random.rand(1,num_cells*2)/10.0

        state = torch.from_numpy(curr_state).float().to(device=device)
        return state
    
    def move(self, piece_arrays: dict) -> int:
        curr_state = self.get_state_array(piece_arrays=piece_arrays)
        state = self.process_state(curr_state=curr_state)
        q_vals = self.model(state)
        q_vals_ = q_vals_.data.cpu().numpy()

        if random.random() < self.epsilon:
            action_ = np.random.randint(0,7)
        else:
            action_ = np.argmax(q_vals)
        self.current_sqars[0] = state
        self.current_sqars[1] = q_vals
        self.current_sqars[2] = action_

        return action_
    
    def get_reward(self, move_result):
        if move_result is None:
            reward = self.rewards_values['move']
        elif 'draw' in move_result:
            reward = self.rewards_values['draw']
        else:
            win_side = int(move_result.split("_")[-1])
            reward = self.rewards_values['win'] if win_side == self.turn else self.rewards_values['loss']
        self.current_sqars[3] = reward
        # self.rewards.append(reward)

class RLBotPlayerDDQN(RLBotPlayer):
    def __init__(self, name: str, turn: int, player_number: int) -> None:
        super().__init__(name, turn, player_number)
        self.memory = deque(maxlen=1000)
        self.batch_size = 200


        self.target_sync_freq = 200
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, new_piece_arrays, result): # might need to get the state here
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)
        self.current_sqars[4] = new_state

        reward = self.get_reward(result)

        curr_experience = (
            self.current_sqars[0],
            self.current_sqars[2],
            self.current_sqars[3],
            self.current_sqars[4],
            int(result is not None)
        )

        self.memory.append(curr_experience)

        if len(self.memory) <= self.batch_size:
            return None
        
        minibatch = random.sample(self.memory, self.batch_size)
        s_batch = torch.cat([s for (s,a,r,s2,d) in minibatch]).to(device=device)
        a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch]).to(device=device)
        r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch]).to(device=device)
        s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch]).to(device=device)
        d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch]).to(device=device)

        q1 = self.model(s_batch)

        with torch.no_grad():
            new_q = self.target_model(s2_batch)

        max_q = torch.max(new_q, dim=1)

        Y = r_batch + self.gamma*((1-d_batch)*max_q[0])
        X = q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(X,Y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        if self.n_moves % self.target_sync_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
