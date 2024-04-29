import numpy as np                
import gym
from kaggle_environments import evaluate, make
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class for our model
class DeepModel(nn.Module):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        layers = []
        input_dim = num_states
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())  # Activation function
            input_dim = hidden
        layers.append(nn.Linear(hidden_units[-1], num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# class for our Deeq Q Network
class DQN:
    def __init__(self, name: str, turn: int, player_number:int, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.name = name
        self.turn = turn
        self.player_number = player_number
        
        self.num_actions = num_actions   # number of actions + 1 (6x7)+1 = 43
        self.gamma = gamma               # 0.99

        self.batch_size = batch_size     # batch size for experience replay
        self.experience = {'s' : [], 'a' : [], 'r' : [], 's2' : [], 'done' : []} # experience replay
        self.max_experiences = max_experiences # experience replay size
        self.min_experiences = min_experiences # minimum experience replay size before training

        self.model = DeepModel(num_states, hidden_units, num_actions) # model
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr) # optimizer
        self.criterion = nn.MSELoss()                                 # loss

    # append experience to the buffer is, if the buffer is full pop the oldest one
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    # train the agent
    def train(self, TargetNet):

        # don't train before the reaching the minimum experience replay size
        if len(self.experience['s']) < self.min_experiences:
            return 0

        # Random sample batches from experience replay buffer
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        # Get the states from the samples of the experience bufferand preprocess them.
        states = torch.FloatTensor([self.preprocess(self.experience['s'][i]) for i in ids]).to(device)

        actions = torch.LongTensor([self.experience['a'][i] for i in ids]).to(device)  # Retrieve the actions
        rewards = torch.FloatTensor([self.experience['r'][i] for i in ids]).to(device) # Retrieve the rewards

        # Get the next states from the samples of the experience bufferand preprocess them.
        next_states = torch.FloatTensor([self.preprocess(self.experience['s2'][i]) for i in ids]).to(device)
         
        # Retrieve the 'done' flags
        dones = torch.BoolTensor([self.experience['done'][i] for i in ids]).to(device)           

        # Compute Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = TargetNet.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute the loss between the predicted Q-values and the actual Q-values
        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad() # Reset gradients before performing a backward pass.
        loss.backward()            # backpropagation
        self.optimizer.step()      # update weights  networ

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # function to preprocess state before packaging
    def preprocess(self, state) :
        result = list(state['board'][:])
        result.append(state['mark'])
        return np.array(result, dtype=np.float32)

    # def preprocess(self, state):
    #   result = state['board'][:]
    #   result.append(state['mark'])
    #   return result

    # function for prediction
    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())

    def get_action(self, state, epsilon):
        # if we explore select a random available column
        if np.random.random() < epsilon:
            return int(np.random.choice([c for c in range(self.num_actions) if state['board'][c] == 0]))

        prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
        for i in range(self.num_actions):
            if state['board'][i] != 0 :
                prediction[i] = -1e7
        # return the column with largest predicted Q value
        return int(np.argmax(prediction))