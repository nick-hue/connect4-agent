import numpy as np                
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
    def __init__(self, name: str, turn: int, player_number:int, num_states=43, num_actions=7, hidden_units=[100, 200, 200, 100], gamma=0.99, max_experiences=17500, min_experiences=100, batch_size=32, lr=0.001):
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
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])

        actions = np.asarray([self.experience['a'][i] for i in ids])  # Retrieve the actions
        rewards = np.asarray([self.experience['r'][i] for i in ids])  # Retrieve the rewards

        # Get the next states from the samples of the experience bufferand preprocess them.
        next_states = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids]) # Retrieve the 'done' flags

        # Calculate the maximum Q-value for the next states using the target network.
        value_next = np.max(TargetNet.predict(next_states).detach().numpy(), axis = 1)
        # Compute the actual Q-values using the Bellman equation.
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        # Make the actions table into tensor input like
        actions = np.expand_dims(actions, axis = 1)
        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)

        # Calculate the Q-values for the actions taken using the current Q-network.
        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim = 1)
        actual_values = torch.FloatTensor(actual_values)

        # Reset gradients before performing a backward pass.
        self.optimizer.zero_grad()
        # Compute the loss between the predicted Q-values and the actual Q-values
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()           # backpropagation
        self.optimizer.step()     # update weights  networ

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def save_weights(self, path):
        torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict()}, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        print("Loaded")
        print(checkpoint['optimizer'])

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))
    # function to preprocess state before packaging
    def preprocess(self, state) :
        result = list(state['board'][:])
        result.append(state['mark'])
        return np.array(result, dtype=np.float32)

    # function for prediction
    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())

    # function to get the next action of our model 
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