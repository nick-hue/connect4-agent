import numpy as np                
import gym
from kaggle_environments import evaluate, make
import torch
import torch.nn as nn
import torch.optim as optim

class DeepModel(nn.Module) :

    # constructor method (inisialisai)
    def __init__(self, num_states, hidden_units, num_actions) :

        super(DeepModel, self).__init__()

        # mengkonstruksi hidden layer (perhatikan ilustrasi struktur neural network pada gambar di atas)
        # akan dicoba activation function berupa ReLU
        self.hidden_layers = []
        for i in range(len(hidden_units)) :
            # untuk hidden layer pertama
            if i == 0 :
                self.hidden_layers.append(nn.Linear(num_states, hidden_units[i]))
            # untuk hidden layer berikutnya
            else :
                self.hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
        self.output_layer = nn.Linear(hidden_units[-1], num_actions)

    def forward(self, x) :
        for layer in self.hidden_layers :
            x = layer(x).clamp(min=0)
        x = self.output_layer(x)
        return x

# Mendefinisikan kelas untuk agent
# Secara umum, agent harus punya 3 kemampuan : bermain beberapa permainan, mengingatnya, dan memperkirakan reward dari tiap state
# yang muncul pada permainan

class DQN :

    # constructor method (inisialisasi)
    def __init__(self, name, turn, player_number, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr) :
        self.name = name
        self.turn = turn
        self.player_number = player_number

        self.num_actions = num_actions   # banyaknya aksi
        self.gamma = gamma               # porsi future reward terhadap immadiate reward

        # inisialisasi atribut untuk mekanisme mengingat permainan
        self.batch_size = batch_size
        self.experience = {'s' : [], 'a' : [], 'r' : [], 's2' : [], 'done' : []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        # inisialisasi atribut untuk perkiraan reward dengan neural network di setiap state
        self.model = DeepModel(num_states, hidden_units, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()

    # fungsi untuk mengatur permainan yang diingat lalu melakukan training neural network dari sana

    # membuang ingatan state paling awal bila sudah melebihi batas memori
    def add_experience(self, exp) :
        if len(self.experience['s']) >= self.max_experiences :
            for key in self.experience.keys() :
                self.experience[key].pop(0)
        for key, value in exp.items() :
            self.experience[key].append(value)

    # melakukan training dari neural network
    # panduannya bisa dilihat di https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def train(self, TargetNet) :

        # hanya melakukan training bila state yang diingat telah melebihi batas minimal
        if len(self.experience['s']) < self.min_experiences :
            return 0

        # mengambil ingatan permainan secara random sesuai ukuran batch
        ids = np.random.randint(low = 0, high = len(self.experience['s']), size = self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # mengambil label
        next_states = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        # untuk semua ingatan yang diambil, kita harus memprediksi maksimal Q-value dari state setelahnya
        value_next = np.max(TargetNet.predict(next_states).detach().numpy(), axis = 1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        actions = np.expand_dims(actions, axis = 1)
        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)
        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim = 1)
        actual_values = torch.FloatTensor(actual_values)

        self.optimizer.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()

    def copy_weights(self, TrainNet) :
        self.model.load_state_dict(TrainNet.state_dict())

    def save_weights(self, path) :
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path) :
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def preprocess(self, state) :
        result = list(state['board'][:])
        result.append(state['mark'])
        return np.array(result, dtype=np.float32)

    def predict(self, inputs) :
        return self.model(torch.from_numpy(inputs).float())

    def get_action(self, state, epsilon) :
        if np.random.random() < epsilon :
            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))
        else :
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
            for i in range(self.num_actions) :
                if state.board[i] != 0 :
                    prediction[i] = -1e7
            return int(np.argmax(prediction))