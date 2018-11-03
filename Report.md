### Learning Algorithm
I implemented few flavors of deep Q network and One can choose the type of network through parameters while constructing the agent class. Following types of DQN are implemented
- Original Google Deepmind DQN (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- Double DQN (https://arxiv.org/abs/1509.06461)
- Prioritized replay buffer (https://arxiv.org/abs/1511.05952)
- Dueling DQN (https://arxiv.org/abs/1511.06581)
- Recurrent DQN (using GRU cell)

The learning agent is created by calling createAgent function in dqn_agent.py with these parameters:
- state_size: the number of states
- action_size: the number of actions
- seed: random seed
- buffer_size: maximum number of elements in the replay buffer
- batch_size: mini batch size
- gamma: reward discount rate
- tau: target network update rate. Use 1 if the target network is updated entirely from the local network at once
- lr: network learning rate on the Adam optimizer
- update_every: how often we learn the learning step (i.e. 4 means the learning step is executed ever 4 action taken)
- update_target_network_every: how often the target network is updated from the local network 
- alpha: the weight to control the importance of the priority to calculate sampling probability (0 means random sampling)
- sequence_length: if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
- use_double_DQN: true to use the double DQN
- use_dueling_network: true to use the dueling DQN

The entry point of the training is train() function in training.py. The train function takes following parameters:
- num_episodes: the number of episodes to run
- eps_start: the starting epsilon value for epsilon greedy algorithm
- eps_end: minimum epsilon value
- eps_decay: the decay rate for epsilon value per episode
- beta_start: the starting beta value for important sampling weight described in the paper (https://arxiv.org/abs/1511.05952)
- episode_for_beta_one: the episode at which the beta value is set to 1.0

### network architecture
Following network parameters are used (Recurrent Dueling Double DQN with a prioritized replay buffer):
- 1 layer of GRU layer with 256 hidden states and tanh activation function
- value network with 128 hidden states and a Relu activation function
- advantage network with 128 hidden states and a Relu activation function

### hyperparameters


