# navigation
This is my solution to Udicity reinfocement learning nanodegree **Navigation** project. Deep Q networks are used.
### environment
The environment is provided by Udacity and it is made from the Unity ML agent (https://github.com/Unity-Technologies/ml-agents).
The agnet navigates the environmnet while eating yellow and blue bananas.
- The environment returns 37 floats as the states in each step.
- There are four actions available to the agent
  - 0: walk forward
  - 1: walk backward
  - 2: turn left
  - 3: turn right
- Rewards
  - 1: eating a yellow banana
  - -1: eating a blue banana
  - 0: otherwise
 
 The environment is considered as solved if the agent receives an average reward over 100 episodes of at least +13.
  
### Getting started
This project depends on Pytorch and Unity ML agents.
- Pytorch: https://pytorch.org/
- Unity ML agents: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

**It is important to note that the agent provided by Udacity is only compatible with Unity ML agent version 0.4.0. It can be downloaded from https://github.com/Unity-Technologies/ml-agents/releases.**

### DQN implementation
I implemented few flavors of deep Q network and One can choose the type of network through parameters while constructing the agent class. Following types of DQN are implemented
- Original Google Deepmind DQN (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- Double DQN (https://arxiv.org/abs/1509.06461)
- Prioritized replay buffer (https://arxiv.org/abs/1511.05952)
- Dueling DQN (https://arxiv.org/abs/1511.06581)
- Recurrent DQN

Pytorch is used as the machine learning library.

The learning angen is created by calling createAgent function in dqn_agent.py with these parameters:
- state_size: the numbser of states
- action_size: the number of actions
- seed: randome seed
- buffer_size: the maximumn number of elements in the replay buffer
- batch_size: the mini batch size
- gamma: reward discount rate
- tau: target network update rate. Use 1 if the target network is udpated entirely from the local network at once
- lr: network learning rate on the adam optimizer
- update_every: how often we learn the learning step (i.e. 4 means the learning step is excuted ever 4 action taken)
- update_target_network_every: how often the target newtork is updated from the local network 
- alpha: the weight to control the importance of the priority to calcuate sampling probability (0 means random sampling)
- sequence_length: if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
- use_double_DQN: ture to use the double DQN
- use_dueling_network: true to use the dueling DQN






