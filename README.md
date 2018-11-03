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
### Jupyter notebook
Udacity provide me with a Jupyter notebook with the information about creating and setting up the environment.
### DQN implementation
I implemented few flavors of deep Q network and One can choose the type of network through parameters while constructing the agent class. Following types of DQN are implemented
- Original Google Deepmind DQN (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- Double DQN (https://arxiv.org/abs/1509.06461)
- Prioritized replay buffer (https://arxiv.org/abs/1511.05952)
- Dueling DQN (https://arxiv.org/abs/1511.06581)
- Recurrent DQN
Pytorch is used as the machine learning library.






