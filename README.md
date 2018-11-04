# navigation
This is my solution to Udicity reinfocement learning nanodegree **Navigation** project. Deep Q networks are used.
### environment
The environment is provided by Udacity and it is made from the Unity ML agent (https://github.com/Unity-Technologies/ml-agents).
The agent navigates the environmnet while eating yellow and blue bananas.
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
  
### getting started
This project depends on Pytorch and Unity ML agents.
- Pytorch: https://pytorch.org/
- Unity ML agents: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
- Udacity navigation environment: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

**It is important to note that the agent provided by Udacity is only compatible with Unity ML agent version 0.4.0. It can be downloaded from https://github.com/Unity-Technologies/ml-agents/releases.**

### instructions
A train can be started by running training.py. The path and the executable of the environment should be passed in as the first argument.

'''
python training.py Banana_Windows_x86_64/Banana.exe
'''







