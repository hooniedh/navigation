from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import *
from collections import deque
import matplotlib.pyplot as plt
import sys


def Train(env, agent, num_episodes=2000, eps_start=1.0, eps_end=0.01,
          eps_decay=0.995, beta_start=0.4, episode_for_beta_one=1500):
    """Summary

    Args:
        env (Unity ML environment): environment
        agent (DQN agent): the learning agent
        num_episodes (int, optional): the number of episodes to run
        eps_start (float, optional): the starting epsilon value for epsilon greedy algorithm
        eps_end (float, optional): minimum epsilon value
        eps_decay (float, optional): he decay rate for epsilon value per episode
        beta_start (float, optional): the starting beta value for important sampling weight described in the paper (https://arxiv.org/abs/1511.05952)
        episode_for_beta_one (int, optional): the episode at which the beta value is set to 1.0
    """
    score_report_every = 100
    scores = []
    scores_window = deque(maxlen=score_report_every)
    eps = eps_start
    beta = beta_start
    beta_inc = (1 - beta_start) / episode_for_beta_one
    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        state = env_info.vector_observations[0]                # get the current state
        score = 0                                              # initialize the scre
        agent.startEpisode()
        while True:
            action = agent.act(state, eps)                     # choose the action using the current policy
            env_info = env.step(int(action))[brain_name]       # get the transition from the environment with the chosen action
            next_state = env_info.vector_observations[0]       # get the next state
            reward = env_info.rewards[0]                       # get the reward
            score += reward
            done = env_info.local_done[0]                      # see if episode has finished
            agent.step(state, action, reward, next_state, done, beta)
            state = next_state
            if done:                                       # exit loop if episode finished
                break
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        beta = min(beta + beta_inc, 1.0)
        scores_window.append(score)
        scores.append(score)
        if episode % score_report_every == 0:
            print("Score: {} at episode {}".format(np.mean(scores_window), episode))

    env.close()
    plot(scores, "rewards.png")
    torch.save(agent.qnetwork_local.state_dict(), 'model.pth')


def plot(scores, file_name):
    xlable = np.arange(len(scores), dtype=int)
    plt.plot(xlable, scores)
    plt.ylabel('total rewards')
    plt.xlabel('episode')
    plt.savefig(file_name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        env = UnityEnvironment(file_name=sys.argv[1])
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)
        agent = createAgent(state_size, action_size,
                            0,                                  # random seed
                            sequence_length=10,                 # sequence length for the recurrent DQN
                            use_double_DQN=True,                # Double DQN
                            use_dueling_network=True            # Dueling network
                            )
        print("start training....")
        Train(env, agent)
    else:
        print("usage - python training.py path to the agent executable")
