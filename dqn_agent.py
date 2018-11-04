import torch
from dqn_agent_base import BaseAgent


def createAgent(state_size, action_size, seed,
                buffer_size=int(3e5), batch_size=64, gamma=0.99, tau=1,
                lr=5e-4, update_every=4, update_target_network_every=2500,
                alpha=0.6, sequence_length=1,
                use_double_DQN=True, use_dueling_network=True):
    """Create a learning agent

    Args:
        state_size (flat): the number of states
        action_size (float): the number of actions
        seed (int): random seed
        buffer_size (int, optional): the maximum number of elements in the replay buffer
        batch_size (int, optional): mini batch size
        gamma (float, optional): reward discount rate
        tau (int, optional): target network update rate. Use 1 if the target network is updated entirely from the local network at once
        lr (float, optional): network learning rate on the Adam optimizer
        update_every (int, optional): how often we learn the learning step (i.e. 4 means the learning step is executed ever 4 action taken)
        update_target_network_every (int, optional): how often the target network is updated from the local network
        alpha (float, optional): the weight to control the importance of the priority to calculate sampling probability (0 means random sampling)
        sequence_length (int, optional): if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
        use_double_DQN (bool, optional): true to use the double DQN
        use_dueling_network (bool, optional): true to use the dueling DQN
    """

    if use_double_DQN is True:
        return DoubleDqnAgent(state_size, action_size, seed,
                              buffer_size, batch_size, gamma, tau,
                              lr, update_every, update_target_network_every,
                              alpha, sequence_length,
                              use_dueling_network)
    else:
        return Agent(state_size, action_size, seed,
                     buffer_size, batch_size, gamma, tau,
                     lr, update_every, update_target_network_every,
                     alpha, sequence_length,
                     use_dueling_network)


class Agent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                 buffer_size, batch_size, gamma, tau,
                 lr, update_every, update_target_network_every,
                 alpha, sequence_length,
                 use_dueling_network):
        """Initialize an Agent object.

        Args:
        state_size (flat): the number of states
        action_size (float): the number of actions
        seed (int): random seed
        buffer_size (int, optional): the maximum number of elements in the replay buffer
        batch_size (int, optional): mini batch size
        gamma (float, optional): reward discount rate
        tau (int, optional): target network update rate. Use 1 if the target network is updated entirely from the local network at once
        lr (float, optional): network learning rate on the Adam optimizer
        update_every (int, optional): how often we learn the learning step (i.e. 4 means the learning step is executed ever 4 action taken)
        update_target_network_every (int, optional): how often the target network is updated from the local network
        alpha (float, optional): the weight to control the importance of the priority to calculate sampling probability (0 means random sampling)
        sequence_length (int, optional): if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
        use_dueling_network (bool, optional): true to use the dueling DQN
        """
        super().__init__(state_size, action_size, seed,
                         buffer_size, batch_size, gamma, tau,
                         lr, update_every, update_target_network_every,
                         alpha, sequence_length,
                         use_dueling_network)


class DoubleDqnAgent(BaseAgent):
    def __init__(self, state_size, action_size, seed,
                 buffer_size, batch_size, gamma, tau,
                 lr, update_every, update_target_network_every,
                 alpha, sequence_length,
                 use_dueling_network):
        """Initialize an Double DQN Agent object.

        Args:
        state_size (flat): the number of states
        action_size (float): the number of actions
        seed (int): random seed
        buffer_size (int, optional): the maximum number of elements in the replay buffer
        batch_size (int, optional): mini batch size
        gamma (float, optional): reward discount rate
        tau (int, optional): target network update rate. Use 1 if the target network is updated entirely from the local network at once
        lr (float, optional): network learning rate on the Adam optimizer
        update_every (int, optional): how often we learn the learning step (i.e. 4 means the learning step is executed ever 4 action taken)
        update_target_network_every (int, optional): how often the target network is updated from the local network
        alpha (float, optional): the weight to control the importance of the priority to calculate sampling probability (0 means random sampling)
        sequence_length (int, optional): if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
        use_double_DQN (bool, optional): true to use the double DQN
        use_dueling_network (bool, optional): true to use the dueling DQN
        """
        super().__init__(state_size, action_size, seed,
                         buffer_size, batch_size, gamma, tau,
                         lr, update_every, update_target_network_every,
                         alpha, sequence_length,
                         use_dueling_network)

    def getTargetValue(self, next_states, rewards, dones):
        """Get target Q value given the next state
        from the target network

        Args:
            next_states (tensor array): the batch of next states returned from the environment
            rewards (tensor array): the batch of rewards returned from the environment
            dones (tensor array): the batch of done flags returned from the environment

        Returns:
            TYPE: the target value (tensor array)
        """
        actions_from_local = torch.argmax(self.qnetwork_local(next_states).detach(), 1).unsqueeze(1)
        target_max = self.qnetwork_target(next_states).detach().gather(1, actions_from_local)
        target = rewards + (self.gamma * target_max * (1 - dones))

        return target
