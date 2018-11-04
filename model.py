import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model.
    Basic Q Network

    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = [80, 64]

        self.seq = nn.Sequential(
            nn.Linear(state_size, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.seq(state)


class DuelQNetwork(nn.Module):
    """Actor (Policy) Model.
    Dueling network structure
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        front_layers = [256, 128]
        value_size = 64
        advantage_size = 64

        self.front_seq = nn.Sequential(
            nn.Linear(state_size, front_layers[0]),
            nn.ReLU(),
            nn.Linear(front_layers[0], front_layers[1]),
            nn.ReLU())

        # value network
        self.value_seq = nn.Sequential(nn.Linear(front_layers[1], value_size),
                                       nn.ReLU(),
                                       nn.Linear(value_size, 1))

        # advantage network
        self.advantage_seq = nn.Sequential(nn.Linear(front_layers[1], advantage_size),
                                           nn.ReLU(),
                                           nn.Linear(advantage_size, action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        front_output = self.front_seq(state)
        value = self.value_seq(front_output)
        advantages = self.advantage_seq(front_output)
        avg_advantages = advantages.mean(1).unsqueeze(1)

        q = value + (advantages - avg_advantages)

        return q


class DuelQGRUNetwork(nn.Module):
    """Actor (Policy) Model.
    Recurrent Dueling network structure
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQGRUNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        gru_hidden_size = 256
        value_size = 128
        advantage_size = 128

        # GRU cell
        self.gru = nn.GRU(state_size, gru_hidden_size, 1, batch_first=True)

        # value network
        self.value_seq = nn.Sequential(nn.Linear(gru_hidden_size, value_size),
                                       nn.ReLU(),
                                       nn.Linear(value_size, 1))

        # advantage network
        self.advantage_seq = nn.Sequential(nn.Linear(gru_hidden_size, advantage_size),
                                           nn.ReLU(),
                                           nn.Linear(advantage_size, action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        gru_output, hidden_output = self.gru(state)
        sequence_length = gru_output.size(1)
        gru_output_last = gru_output[:, sequence_length - 1]  # Get the output for the last sequence from the GRU cell

        value = self.value_seq(gru_output_last)
        advantages = self.advantage_seq(gru_output_last)
        avg_advantages = advantages.mean(1).unsqueeze(1)

        q = value + (advantages - avg_advantages)

        return q
