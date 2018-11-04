import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
from model import DuelQNetwork
from model import DuelQGRUNetwork
import torch
import torch.optim as optim


class BaseAgent:
    def __init__(self, state_size, action_size, seed,
                 buffer_size, batch_size, gamma, tau,
                 lr, update_every, update_target_network_every,
                 alpha, sequence_length,
                 use_dueling_network):
        """Base class for agents

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
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_dueling_network is True:
            if sequence_length > 1:
                self.qnetwork_local = DuelQGRUNetwork(state_size, action_size, seed).to(self.device)
                self.qnetwork_target = DuelQGRUNetwork(state_size, action_size, seed).to(self.device)
            else:
                self.qnetwork_local = DuelQNetwork(state_size, action_size, seed).to(self.device)
                self.qnetwork_target = DuelQNetwork(state_size, action_size, seed).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.batch_size = batch_size
        self.update_every = update_every
        self.update_target_network_every = update_target_network_every
        self.gamma = gamma
        self.tau = tau
        self.sequence_length = sequence_length

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, self.device, alpha, sequence_length)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.u_step = 0  # used to control how often we copy local parameters to target parameters

        self.resetSequences()

    def resetSequences(self):
        """reset the state sequences (used for the recurrent DQN)
        """
        if self.sequence_length > 1:
            self.sequence = deque([np.zeros(self.state_size) for i in range(self.sequence_length)], maxlen=self.sequence_length)
            self.state_sequence = deque([np.zeros(self.state_size) for i in range(self.sequence_length)], maxlen=self.sequence_length)
            self.next_state_sequence = deque([np.zeros(self.state_size) for i in range(self.sequence_length)], maxlen=self.sequence_length)

    def startEpisode(self):
        """Perform tasks for starting each episode
        """
        self.resetSequences()

    def step(self, state, action, reward, next_state, done, beta):
        """Add an episode to the memory and run a learning step after sampling from the replay buffer

        Args:
            state (list): current state
            action (int): the action taken according to the policy given the state
            reward (float): the reward from the environment
            next_state (list): the next state returned from the environment given the action
            done (float): flag returned from the environment to indicate if the current episode is done
            beta (float): beta value for important sampling weight
        """

        # save the sample to the memory
        if self.sequence_length > 1:
            self.state_sequence.popleft()
            self.state_sequence.append(state)
            self.next_state_sequence.popleft()
            self.next_state_sequence.append(next_state)
            self.memory.add(list(self.state_sequence), action, reward, list(self.next_state_sequence), done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, beta)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        if self.sequence_length == 1:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            self.sequence.popleft()
            self.sequence.append(state)
            state = torch.from_numpy(np.array(list(self.sequence))).float().to(self.device)
            state = state.view(1, self.sequence_length, -1)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def getTargetValue(self, next_states, rewards, dones):
        """Get target Q value given the next state
        from the target network

        Args:
            next_states (tensor array): the batch of next states returned from the environment
            rewards (tensor array): the batch of rewards returned from the environment
            dones (tensor array): the batch of done flags returned from the environment

        Returns:
            TYPE: Description
        """
        target_net_action_values = self.qnetwork_target(next_states)
        target_max = torch.max(target_net_action_values.detach(), 1)[0].unsqueeze(1)
        target = rewards + (self.gamma * target_max * (1 - dones))

        return target

    def getEstimatedValue(self, states, actions):
        """get the batch of estimated Q values from the local network
        given states and actions batch

        Args:
            states (tensor array): batch of states
            actions (tensor array): batch of actions
        """
        current_q = self.qnetwork_local(states)
        current = current_q.gather(1, actions)

        return current

    def learn(self, experiences, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, probs = experiences

        target = self.getTargetValue(next_states, rewards, dones)
        current = self.getEstimatedValue(states, actions)

        delta = torch.abs(current - target)

        # calculating important sampling weight for prioritized sampling
        num_samples_in_memory = len(self.memory.memory)
        min_prob = np.min(self.memory.probs[:num_samples_in_memory])
        weight = (1 / (probs * num_samples_in_memory)) ** beta

        # normalize with the maximum weight. the min prob sample generates the maximum weight
        normalized_weight = weight / ((1 / (min_prob * num_samples_in_memory)) ** beta)

        # Calculating Huber loss
        huber_threshold = 1.
        is_small_delta = (delta < huber_threshold).to(torch.float)
        loss = 0.5 * (delta ** 2) * is_small_delta + huber_threshold * (delta - 0.5 * huber_threshold) * (1 - is_small_delta)
        loss *= normalized_weight

        # Performing back propagation and update the target network
        self.backProp(loss.mean())

        # Update the sampling probability (prioritized replay algorithm)
        new_probs = (delta + 1e-5).detach().cpu().numpy()
        self.memory.updateProbs(indices, new_probs)

    def backProp(self, loss):
        """performs back propagation

        Args:
            loss (pytorch loss function): Description
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.u_step = (self.u_step + 1) % self.update_target_network_every
        if self.u_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha, sequence_length):
        """Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (int): Pytorch device (CPU or CUDA)
            alpha (float): the weight to control the importance of the priority to calculate sampling probability (0 means random sampling)
            sequence_length (int): if this is greater than 1, the recurrent dueling DQN is used. Use 1 for non-recurrent networks.
        """
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.buffer_size = buffer_size
        self.probs = np.zeros(buffer_size)
        self.index = 0
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.alpha = alpha
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_prob = self.probs.max() if len(self.memory) > 0 else 1
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.index] = e
        self.probs[self.index] = max_prob           # set the maximum probability when the sample is added
        self.index = (self.index + 1) % self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        num_samples = len(self.memory)
        total_prob = sum(self.probs)
        sample_indices = np.random.choice(np.arange(num_samples), self.batch_size, p=self.probs[:num_samples] / total_prob)

        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in sample_indices])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in sample_indices])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in sample_indices]).astype(np.uint8)).float().to(self.device)
        indices = np.vstack([idx for idx in sample_indices])
        probs = torch.from_numpy(np.vstack([self.probs[idx] / total_prob for idx in sample_indices])).float().to(self.device)
        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in sample_indices])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in sample_indices])).float().to(self.device)

        if self.sequence_length > 1:
            states = states.view(self.batch_size, self.sequence_length, -1)
            next_states = next_states.view(self.batch_size, self.sequence_length, -1)

        return (states, actions, rewards, next_states, dones, indices, probs)

    def updateProbs(self, indices, new_probs):
        """Update the sample probability according to the prioritized replay algorithm

        Args:
            indices (array): indices of the samples to update the sample probability
            new_probs (array): sample probability for the samples
        """
        self.probs[indices] = new_probs ** self.alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
