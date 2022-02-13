import torch.nn as nn
import torch
import numpy as np
from tools.mask import CategoricalMasked
from tools.dimension_reduction import get_nodes_vectors

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GNNNet(nn.Module):
    def __init__(self):
        super(GNNNet, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(53, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, 78), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        obs = self.network(x)
        return self.actor(obs), self.critic(obs)


class GNNAgent:
    def __init__(self, net: GNNNet, size):
        self.net = net
        self.size = size
        self.map_size = size*size

    def get_action(self, obs=None, graphs=None, invalid_action_masks=None, action_plane_space=None):
        if graphs is None:
            graphs = get_nodes_vectors(obs, self.size)
        logits, value = self.net(graphs)
        logits = logits.cpu()
        value = value.cpu()
        grid_logits = logits.reshape(-1, action_plane_space.sum())
        split_logits = torch.split(grid_logits, action_plane_space.tolist(), dim=1)
        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(invalid_action_masks, action_plane_space.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam)
            for (logits, iam) in zip(split_logits, split_invalid_action_masks)
        ]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        num_predicted_parameters = len(action_plane_space)
        log_prob = log_prob.T.view(-1, self.map_size, num_predicted_parameters)
        action = action.T.view(-1, self.map_size, num_predicted_parameters)
        return action, value, log_prob.sum(1).sum(1)

    def get_entropy(self, x, graphs=None, action=None, invalid_action_masks=None, envs=None):
        if graphs is None:
            graphs = x
        logits, value = self.net(graphs)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)

        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        action = action.view(-1, action.shape[-1]).T
        split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam)
            for (logits, iam) in zip(split_logits, split_invalid_action_masks)
        ]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)
        log_prob = log_prob.T.view(-1, self.map_size, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.map_size, num_predicted_parameters)
        return value, log_prob.sum(1).sum(1), entropy.sum(1).sum(1)

    def get_value(self, graphs):
        return self.net.critic(self.net.network(graphs))