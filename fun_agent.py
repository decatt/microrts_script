import torch.nn as nn
import torch
import numpy as np
from mask import CategoricalMasked

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class WorkerNet(nn.Module):
    def __init__(self, action_space=None, goal_size=12):
        super(WorkerNet, self).__init__()
        if action_space is None:
            action_space = [100, 6, 4, 4, 4, 4, 7, 49]
        self.version = 0

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(32 * 6 * 6, 256)),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(256+goal_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, sum(action_space)), std=0.01))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(256+goal_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1))

    def forward(self, x, g):
        x = x.permute((0, 3, 1, 2))
        obs = torch.cat((self.network(x), g), dim=1)
        return self.actor(obs), self.critic(obs)


class WorkerAgent:
    def __init__(self, net, action_space: list):
        self.net = net
        self.action_space = action_space

    def get_log_prob_entropy_value(self, x, g, action, masks):
        """
        :param x:
        :param g:
        :param action:
        :param masks:
        :return :log_prob entropy value
        """
        logits, value = self.net(x, g)
        split_logits = torch.split(logits, self.action_space, dim=1)
        split_masks = torch.split(masks, self.action_space, dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam, use_gpu=True) for (logits, iam) in zip(split_logits, split_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action.T, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return log_prob, entropy, value

    def get_value(self, x, g):
        return self.net(x, g)[1]

    def get_action(self, x, g, unit_masks=None, action_masks=None):
        """
        :param x:
        :param g:
        :param unit_masks:
        :param action_masks:
        :return : action, log_probs, mask, value
        """
        logits, value = self.net(x, g)
        logits = logits.cpu()
        split_logits = torch.split(logits, self.action_space, dim=1)
        # unit_masks = torch.Tensor(unit_masks)
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=unit_masks)]
        action_components = [multi_categoricals[0].sample()]
        a_masks = torch.zeros((len(action_components[0]), 78))
        for i in range(len(action_components[0])):
            a_masks[i] = action_masks[i][action_components[0][i]]
        split_suam = torch.split(a_masks, self.action_space[1:], dim=1)
        multi_categoricals = multi_categoricals + [
            CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
        ]
        masks = torch.cat((unit_masks, a_masks), 1)
        action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
        action = torch.stack(action_components)
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        return action.T.cpu().numpy(), log_prob.cpu().sum(0).reshape(-1), masks.cpu(), value.cpu().reshape(-1)


class ManagerNet(nn.Module):
    def __init__(self, goal_length=12):
        super(ManagerNet, self).__init__()
        self.version = 0

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(32 * 6 * 6, 256)),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor_m = nn.Sequential(
            layer_init(nn.Linear(256, goal_length), std=0.01),
            nn.Sigmoid())
        # self.actor_s = layer_init(nn.Linear(256, goal_length), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        obs = self.network(x)
        return self.actor_m(obs), self.critic(obs)


class ManagerAgent:
    def __init__(self, net, num_env):
        self.net = net
        self.num_env = num_env

    def get_goal(self, x, std=0.01):
        """

        :param x:
        :param std:
        :return: goals, value, log_probs
        """
        m, value = self.net(x)
        m = 10*m.reshape((-1, 12))
        normal_distributions = torch.distributions.Normal(m, std)
        a = normal_distributions.sample()
        goals = a.reshape(self.num_env, 12)
        log_probs = normal_distributions.log_prob(a).reshape(self.num_env, 12)
        return goals.cpu(), value.reshape(-1).cpu(), log_probs.sum(1).cpu()

    def get_entropy(self, x, a, std=0.01):
        m, value = self.net(x)
        m = m.reshape((-1, 12))
        normal_distributions = torch.distributions.Normal(m, std)
        log_probs = normal_distributions.log_prob(a).reshape(-1, 12)
        entropy = normal_distributions.entropy().reshape(-1, 12)
        return value, log_probs.sum(1), entropy.sum(1)

    def get_value(self, x):
        return self.net(x)[1]
