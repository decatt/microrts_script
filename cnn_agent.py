import torch.nn as nn
import torch
import numpy as np
from mask import CategoricalMasked

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RTSNet(nn.Module):
    def __init__(self, action_space=None):
        super(RTSNet, self).__init__()
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

        self.actor = layer_init(nn.Linear(256, sum(action_space)), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        obs = self.network(x)
        return self.actor(obs), self.critic(obs)


class RTSAgent:
    def __init__(self, net: RTSNet, action_space: list):
        self.net = net
        self.action_space = action_space

    def get_log_prob_entropy_value(self, x, action, masks):
        logits, value = self.net(x)
        split_logits = torch.split(logits, self.action_space, dim=1)
        split_masks = torch.split(masks, self.action_space, dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam, use_gpu=True) for (logits, iam) in zip(split_logits, split_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action.T, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return log_prob.sum(0), entropy.sum(0), value

    def get_value(self, x):
        return self.net(x)[1]

    def get_action(self, x, unit_masks=None, action_masks=None):
        """

        :param x:
        :param unit_masks:
        :param action_masks:
        :return: action log_prob mask value
        """
        logits, value = self.net(x)
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

