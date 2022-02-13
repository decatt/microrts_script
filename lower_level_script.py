import numpy as np
import torch
from numpy.random import choice
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0:
        return 0
    return choice(range(len(logits)), p=logits / sum(logits))


def near_target(unit_pos_x: int, unit_pos_y: int, target_x: int, target_y: int) -> bool:
    if unit_pos_x + 1 == target_x and unit_pos_y == target_y:
        return True
    if unit_pos_x - 1 == target_x and unit_pos_y == target_y:
        return True
    if unit_pos_x == target_x and unit_pos_y + 1 == target_y:
        return True
    if unit_pos_x == target_x and unit_pos_y - 1 == target_y:
        return True
    return False


def get_default_action(unit, action_temp):
    action = [
        unit,
        sample(action_temp[0:6]),  # action type: NOOP, move, harvest, return, produce, attack
        sample(action_temp[6:10]),  # move parameter
        sample(action_temp[10:14]),  # harvest parameter
        sample(action_temp[14:18]),  # return parameter
        sample(action_temp[18:22]),  # produce_direction parameter
        sample(action_temp[22:29]),  # produce_unit_type parameter
        sample(action_temp[29:78]),  # attack_target parameter
    ]
    return action


def move_to_target(x, y, target, action, action_temp):
    move_parameter = action_temp[6:10]
    d = -1
    if action_temp[1] == 1:
        # action[1] = 1
        if target[0] > x and move_parameter[1] == 1:
            action[2] = 1
            d = 1
        if target[1] > y and move_parameter[2] == 1:
            action[2] = 2
            d = 2
        if target[0] < x and move_parameter[3] == 1:
            action[2] = 3
            d = 3
        if target[1] < y and move_parameter[0] == 1:
            action[2] = 0
            d = 0
    return action, d


def auto_attack(units, action_mask):
    actions = []
    for unit in units:
        action_temp = action_mask[unit]
        action = get_default_action(unit, action_temp)
        if action_temp[5] == 1:
            action[1] = 5
        actions.append(action)
    return actions


def auto_return(units, action_mask):
    actions = []
    for unit in units:
        action_temp = action_mask[unit]
        action = get_default_action(unit, action_temp)
        if action_temp[3] == 1:
            action[1] = 3
        actions.append(action)
    return actions


def harvest_resource(gs: torch.Tensor, worker: int, action_temp, target):
    workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17]
    barracks = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[16]
    resource = torch.ones((16, 16)).to(device) - gs.permute((2, 0, 1))[5]
    resource_workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17] * gs.permute((2, 0, 1))[6]
    worker_without_resource = workers - resource_workers
    # actions = []
    x = worker % 16
    y = worker // 16
    move_parameter = action_temp[6:10]
    action = get_default_action(worker, action_temp)
    if resource_workers[y][x] == 1:
        target = (2, 2)
    elif worker_without_resource[y][x] == 1 and x <= 2 and y <= 2:
        d1 = abs(x - 0) + abs(y - 0)
        d2 = abs(x - 0) + abs(y - 1)
        if d1 > d2 and resource[1][0] == 1:
            target = (0, 1)
        elif resource[0][0] == 1:
            target = (0, 0)
    if action_temp[5] == 1:  # attack
        action[1] = 5
    elif action_temp[3] == 1:  # return
        action[1] = 3
    elif action_temp[2] == 1 and x < 8 and y < 8:  # harvest
        action[1] = 2
    elif action_temp[4] == 1 and ((y, x) == (0, 2) or (y, x) == (0, 4) or (y, x) == (1, 3)) and barracks.sum() < 1:
        action[1] = 4
        if (y, x) == (0, 2):
            action[5] = 1
        if (y, x) == (0, 4):
            action[5] = 3
        if (y, x) == (1, 3):
            action[5] = 0
    elif action_temp[1] == 1 and workers[x][y] == 1:
        move_parameter = action_temp[6:10]
        action[1] = 1
        if target[0] > x and move_parameter[1] == 1:
            action[2] = 1
        if target[1] > y and move_parameter[2] == 1:
            action[2] = 2
        if target[0] < x and move_parameter[3] == 1:
            action[2] = 3
        if target[1] < y and move_parameter[0] == 1:
            action[2] = 0
    return action, target


def barracks_produce(action, unit_type):
    if action[1] == 4 and action[6] >= 4:
        action[6] = unit_type
        action[5] = 1
    return action


def move_to_region(region):
    pass


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HigherLevelNetwork(torch.nn.Module):
    def __init__(self):
        super(HigherLevelNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            # layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor = layer_init(nn.Linear(256, 8), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))


def force_harvest_resource(gs: torch.Tensor, worker: int, action_temp):
    force_action = True
    workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17]
    resource = torch.ones((16, 16)).to(device) - gs.permute((2, 0, 1))[5]
    resource_workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17] * gs.permute((2, 0, 1))[6]
    worker_without_resource = workers - resource_workers
    # actions = []
    x = worker % 16
    y = worker // 16
    if workers[y][x] == 1:
        action = get_default_action(worker, action_temp)
        target = (2, 2)
        if resource_workers[y][x] == 1:
            target = (2, 2)
        elif worker_without_resource[y][x] == 1 and x <= 2 and y <= 2:
            d1 = abs(x - 0) + abs(y - 0)
            d2 = abs(x - 0) + abs(y - 1)
            if d1 > d2 and resource[1][0] == 1:
                target = (0, 1)
            elif resource[0][0] == 1:
                target = (0, 0)
        if action_temp[5] == 1:  # attack
            action[1] = 5
        elif action_temp[3] == 1:  # return
            action[1] = 3
        elif action_temp[2] == 1 and x < 8 and y < 8:  # harvest
            action[1] = 2
        elif action_temp[1] == 1 and workers[x][y] == 1:
            move_parameter = action_temp[6:10]
            action[1] = 1
            if target[0] > x and move_parameter[1] == 1:
                action[2] = 1
            elif target[1] > y and move_parameter[2] == 1:
                action[2] = 2
            elif target[0] < x and move_parameter[3] == 1:
                action[2] = 3
            elif target[1] < y and move_parameter[0] == 1:
                action[2] = 0
        else:
            force_action = False
    else:
        force_action = False
        action = []
    return action, force_action


def move_to_target_and_act(gs: torch.Tensor, unit, target, action_temp, size):
    workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17]
    barracks = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[16]
    u_x = unit % size
    u_y = unit // size
    t_x = target % size
    t_y = target // size
    action = get_default_action(unit, action_temp)
    unit_por = gs[u_y][u_x]
    action, d = move_to_target(u_x, u_y, (t_x, t_y), action, action_temp)
    if not near_target(u_x, u_y, t_x, t_y):
        if unit_por[15] != 1 and unit_por[14] != 1:
            action[1] = 1
    if action_temp[5] == 1:  # attack
        action[1] = 5
    elif action_temp[3] == 1:  # return
        action[1] = 3
    elif action_temp[2] == 1 and u_x < 5 and u_y < 5:  # harvest
        action[1] = 2
    elif action_temp[4] == 1 and (
            (u_y, u_x) == (0, 2) or (u_y, u_x) == (0, 4) or (u_y, u_x) == (1, 3)) and barracks.sum() < 1 and unit_por[
        17] == 1:
        action[1] = 4
        if (u_y, u_x) == (0, 2):
            action[5] = 1
        if (u_y, u_x) == (0, 4):
            action[5] = 3
        if (u_y, u_x) == (1, 3):
            action[5] = 0
    elif action_temp[4] and unit_por[16] == 1:
        action[1] = 4
        action[6] = 6
    elif action_temp[4] and unit_por[15] == 1 and workers.sum() < 4:
        action[1] = 4
    return action


def simple_astar_path_finding(state, unit, target, size):
    next_pos = [unit + np.array([0, -1]), np.array(unit + [1, 0]), unit + np.array([0, 1]), unit + np.array([-1, 0])]
    next_pos_value = astar_get_next_pos_value(state, unit, target, size)
    for i in range(4):
        if next_pos_value[i] < 1000:
            nn_pos_value = astar_get_next_pos_value(state, next_pos[i], target, size)
            next_pos_value[i] = min(nn_pos_value)
    return np.argmin(next_pos_value)


def astar_get_next_pos_value(state, unit, target, size):
    next_pos = [unit + np.array([0, -1]), np.array(unit + [1, 0]), unit + np.array([0, 1]), unit + np.array([-1, 0])]
    next_pos_value = np.array([1000, 1000, 1000, 1000])
    for i in range(4):
        if 0 <= next_pos[i][0] < size and 0 <= next_pos[i][1] < size:
            if state[next_pos[i][1]][next_pos[i][0]] == 0:
                next_pos_value[i] = abs(next_pos[i][0] - target[0]) + abs(next_pos[i][1] - target[1])
    return next_pos_value


def astar_script(gs, unit, target, size, action_temp):
    if unit == target:
        return np.array([unit, 0, 0, 0, 0, 0, 0, 0])
    produce_type = 0
    attack_pos = get_attack_pos(unit, target, size)

    if gs[unit // size][unit % size][15] == 1:
        action_type = 4
        produce_type = 3
    elif gs[unit // size][unit % size][16] == 1:
        action_type = 4
        produce_type = 4
    elif gs[target // size][target % size][14] == 1:
        action_type = 2
    elif gs[target // size][target % size][15] == 1:
        action_type = 3
    else:
        action_type = 1

    if action_temp[5] == 1 and gs[target // size][target % size][12] == 1 and (abs(unit//size - target//size)+abs(unit%size - target%size))<4:
        action_type = 5

    if gs[unit // size][unit % size][20] == 0 or action_type != 5:
        if unit - target == size:
            d = 0
        elif unit - target == -1:
            d = 1
        elif unit - target == -size:
            d = 2
        elif unit - target == 1:
            d = 3
        else:
            ast_state = gs[:, :, 1] + gs[:, :, 11] + gs[:, :, 12]
            ast_state[target // size][target % size] = 0
            d = simple_astar_path_finding(ast_state, np.array([unit % size, unit // size]), np.array([target % size, target // size]), size)
            if gs[unit // size][unit % size][15] == 0 and gs[unit // size][unit % size][16] == 0:
                action_type = 1
    else:
        d = 0

    action = np.array([unit, action_type, d, d, d, d, produce_type, attack_pos])
    return action


def get_attack_pos(unit, target, size):
    p = np.array([3, 3]) + np.array([target // size - unit // size, target % size - unit % size])
    pos = 7 * p[0] + p[1]
    if pos >= 49:
        return 0
    else:
        return pos


def area_type_script(area_type, state, pos):
    target = pos
    pos_state = state[pos]
    if area_type == 0:
        if pos_state[17] == 1:
            if pos_state[19] == 1:
                target = 34
            else:
                target = 16
        else:
            target = 256
    if area_type == 1:
        target = 238

    return target


class State:
    def __init__(self, d, grid_state):
        self.grid_state = grid_state
        self.d = d

    def __lt__(self, other):
        if self.d < other.d:
            return True
