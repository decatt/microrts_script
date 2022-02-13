import numpy as np
from tools.lower_level_script import astar_script, simple_astar_path_finding


class ScriptAgent:
    def __init__(self, size, base, barracks, op_base, resource):
        self.size = size
        self.base = base
        self.op_base = op_base
        self.barracks = barracks
        self.op_barracks = -1
        self.resource = resource
        self.action_size = (1, size * size, 7)
        self.nearest_op_unit = -1
        self.op_building_unit = -1
        self.rally_point_worker = -1
        self.rally_point_soldier = -1
        self.max_worker = 5
        self.num_harvest_worker = 1
        self.num_battle_worker = 1
        self.barracks_unit = 6  # light 4 heavy 5 range 6

    def get_action(self, states, masks):
        actions = np.zeros(self.action_size)
        for i in range(len(states)):
            state = states[i]
            units = masks[i].sum(1)
            self.check(state)
            self.set_rally_point(state)
            for pos in range(len(units)):
                target = pos
                if units[pos] > 0:
                    nu = self.get_nearest_unit(pos, state)
                    pos_x = pos % self.size
                    pos_y = pos // self.size
                    """
                    worker action
                    """
                    if state[pos_y][pos_x][17] == 1:
                        if nu[0] < 2:
                            target = nu[1]
                        elif state[pos_y][pos_x][6] == 1:
                            target = self.base
                        elif pos_x < 3 and pos_y < 3 and state[:3, :3, 14].sum() > 0:
                            ns = self.get_nearest_unit(pos, state, unit_type=14)
                            target = ns[1]
                        elif self.rally_point_worker > 0:
                            target = self.rally_point_worker
                        elif self.op_building_unit > 0:
                            target = self.op_building_unit
                        elif self.op_barracks > 0:
                            target = self.op_barracks
                        elif self.op_base > 0:
                            target = self.op_base
                        else:
                            target = nu[1]

                    """
                    base action
                    """
                    if state[pos_y][pos_x][15] == 1:
                        if (state[:3, :3, 17] * state[:3, :3, 11]).sum() < self.num_harvest_worker and state[:3, :3,
                                                                                                       14].sum() > 0:
                            target = self.resource[0]
                        else:
                            if self.num_battle_worker > 0:
                                target = nu[1]
                    """
                    barracks action
                    """
                    if state[pos_y][pos_x][16] == 1:
                        if self.op_barracks > 0:
                            target = self.op_barracks
                        elif self.op_base > 0:
                            target = self.op_base
                    """
                    soldiers action
                    """
                    if state[pos_y][pos_x][18] == 1 or state[pos_y][pos_x][19] == 1 or state[pos_y][pos_x][20] == 1:
                        if nu[0] < 4:
                            target = nu[1]
                        elif self.rally_point_soldier > 0 and state[:, :, 12].sum() > 3:
                            target = self.rally_point_soldier
                        else:
                            if self.op_building_unit > 0:
                                target = self.op_building_unit
                            elif self.op_barracks > 0:
                                target = self.op_barracks
                            elif self.op_base > 0:
                                target = self.op_base

                    action = astar_script(state, pos, target, self.size, masks[i][pos])[1:]
                    """
                    build barracks
                    """
                    if state[pos_y][pos_x][17] == 1 and masks[i][pos][4] == 1 and (
                            state[:, :, 16] * state[:, :, 11]).sum() == 0 and (
                            state[:, :, 17] * state[:, :, 11] * state[:, :, 25]).sum() == 0:
                        if pos - self.barracks == self.size:
                            action = np.array([4, 0, 0, 0, 0, 2, 0])
                        if pos - self.barracks == -1:
                            action = np.array([4, 1, 1, 1, 1, 2, 0])
                        if pos - self.barracks == -self.size:
                            action = np.array([4, 2, 2, 2, 2, 2, 0])
                        if pos - self.barracks == 1:
                            action = np.array([4, 3, 3, 3, 3, 2, 0])
                    if state[pos_y][pos_x][16] == 1:
                        action[5] = self.barracks_unit
                    actions[i][pos] = action
        return actions

    """
    check state
    """

    def check(self, state):
        op_base_state = state[:, :, 15] * state[:, :, 12]
        op_barracks_state = state[:, :, 16] * state[:, :, 12]
        op_building_unit = state[:, :, 17] * state[:, :, 12] * state[:, :, 25]
        for y in range(self.size):
            for x in range(self.size):
                if op_base_state[y][x] > 0:
                    self.op_base = self.size * y + x
                elif op_barracks_state[y][x] > 0:
                    self.op_barracks = self.size * y + x
                elif op_building_unit[y][x] > 0:
                    self.op_building_unit = self.size * y + x
        if op_base_state.sum() == 0:
            self.op_base = -1
        if op_barracks_state.sum() == 0:
            self.op_barracks = -1
        if op_building_unit.sum() == 0:
            self.op_building_unit = -1
        if (state[:, :, 16] * state[:, :, 12]).sum() > 0 or (
                state[:, :, 17] * state[:, :, 12] * state[:, :, 25]).sum() > 0:
            self.num_battle_worker = -1
            self.num_harvest_worker = 2
        else:
            self.num_battle_worker = 1
            self.num_harvest_worker = 1

    def get_nearest_unit(self, pos, state, unit_type=None):
        nearest_unit = (1000, -1)
        if unit_type is None:
            op_units = state[:, :, 12] - state[:, :, 15] * state[:, :, 12] - state[:, :, 16] * state[:, :, 12]
        elif unit_type == 14:
            op_units = state[:, :, 14]
        else:
            op_units = state[:, :, 12] * state[:, :, unit_type] - state[:, :, 15] * state[:, :, 12] - state[:, :,
                                                                                                      16] * state[:, :,
                                                                                                            12]
        pos_x = pos % self.size
        pos_y = pos // self.size
        for y in range(self.size):
            for x in range(self.size):
                if op_units[y][x] > 0:
                    dis = abs(pos_x - x) + abs(pos_y - y)
                    if dis < nearest_unit[0]:
                        nearest_unit = (dis, self.size * y + x)
        return nearest_unit

    def set_rally_point(self, state):
        if (state[:, :, 11] * (state[:, :, 18] + state[:, :, 19] + state[:, :, 20])).sum() > 5:
            self.rally_point_soldier = -1
        else:
            self.rally_point_soldier = 67

        if (state[:, :, 12] * (state[:, :, 18] + state[:, :, 19] + state[:, :, 20])).sum() == 0:
            self.rally_point_soldier = -1
        else:
            self.rally_point_soldier = 67
