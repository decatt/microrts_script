import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from agents.aster_script_agent import ScriptAgent
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from numpy.random import choice
import sys
import time


def sample(l):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(l) == 0:
        return 0
    return choice(range(len(l)), p=l / sum(l))


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


def init_seeds(torch_seed=0, seed=0):
    torch.manual_seed(torch_seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(torch_seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(torch_seed)  # Sets the seed for generating random numbers on all GPUs.
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed = 0

    num_envs = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
    path_pt = './model/cnn_ppo_2022012906_worker.pt'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    size = 16
    map_path = "maps/24x24/basesWorkers24x24.xml"
    if size == 10:
        map_path = "maps/10x10/basesWorkers10x10.xml"
    elif size == 16:
        map_path = "maps/16x16/basesWorkers16x16.xml"

    init_seeds()
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=num_envs,
        max_steps=5000,
        render_theme=2,
        ai2s=[microrts_ai.mixedBot for _ in range(num_envs)],
        map_paths=[map_path for _ in range(num_envs)],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = VecMonitor(envs)
    next_obs = envs.reset()

    outcomes = []

    resource1 = 0

    sa = ScriptAgent(size=16, base=34, barracks=48,  op_base=221, resource=[0, 16])
    for games in range(100):
        for step in range(5000):
            envs.render()
            masks = np.array(envs.get_action_mask())
            action = sa.get_action(next_obs, masks)
            next_obs, rs, ds, infos = envs.step(action)
            time.sleep(0.01)

        print("\r", end="")
        print("Progress: {}%: ".format(games), "▋" * (games // 2), end="")
        sys.stdout.flush()
    print(sum(outcomes) / len(outcomes))
    print('end')