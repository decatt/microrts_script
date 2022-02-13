import torch
import random
import numpy as np
import time
import tensorboardX
import collections
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from agents.fun_agent import WorkerAgent, WorkerNet, ManagerAgent, ManagerNet
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from numpy.random import choice

import sys


# base barracks worker light heavy range
def embedding(states, num_env):
    zs = torch.zeros((num_env, 12))
    n = 0
    states = torch.Tensor(states.copy())
    for state in states:
        z = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        z[0] = (state[:, :, 11] * state[:, :, 15]).sum()
        z[1] = (state[:, :, 11] * state[:, :, 16]).sum()
        z[2] = (state[:, :, 11] * state[:, :, 17]).sum()
        z[3] = (state[:, :, 11] * state[:, :, 18]).sum()
        z[4] = (state[:, :, 11] * state[:, :, 19]).sum()
        z[5] = (state[:, :, 11] * state[:, :, 20]).sum()
        z[6] = (state[:, :, 12] * state[:, :, 15]).sum()
        z[7] = (state[:, :, 12] * state[:, :, 16]).sum()
        z[8] = (state[:, :, 12] * state[:, :, 17]).sum()
        z[9] = (state[:, :, 12] * state[:, :, 18]).sum()
        z[10] = (state[:, :, 12] * state[:, :, 19]).sum()
        z[11] = (state[:, :, 12] * state[:, :, 20]).sum()
        zs[n] = z
        n = n + 1
    return zs


def get_inner_reward(goal, c_obs, b_obs, num_env):
    res = torch.zeros((num_env,))
    v_zs = embedding(c_obs, num_env)
    v_zs_ = embedding(b_obs, num_env)
    for i_env in range(num_env):
        vector_a = v_zs_[i_env] - v_zs[i_env]
        vector_b = v_zs_[i_env] - goal[i_env]
        num = torch.mm(vector_a.view((1, 12)), vector_b.view((12, 1)))
        denom = torch.norm(vector_a) * torch.norm(vector_b)
        if denom!=0:
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            res[i_env] = sim[0]*torch.norm(v_zs_[i_env] - v_zs[i_env])
        else:
            res[i_env] = 0.0
    return res


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

    num_envs = 20
    num_steps = 512
    gamma = 0.99
    gae_lambda = 0.95

    device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
    path_pt = './model/gnn53/2021122901.pt'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ais = []
    for i in range(num_envs):
        ais.append(microrts_ai.naiveMCTSAI)

    size = 10
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
        ai2s=[microrts_ai.coacAI for _ in range(num_envs)],
        map_paths=[map_path for _ in range(num_envs)],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = VecMonitor(envs)
    next_obs = envs.reset()

    manager_net = ManagerNet().to(device)
    manager_agent = ManagerAgent(manager_net, num_envs)

    worker_net = WorkerNet().to(device)
    worker_agent = WorkerAgent(worker_net, action_space=[100, 6, 4, 4, 4, 4, 7, 49])

    outcomes = collections.deque(maxlen=100)
    worker_actions = torch.zeros((num_steps, num_envs, 8))
    worker_log_probs = torch.zeros((num_steps, num_envs))
    worker_values = torch.zeros((num_steps, num_envs))
    worker_rewards = torch.zeros((num_steps, num_envs))
    worker_masks = torch.zeros((num_steps, num_envs, 178))
    worker_obs = torch.zeros((num_steps, num_envs, 10, 10, 27))
    worker_ds = torch.zeros((num_steps, num_envs))
    worker_advantages = torch.zeros((num_steps,num_envs))
    worker_goals = torch.zeros((num_steps, num_envs, 12))

    # manager_step
    c = 8

    manager_actions = torch.zeros((num_steps//c, num_envs, 12))
    manager_log_probs = torch.zeros((num_steps//c, num_envs))
    manager_values = torch.zeros((num_steps//c, num_envs))
    manager_rewards = torch.zeros((num_steps//c, num_envs))
    manager_obs = torch.zeros((num_steps//c, num_envs, 10, 10, 27))
    manager_advantages = torch.zeros((num_steps//c, num_envs))
    manager_ds = torch.zeros((num_steps//c, num_envs))

    goals = torch.zeros((num_envs, 12))
    last_done = torch.zeros((num_envs, ))
    rewards = torch.zeros((num_envs, ))

    worker_learning_rate = 2.5e-4

    worker_optimizer = torch.optim.Adam(params=worker_net.parameters(), lr=worker_learning_rate)

    manager_learning_rate = 2.5e-4

    manager_optimizer = torch.optim.Adam(params=manager_net.parameters(), lr=manager_learning_rate)

    writer = tensorboardX.SummaryWriter()
    s = 0
    for update in range(100000):
        for step in range(num_steps):
            s = s+1
            manager_step = step//c
            action = np.zeros((num_envs, size*size, 7))
            envs.render()
            obs = next_obs
            if step % c == 0 or step == 0:
                with torch.no_grad():
                    goals, manager_values[manager_step], manager_log_probs[manager_step] = manager_agent.get_goal(torch.Tensor(obs).to(device))
                    manager_actions[manager_step] = goals.float()
                    manager_obs[manager_step] = torch.Tensor(obs)

            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).float()
                action_plane_space = envs.action_plane_space.nvec
                unit_mask = torch.where(invalid_action_masks.sum(dim=2) > 0, 1.0, 0.0)

                unit_action, worker_log_probs[step], worker_masks[step], worker_values[step] = worker_agent.get_action(
                    torch.Tensor(obs).to(device), goals.to(device), unit_masks=unit_mask, action_masks=invalid_action_masks)

                worker_actions[step] = torch.Tensor(unit_action)
                worker_obs[step] = torch.Tensor(obs)
            for i in range(num_envs):
                action[i][unit_action[i][0]] = unit_action[i][1:]
            next_obs, rs, ds, infos = envs.step(action)
            rewards = rewards + torch.Tensor(rs)
            if step % c == 0 or step == 0:
                manager_rewards[manager_step] = rewards
                manager_ds[manager_step] = torch.Tensor(ds)
                rewards = torch.zeros((num_envs,))
            worker_ds[step] = torch.Tensor(ds)
            inner_reward = get_inner_reward(goals, next_obs, obs, num_envs)
            worker_rewards[step] = inner_reward
            last_done = torch.Tensor(ds)

            if ds[0]:
                if get_units_number(11, torch.Tensor(obs), 0) > get_units_number(12, torch.Tensor(obs), 0):
                    outcomes.append(1)
                else:
                    outcomes.append(0)
                break

            writer.add_scalar('outcomes', sum(outcomes), s)
            writer.add_scalar('rewards', sum(rs)/num_envs, s)

        with torch.no_grad():
            last_obs = torch.Tensor(next_obs).float().to(device)
            worker_last_value = worker_agent.get_value(last_obs, goals.to(device)).reshape(1, -1).cpu()
            worker_last_done = last_done
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    worker_next_non_terminal = 1.0 - worker_last_done
                    worker_next_values = worker_last_value
                else:
                    worker_next_non_terminal = 1.0 - worker_ds[t + 1]
                    worker_next_values = worker_values[t + 1]
                worker_delta = worker_rewards[t] + gamma * worker_next_values * worker_next_non_terminal - worker_values[t]
                worker_advantages[t] = last_gae_lam = worker_delta + gamma * gae_lambda * worker_next_non_terminal * last_gae_lam
            worker_returns = worker_advantages + worker_values

        with torch.no_grad():
            last_obs = torch.Tensor(next_obs).float().to(device)
            manager_last_value = manager_agent.get_value(last_obs).reshape(1, -1).cpu()
            manager_last_done = last_done
            last_gae_lam = 0
            for t in reversed(range(num_steps//c)):
                if t == num_steps//c - 1:
                    manager_next_non_terminal = 1.0 - manager_last_done
                    manager_next_values = manager_last_value
                else:
                    manager_next_non_terminal = 1.0 - manager_ds[t + 1]
                    manager_next_values = manager_values[t + 1]
                manager_delta = manager_rewards[t] + gamma * manager_next_values * manager_next_non_terminal - manager_values[t]
                manager_advantages[t] = last_gae_lam = manager_delta + gamma * gae_lambda * manager_next_non_terminal * last_gae_lam
            manager_returns = manager_advantages + manager_values

        # worker training
        clip_vloss = True
        update_epochs = 4
        minibatch_size = 32
        clip_coef = 0.1
        ent_coef = 0.01
        vf_coef = 0.5

        b_worker_actions = worker_actions.reshape((-1, 8)).to(device)
        b_worker_log_probs = worker_log_probs.reshape((-1,)).to(device)
        b_worker_values = worker_values.reshape((-1,)).to(device)
        b_worker_rewards = worker_rewards.reshape((-1,)).to(device)
        b_worker_masks = worker_masks.reshape((-1, 178)).to(device)
        b_worker_obs = worker_obs.reshape((-1, 10, 10, 27)).to(device)
        b_worker_ds = worker_ds.reshape((-1,)).to(device)
        b_worker_advantages = worker_advantages.reshape((-1, )).to(device)
        b_worker_returns = worker_returns.reshape((-1, )).to(device)
        b_worker_goals = worker_goals.reshape((-1, 12)).to(device)

        inds = np.arange(num_steps*num_envs, )
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, num_steps*num_envs, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_worker_advantages[minibatch_ind]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_log_prob, entropy, new_values = worker_agent.get_log_prob_entropy_value(
                    b_worker_obs[minibatch_ind], b_worker_goals[minibatch_ind], b_worker_actions[minibatch_ind], b_worker_masks[minibatch_ind])

                ratio = (new_log_prob - b_worker_log_probs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_worker_log_probs[minibatch_ind] - new_log_prob).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                if clip_vloss:
                    v_loss_unclipped = ((new_values - b_worker_returns[minibatch_ind]) ** 2)
                    v_clipped = b_worker_values[minibatch_ind] + torch.clamp(new_values - b_worker_values[minibatch_ind], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_worker_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_worker_returns[minibatch_ind]) ** 2)

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                worker_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker_net.parameters(), 0.5)
                worker_optimizer.step()

        b_manager_actions = manager_actions.reshape((-1, 12)).to(device)
        b_manager_log_probs = manager_log_probs.reshape((-1,)).to(device)
        b_manager_values = manager_values.reshape((-1,)).to(device)
        b_manager_rewards = manager_rewards.reshape((-1,)).to(device)
        b_manager_obs = manager_obs.reshape((-1, 10, 10, 27)).to(device)
        b_manager_ds = manager_ds.reshape((-1,)).to(device)
        b_manager_advantages = manager_advantages.reshape((-1,)).to(device)
        b_manager_returns = manager_returns.reshape((-1,)).to(device)

        inds = np.arange((num_steps//c) * num_envs, )
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, (num_steps//c) * num_envs, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_manager_advantages[minibatch_ind]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_values, new_log_prob, entropy = manager_agent.get_entropy(b_manager_obs[minibatch_ind], b_manager_actions[minibatch_ind])
                ratio = (new_log_prob - b_manager_log_probs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_manager_log_probs[minibatch_ind] - new_log_prob).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                if clip_vloss:
                    v_loss_unclipped = ((new_values - b_manager_returns[minibatch_ind]) ** 2)
                    v_clipped = b_manager_values[minibatch_ind] + torch.clamp(
                        new_values - b_manager_values[minibatch_ind], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_manager_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_manager_returns[minibatch_ind]) ** 2)

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                manager_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(manager_net.parameters(), 0.5)
                manager_optimizer.step()

        if update % 100 == 0:
            print('update: '+str(update)+' outcomes: '+str(sum(outcomes)))
            date = time.strftime("%Y%m%d%H", time.localtime())
            path_pt1 = './model/fun_ppo/worker_' + date + '.pt'
            torch.save(worker_net.state_dict(), path_pt1)
            path_pt2 = './model/fun_ppo/manager_' + date + '.pt'
            torch.save(manager_net.state_dict(), path_pt2)

    writer.close()


