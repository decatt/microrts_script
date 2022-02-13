import torch
import random
import numpy as np
import time
import collections
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from agents.cnn_agent import RTSAgent, RTSNet
from stable_baselines3.common.vec_env import VecMonitor
from numpy.random import choice
import tensorboardX


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
    num_steps = 1000
    gamma = 0.99
    gae_lambda = 0.95
    clip_vloss = True
    update_epochs = 4
    minibatch_size = 5000
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5
    lr = lambda f: f * learning_rate

    device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
    # path_pt = './model/gnn53/2021122901.pt'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ais = []
    for i in range(num_envs):
        ais.append(microrts_ai.coacAI)

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
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.randomAI for _ in range(num_envs)],
        map_paths=[map_path for _ in range(num_envs)],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = VecMonitor(envs)
    next_obs = envs.reset()

    net = RTSNet().to(device).share_memory()
    agent = RTSAgent(net, action_space=[100, 6, 4, 4, 4, 4, 7, 49])

    # net.load_state_dict(torch.load(path_pt, map_location=device))

    tittle = 'GNN against worker rush in ' + str(size) + 'x' + str(size)

    actions = torch.zeros((num_steps, num_envs, 8))
    log_probs = torch.zeros((num_steps, num_envs))
    values = torch.zeros((num_steps, num_envs))
    rewards = torch.zeros((num_steps, num_envs))
    masks = torch.zeros((num_steps, num_envs, 178))
    states = torch.zeros((num_steps, num_envs, 10, 10, 27))
    ds = torch.zeros((num_steps, num_envs))
    advantages = torch.zeros((num_steps, num_envs))

    last_done = torch.zeros((num_envs,))
    learning_rate = 2.5e-4

    optimizer = torch.optim.Adam(params=agent.net.parameters(), lr=learning_rate, eps=1e-5)

    outcomes = collections.deque(maxlen=100)
    writer = tensorboardX.SummaryWriter()
    s = 0
    # writer.add_graph(net, torch.randn((num_envs, 10, 10, 27)).to(device))
    for update in range(1, 10001):
        frac = 1.0 - (update - 1.0) / 10000
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
        for step in range(num_steps):
            s = s+1
            action = np.zeros((num_envs, size*size, 7))
            envs.render()
            obs = next_obs
            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).float()
                action_plane_space = envs.action_plane_space.nvec
                unit_mask = torch.where(invalid_action_masks.sum(dim=2) > 0, 1.0, 0.0)
                unit_action, log_probs[step], masks[step], values[step] = agent.get_action(
                    torch.Tensor(obs).to(device), unit_masks=unit_mask, action_masks=invalid_action_masks)
                states[step] = torch.Tensor(obs)
            for i in range(num_envs):
                action[i][unit_action[i][0]] = unit_action[i][1:]
            next_obs, rs, done, infos = envs.step(action)
            ds[step] = torch.Tensor(done).float()
            rewards[step] = torch.tensor(rs)
            last_done = torch.Tensor(done).float()
            if done[0]:
                if get_units_number(11, torch.Tensor(obs), 0) > get_units_number(12, torch.Tensor(obs), 0):
                    outcomes.append(1)
                else:
                    outcomes.append(0)
                break
            writer.add_scalar('outcomes', sum(outcomes), s)
            writer.add_scalar('log_prob', log_probs[step].sum()/num_envs, s)
            writer.add_scalar('reward', sum(rs)/num_envs, s)
            writer.add_scalar('value', values[step].sum()/num_envs, s)

        with torch.no_grad():
            last_obs = torch.Tensor(next_obs).float().to(device)
            last_value = agent.get_value(last_obs).reshape(1, -1).cpu()
            last_done = last_done
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - last_done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - ds[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + values

        b_actions = actions.reshape((-1, 8)).to(device)
        b_log_probs = log_probs.reshape((-1,)).to(device)
        b_values = values.reshape((-1,)).to(device)
        b_rewards = rewards.reshape((-1,)).to(device)
        b_masks = masks.reshape((-1, 178)).to(device)
        b_obs = states.reshape((-1, 10, 10, 27)).to(device)
        b_ds = ds.reshape((-1,)).to(device)
        b_advantages = advantages.reshape((-1,)).to(device)
        b_returns = returns.reshape((-1,)).to(device)

        inds = np.arange(num_steps * num_envs, )
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, num_steps * num_envs, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_log_prob, entropy, new_values = agent.get_log_prob_entropy_value(
                    b_obs[minibatch_ind], b_actions[minibatch_ind], b_masks[minibatch_ind])

                ratio = (new_log_prob - b_log_probs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_log_probs[minibatch_ind] - new_log_prob).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                if clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(
                        new_values - b_values[minibatch_ind], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        if update % 50 == 0:
            date = time.strftime("%Y%m%d%H", time.localtime())
            path_pt = './model/cnn_ppo_' + date + '_worker.pt'
            torch.save(net.state_dict(), path_pt)

    writer.close()
