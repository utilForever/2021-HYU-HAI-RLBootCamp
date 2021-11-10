import argparse
import gym
import os
import ptan

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import utils

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 8

GRAD_BATCH = 64
TRAIN_BATCH = 2

REWARD_BOUND = 18


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def grads_func(_proc_name, _net, _device, _train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]

    agent = ptan.agent.PolicyAgent(lambda x: _net(x)[0], device=_device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    _batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=_proc_name)

    with utils.RewardTracker(writer, REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, 100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()

                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break

                _batch.append(exp)
                if len(_batch) < GRAD_BATCH:
                    continue

                data = utils.unpack_batch(_batch, _net, device=_device, last_val_gamma=GAMMA ** REWARD_STEPS)
                states_v, actions_t, vals_ref_v = data

                _batch.clear()

                _net.zero_grad()
                logits_v, value_v = _net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_p_a = log_prob_v[range(GRAD_BATCH), actions_t]
                log_prob_actions_v = adv_v * log_p_a
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                ent = (prob_v * log_prob_v).sum(dim=1).mean()
                entropy_loss_v = ENTROPY_BETA * ent

                loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                loss_v.backward()

                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)

                nn_utils.clip_grad_norm_(_net.parameters(), CLIP_GRAD)
                grads = [
                    param.grad.data.cpu().numpy()
                    if param.grad is not None else None
                    for _param in _net.parameters()
                ]
                _train_queue.put(grads)

    _train_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"

    env = make_env()
    net = utils.A2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    for proc_idx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}#{proc_idx}"
        p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()