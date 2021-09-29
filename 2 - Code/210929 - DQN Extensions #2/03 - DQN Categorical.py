import argparse
import gym
import ptan
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine

from lib import dqn_extra, dqn_model, epsilon_tracker, hyper_params, utils


def calc_loss(batch, _net, _target_net, gamma, _device="cpu"):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(_device)
    actions_v = torch.tensor(actions).to(_device)
    next_states_v = torch.tensor(next_states).to(_device)

    next_distr_v, next_qvals_v = _target_net.both(next_states_v)
    next_acts = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = _target_net.apply_softmax(next_distr_v)
    next_distr = next_distr.data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_acts]
    dones = dones.astype(np.bool)

    proj_distr = dqn_extra.distr_projection(next_best_distr, rewards, dones, gamma)

    distr_v = _net(states_v)
    sa_vals = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(sa_vals, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(_device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


if __name__ == "__main__":
    random.seed(hyper_params.SEED)
    torch.manual_seed(hyper_params.SEED)

    params = hyper_params.HYPER_PARAMS['pong']

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(hyper_params.SEED)

    net = dqn_extra.CategoricalDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = epsilon_tracker.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine_for_batch, batch):
        optimizer.zero_grad()

        loss_v = calc_loss(batch, net, target_net.target_model, gamma=params.gamma, _device=device)
        loss_v.backward()

        optimizer.step()
        epsilon_tracker.frame(engine_for_batch.state.iteration)

        if engine_for_batch.state.iteration % params.target_net_sync == 0:
            target_net.sync()

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }


    engine = Engine(process_batch)
    utils.setup_ignite(engine, params, exp_source, "07_DQN_Categorical")
    engine.run(utils.batch_generator(buffer, params.replay_initial, params.batch_size))
