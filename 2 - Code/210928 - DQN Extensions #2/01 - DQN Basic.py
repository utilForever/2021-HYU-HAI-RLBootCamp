import argparse
import gym
import ptan
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, epsilon_tracker, hyper_params, utils

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

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = epsilon_tracker.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine_for_batch, batch):
        optimizer.zero_grad()

        loss_v = utils.calc_loss_dqn(batch, net, target_net.target_model, gamma=params.gamma, device=device)
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
    utils.setup_ignite(engine, params, exp_source, "01_DQN_Baseline")
    engine.run(utils.batch_generator(buffer, params.replay_initial, params.batch_size))
