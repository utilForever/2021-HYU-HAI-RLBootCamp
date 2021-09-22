import argparse
import gym
import ptan
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, dqn_extra, epsilon_tracker, hyper_params, utils

NOISY_SNR_EVERY_ITERS = 100

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

    net = dqn_extra.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine_for_batch, batch):
        optimizer.zero_grad()

        loss_v = utils.calc_loss_dqn(batch, net, target_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()

        optimizer.step()

        if engine_for_batch.state.iteration % params.target_net_sync == 0:
            target_net.sync()

        if engine_for_batch.state.iteration % NOISY_SNR_EVERY_ITERS == 0:
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                engine_for_batch.state.metrics[f'snr_{layer_idx + 1}'] = sigma_l2

        return {
            "loss": loss_v.item(),
        }


    engine = Engine(process_batch)
    utils.setup_ignite(engine, params, exp_source, "04_DQN_NoisyNetwork", extra_metrics=('snr_1', 'snr_2'))
    engine.run(utils.batch_generator(buffer, params.replay_initial, params.batch_size))
