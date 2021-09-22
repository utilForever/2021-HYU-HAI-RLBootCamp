import argparse
import gym
import ptan
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from ignite.engine import Engine

from lib import dqn_model, epsilon_tracker, hyper_params, utils

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


def calc_loss_double_dqn(batch, _net, _target_net, gamma, _device="cpu", double=True):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)

    states_v = torch.tensor(states).to(_device)
    actions_v = torch.tensor(actions).to(_device)
    rewards_v = torch.tensor(rewards).to(_device)
    done_mask = torch.BoolTensor(dones).to(_device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(_device)

        if double:
            next_state_acts = net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = _target_net(next_states_v).gather(1, next_state_acts).squeeze(-1)
        else:
            next_state_vals = _target_net(next_states_v).max(1)[0]

        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v

    return nn.MSELoss()(state_action_vals, exp_sa_vals)


if __name__ == "__main__":
    random.seed(hyper_params.SEED)
    torch.manual_seed(hyper_params.SEED)

    params = hyper_params.HYPER_PARAMS['pong']

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
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

        loss_v = calc_loss_double_dqn(batch, net, target_net.target_model, gamma=params.gamma,
                                      _device=str(device), double=args.double)
        loss_v.backward()

        optimizer.step()
        epsilon_tracker.frame(engine_for_batch.state.iteration)

        if engine_for_batch.state.iteration % params.target_net_sync == 0:
            target_net.sync()

        if engine_for_batch.state.iteration % EVAL_EVERY_FRAME == 0:
            eval_states = getattr(engine_for_batch.state, "eval_states", None)

            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
                engine_for_batch.state.eval_states = eval_states

            engine_for_batch.state.metrics["values"] = utils.calc_values_of_states(eval_states, net, device)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }


    engine = Engine(process_batch)
    utils.setup_ignite(engine, params, exp_source, f"03_DQN_Double={args.double}", extra_metrics=('values',))
    engine.run(utils.batch_generator(buffer, params.replay_initial, params.batch_size))
