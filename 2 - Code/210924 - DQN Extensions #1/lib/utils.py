import torch
import torch.nn as nn

import ptan
import ptan.ignite as ptan_ignite

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger

from datetime import timedelta, datetime

from types import SimpleNamespace
from typing import Iterable, Tuple, List

import numpy as np
import warnings


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [], [], [], [], []

    for exp in batch:
        state = np.array(exp.state)

        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)

        if exp.last_state is None:
            last_state = state
        else:
            last_state = np.array(exp.last_state)

        last_states.append(last_state)

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, target_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    with torch.no_grad():
        next_state_vals = target_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)

        print("Episode %d: reward=%.0f, steps=%s, speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']

        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    log_dir = f"runs/{now}-{params.run_name}-{run_name}"
    tensorboard = tensorboard_logger.TensorboardLogger(log_dir=log_dir)

    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tensorboard_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tensorboard.attach(engine, log_handler=handler, event_name=event)

    ptan_ignite.PeriodicEvents().attach(engine)

    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tensorboard_logger.OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tensorboard.attach(engine, log_handler=handler, event_name=event)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)

    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)
