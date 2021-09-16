# 2021-HYU-HAI-RLBootCamp

2021-HYU-HAI-RLBootCamp is the material(lecture notes, examples and assignments) repository for learning the intermediate level of Reinforcement Learning course that I'll teach the club 'HAI' at Hanyang University in the fall of 2021 ~ winter of 2022. Note that examples and assignments in this repository uses [PyTorch](https://pytorch.org/).

## Book

Deep Reinforcement Learning Hands-On - Second Edition (Packt, 2020)

![](https://static.packt-cdn.com/products/9781838826994/cover/smaller)

## Optional Readings

- Reinforcement Learning, An Introduction - Second Edition (MIT Press, 2018)
  - Korean: 단단한 강화학습 (제이펍, 2020)
- Reinforcement Learning (O'Reilly Media, 2020)
- 파이썬과 케라스로 배우는 강화학습 (위키북스, 2020)
- 바닥부터 배우는 강화 학습 (영진닷컴, 2020)

## Contents

- Week 1 (9/8) [[Lecture]](./1%20-%20Lecture/210908%20-%20RL%20Boot%20Camp%2C%20Week%201.pdf)
  - Review about the Basic Knowledge of Reinforcement Learning 
    - MDP (Markov Decision Process)
    - The Bellman Equation
    - Policy & Value Iteration
    - SARSA & Q-Learning
    - Policy Gradient
    - Deep Q-Network (DQN)
    - Actor-Critic
- Week 2 (9/24)
  - DQN Extensions #1
    - N-step DQN
    - Double DQN
    - Noisy Network
    - Prioritized Experience Replay (PER)
- Week 3 (9/29)
  - DQN Extensions #2
    - Dueling DQN
    - Categorical DQN
    - Rainbow
- Week 4 (10/4)
  - Actor-Critic Extensions
    - Advantage Actor-Critic (A2C)
    - Asynchronous Advantage Actor-Critic (A3C)
      - Data Parallelism
      - Gradient Parallelism
- Week 5 (11/1)
  - Example #1: Stock Trading
- Week 6 (11/8)
  - Continuous Action Space
    - Advantage Actor-Critic (A2C)
    - Deep Deterministic Policy Gradient (DDPG)
    - Distributed Distributional DDPG (D4PG)
- Week 7 (11/15)
  - Trust Regions #1
    - Trust Region Policy Optimization (TRPO)
    - Proximal Policy Optimization (PPO)
- Week 8 (11/22)
  - Trust Regions #2
    - Averaged Stochastic K-Timestep Trust Region (ACKTR)
    - Soft Actor-Critic (SAC)
- Week 9 (1/3)
  - Example #2: Robotics
- Week 10 (1/10)
  - Imagination
    - Interpretable Inference for Autonomous Agents (I2A)
- Week 11 (1/17)
  - Multi-Agent RL
    - Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
    - Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)
    - Multi-Agent Proximal Policy Optimization (MAPPO)
    - Multi-Agent Soft Actor-Critic (MASAC)
    - Monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning (QMIX)
- Week 12 (1/24)
  - Example #3: Rubik's Cube

## How To Contribute

Contributions are always welcome, either reporting issues/bugs or forking the repository and then issuing pull requests when you have completed some additional coding that you feel will be beneficial to the main project. If you are interested in contributing in a more dedicated capacity, then please contact me.

## Contact

You can contact me via e-mail (utilForever at gmail.com). I am always happy to answer questions or help with any issues you might have, and please be sure to share any additional work or your creations with me, I love seeing what other people are making.

## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright &copy; 2021-2022 [Chris Ohk](http://www.github.com/utilForever).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
