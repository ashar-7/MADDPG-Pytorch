from copy import deepcopy
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np

import memory
from model import ActorNet, CriticNet

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


class MADDPG_Agent:
    def __init__(self, env, batch_size, replay_capacity, episodes_before_train, device='cpu'):

        self.env = env
        self.n_agents = env.n
        self.memory = memory.ReplayMemory(replay_capacity)

        self.actors = [ActorNet(env.observation_space[i].shape[0], env.action_space[i].n) for i in range(self.n_agents)]
        self.critics = [CriticNet(env.observation_space[i].shape[0], env.n) for i in range(self.n_agents)]

        self.critic_optimizers = [Adam(x.parameters(), lr=0.01) for x in self.critics]
        self.actor_optimizers = [Adam(x.parameters(), lr=0.01) for x in self.actors]

        self.actor_targets = deepcopy(self.actors)
        self.critic_targets = deepcopy(self.critics)

        self.device = device
        self.episodes_before_train = episodes_before_train
        self.batch_size = batch_size

        self.GAMMA = 0.95
        self.epsilon = 0.3

        for x in self.actors:           x.to(device)
        for x in self.critics:          x.to(device)
        for x in self.actor_targets:    x.to(device)
        for x in self.critic_targets:   x.to(device)

    def select_actions(self, nets, states, noise=False):
        # TODO: actions as 1-hot arrays or logits or probabilities (FIND)
        actions = []
        for actor, state in zip(nets, states):
            state_v = torch.from_numpy(state).float().to(self.device)
            action = actor(state_v).data.cpu()
            if noise:
                action = action.numpy() + self.epsilon * np.random.normal(size=action.shape)
            else:
                action = action.numpy()
            action = np.clip(action, -1, 1)
            action_a = np.zeros(action.shape)
            action_a[action.argmax()] = 1
            actions.append(action_a)

        return actions

    def get_all_actions(self, nets, states_batch):
        # I know the select_actions() and get_all_actions() sound similar but I can't think of a good name
        next_actions = []
        for states in states_batch:
            actions = self.select_actions(nets, states)
            next_actions.append(actions)
        next_actions_a = np.array(next_actions).argmax(axis=2)
        all_next_actions_v = torch.from_numpy(next_actions_a).float().to(self.device)

        return all_next_actions_v

    def update(self, episodes_done):
        if episodes_done <= self.episodes_before_train:
            return

        sampled_batch = self.memory.sample(2)
        batch = memory.Experience(*zip(*sampled_batch))

        states_batch = np.array(batch.states)
        actions_batch = np.array(batch.actions)
        next_states_batch = np.array(batch.next_states)
        rewards_batch = np.array(batch.rewards)
        dones_batch = np.array(batch.dones)

        all_next_actions_v = self.get_all_actions(self.actor_targets, next_states_batch)
        all_actions_main_v = self.get_all_actions(self.actors, states_batch)
        all_actions_v = actions_batch.argmax(axis=2)
        all_actions_v = torch.from_numpy(all_actions_v).float().to(self.device)

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            self.critic_optimizers[agent].zero_grad()

            states_a = np.stack(states_batch[:, agent])
            actions_a = np.stack(actions_batch[:, agent])
            next_states_a = np.stack(next_states_batch[:, agent])
            rewards_a = np.stack(rewards_batch[:, agent]).reshape((2,1))
            dones_a = np.stack(dones_batch[:, agent]).reshape((2,1))

            next_states_v = torch.from_numpy(next_states_a).float().to(self.device)
            states_v = torch.from_numpy(states_a).float().to(self.device)
            rewards_v = torch.from_numpy(rewards_a).float().to(self.device)
            dones_v = torch.ByteTensor(dones_a).to(self.device)

            # update critic:
            Q_sa_target = self.critic_targets[agent](next_states_v, all_next_actions_v)
            Q_sa_target = Q_sa_target
            Q_sa_target[dones_v] = 0.0
            Y = rewards_v + self.GAMMA * Q_sa_target

            Q_sa = self.critics[agent](states_v, all_actions_v)

            critic_loss = nn.MSELoss()(Y, Q_sa.detach())
            critic_loss.backward()
            self.critic_optimizers[agent].step()

            # update actor:
            self.actor_optimizers[agent].zero_grad()
            action_i = self.actors[agent](states_v).data.cpu().numpy()
            action_i = action_i.argmax(axis=1)
            action_v = torch.from_numpy(action_i).float().to(self.device)
            all_actions_v[:, agent] = action_v
            actor_loss = -self.critics[agent](states_v, all_actions_v)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizers[agent].step()

            c_loss.append(critic_loss)
            a_loss.append(actor_loss)

        return (c_loss, a_loss)

    def soft_update(self, target, source, t):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def train(self, n_episodes, max_episode_length=100):
        total_steps = 0
        for i_episode in range(n_episodes):

            states = env.reset()
            steps = 0
            total_reward = 0
            for t in range(max_episode_length):
                if i_episode % 100 == 0:
                    env.render()

                actions = self.select_actions(self.actor_targets, states, noise=True)
                next_states, rewards, dones, _ =  env.step(actions)
                total_reward += sum(rewards)

                if (t+1) == max_episode_length or all(dones):
                    dones = [1 for _ in range(env.n)]
                if not all(dones):
                    dones = [0 for _ in range(env.n)]

                self.memory.push(states, actions, next_states, rewards, dones)

                states = next_states
                self.update(i_episode)

                if total_steps % 100 == 0 and total_steps > 0:
                    for i in range(self.n_agents):
                        self.soft_update(self.actor_targets[i], self.actors[i], t=0.01)
                        self.soft_update(self.critic_targets[i], self.critics[i], t=0.01)

                if all(dones):
                    break

                steps += 1
                total_steps += 1

            print('Episode: %d, reward = %f' % (i_episode, total_reward))


if __name__ == '__main__':
    device = 'cpu'
    env = make_env('simple_tag')
    agent = MADDPG_Agent(env, 64, 10000, 10, device=device)
    agent.train(1000, 200)
