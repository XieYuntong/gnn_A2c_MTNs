"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
from collections import namedtuple


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, scale_factor=0.01):
        super().__init__()  # 调用父类的__init__构造函数，并使用相同的参数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.env = env
        self.T = T
        self.s = scale_factor  # 缩放比例

    def parse_obs(self, obs):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time + 1] *
                          self.s for n in self.env.station]).view(1, 1, self.env.num_stations).float(),
            torch.tensor([[(obs[0][n][self.env.time + 1] + self.env.dacc[n][t]) * self.s for n in self.env.station] \
                          for t in range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T,
                                                                                               self.env.num_stations).float(),
            torch.tensor([[sum([(self.env.scenario.demand[i, j][t]) * self.s
                                for j in self.env.station]) for i in self.env.station] for t in
                          range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T,
                                                                                      self.env.num_stations).float()),
            dim=1).squeeze(0).view(2*self.T + 1, self.env.num_stations).T
        # 第一个维度是批次大小，第二个维度是时间步，处理的是单个时间步的数据, 第三个维度是站点的数量

        edge_index = []
        node_set = set()

        # 遍历ordered_stops并构建edge_index
        for line, stops in self.env.ordered_stops.items():
            for i in range(len(stops) - 1):
                src, dst = stops[i], stops[i + 1]
                # 添加源节点和目标节点到node_set中
                node_set.add(src)
                node_set.add(dst)
                # 添加边到edge_index中
                edge_index.append([src, dst])
                edge_index.append([dst, src])

        # 去重
        edge_index = list(set(tuple(edge) for edge in edge_index))

        # 给节点分配ID
        node2id = {node: i for i, node in enumerate(sorted(list(node_set)))}

        # 重新映射边上的节点ID
        edge_index = [[node2id[edge[0]], node2id[edge[1]]] for edge in edge_index]

        # 转换成torch.tensor格式并转换成无向图
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = utils.to_undirected(edge_index)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        data = Data(x, edge_index)

        return data

#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)  # 创建了一个GCNConv图卷积层实例对象
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 2)
        # 定义了三个全连接层（linear layer）

    def forward(self, data):  # 神经网络的前向传播过程
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        out = F.relu(self.conv1(data.x, data.edge_index))  # 激活函数：加入非线性特性
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        # 将节点特征和邻接关系编码为一个实数，用于输出每个节点的动作概率
        return x


#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 2)  # 两个动作的全连接层


    def forward(self, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)  # Sum-pool
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the MTNs control problem.
    """

    def __init__(self, env, input_size, eps=np.finfo(np.float32).eps.item(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device

        self.actor = GNNActor(self.input_size, self.hidden_size)  # 输出一个概率分布，用于选择下一步的行动。
        self.critic = GNNCritic(self.input_size, self.hidden_size)  # 输出当前状态的估计价值，用于评估当前状态的好坏
        self.obs_parser = GNNParser(self.env)  # 解析观测到的状态

        self.optimizers = self.configure_optimizers()  # 配置优化器

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.available_stations = torch.tensor([int(x) for x in list(self.env.selected_stations)], dtype=torch.long, device=self.device)
        self.to(self.device)

    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        device_str = str(self.device)
        x = self.parse_obs(obs).to(device_str)

        a_out = self.actor(x)
        a_out1, a_out2 = torch.split(a_out, 1, dim=1)

        # actor: computes concentration parameters of a Dirichlet distribution
        concentration = F.softplus(a_out1).reshape(-1) + jitter  # 保持正值

        # using a distribution with probabilities given by the softplus of the actor's output
        action2_probs = F.softplus(a_out2).reshape(-1)
        action2_probs = action2_probs / torch.sum(action2_probs)  # 归一化

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value, action2_probs

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, obs):
        concentration, value, action2_probs = self.forward(obs)
        available_stations = self.available_stations
        available_stations = list(available_stations)
        # Map the available station indices to the corresponding indices in concentration
        station_indices = [list(self.env.G.nodes()).index(s) for s in available_stations]
        # Sample action 1: desired proportion of mods at each available station
        concentration = concentration[station_indices]
        m1 = Dirichlet(concentration)
        action1 = m1.sample()

        # Sample action 2: a discrete action from {-3, -2, -1, 0, 1, 2, 3}
        action2_probs = action2_probs[station_indices]
        action2_probs = action2_probs / torch.sum(action2_probs)  # Normalize
        action2 = torch.zeros(len(available_stations), dtype=torch.long, device=self.device)
        action2_values = torch.multinomial(action2_probs, num_samples=len(available_stations), replacement=True) - 3
        action2_values = torch.clamp(action2_values, -3, 3)
        action2[torch.arange(len(available_stations))] = torch.tensor(action2_values, dtype=torch.long, device=self.device).clone().detach()
        # Save both actions and their log probabilities
        m1_log_prob = m1.log_prob(action1)
        action2_probs = torch.clamp(action2_probs, min=1e-6, max=1 - 1e-6)
        action2_log_prob = torch.sum(torch.log(action2_probs + 1e-8), dim=0)
        joint_log_prob = m1_log_prob + action2_log_prob
        # save both actions and their log probabilities
        self.saved_actions.append(SavedAction(joint_log_prob, value))

        return action1.tolist(), action2.tolist()

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss for action
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            value_mean = torch.mean(value)
            R = torch.tensor(R, dtype=torch.float32).to(self.device)
            advantage = R - value_mean.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value_mean, torch.tensor(R, dtype=torch.float32).to(self.device)))

        # take gradient steps
        batch_size_p = len(policy_losses)  # Assuming policy_losses and value_losses have the same length
        batch_size_v = len(value_losses)

        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum() / batch_size_p
        a_loss.backward()
        # Apply gradient clipping for actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizers['a_optimizer'].step()

        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum() / batch_size_v
        v_loss.backward()
        # Apply gradient clipping for critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizers['c_optimizer'].step()

        total_loss = abs(v_loss) + abs(a_loss)

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        # 返回总损失
        return total_loss.item()

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-4, weight_decay=1e-5)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-4, weight_decay=1e-5)
        return optimizers

    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)





