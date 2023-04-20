from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")

from src.env.Env import Scenario, MTNs
from src.algos.a2c_gnn import A2C
from src.misc.utils import dictsum

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=12, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--json_hr', type=int, default=7, metavar='S',
                    help='json_hr (default: 7)')
parser.add_argument('--json_tsetp', type=int, default=1, metavar='S',
                    help='minutes per timestep (default: 1min)')
parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--cplexpath', type=str, default='C:/Program Files/IBM/ILOG/CPLEX_Studio221/opl/bin/x64_win64/',
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=100, metavar='N',
                    help='number of episodes to train agent (default: 16k)')  # 最大回合数
parser.add_argument('--max_steps', type=int, default=60, metavar='N',
                    help='number of steps per episode (default: T=60)')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Define MTNs Simulator Environment
scenario = Scenario(sd=args.seed, tstep=args.json_tsetp)
env = MTNs(scenario)
# Initialize A2C-GNN
model = A2C(env=env, input_size=21).to(device)

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################

    # Initialize lists for logging
    log = {'train_reward': [],
           'train_served_demand': [],
           'train_reb_cost': [],
           'train_operate_cost': [],
           'train_wait_cost': []}
    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator（进度条）
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode
    loss_history = []
    total_rewards = []

    # Create directory for saving checkpoints if it doesn't exist
    ckpt_directory = './saved_files/ckpt/nyc4/'
    if not os.path.exists(ckpt_directory):
        os.makedirs(ckpt_directory)

    # Create directory for saving logs if it doesn't exist
    log_directory = './saved_files/rl_logs/nyc4/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    for i_episode in epochs:
        obs = env.reset()  # initialize environment
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_operating_cost = 0
        episode_waitingtime_cost = 0

        for step in range(T):
            # use GNN-RL policy (Step 1 in paper)
            action1_rl, action2_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {list(env.selected_stations)[i]: int(action1_rl[i] * dictsum(env.acc, env.time)) for i in
                          range(len(env.selected_stations))}
            desiredAdd = {list(env.selected_stations)[i]: int(action2_rl[i]) for i in range(len(env.selected_stations))}
            # take matching step (Step 2 in paper)
            modAction, rebAction, alphaAction, passengerAction = env.optimize(CPLEXPATH=args.cplexpath, PATH='scenario_nyc4', desiredAcc=desiredAcc, desiredAdd=desiredAdd)
            obs, paxreward, done, info = env.pax_step(modAction=modAction, rebAction=rebAction, alphaAction=alphaAction, passengerAction=passengerAction)
            episode_reward += paxreward

            # Store the transition in memory
            model.rewards.append(paxreward)
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_operating_cost += info['operating_cost']
            episode_waitingtime_cost += info['waitingtime_cost']
            # stop episode if terminating conditions are met
            if done:
                break

        # 调用模型的training_step()方法并获取损失值
        loss = model.training_step()

        # 将损失值添加到损失历史中
        loss_history.append(loss)
        total_rewards.append(episode_reward)

        # Send current statistics to screen
        epochs.set_description(f"Episode {i_episode + 1} | Reward: {episode_reward:.2f} | "
                               f"ServedDemand: {episode_served_demand:.2f} | "
                               f"Reb. Cost: {episode_rebalancing_cost:.2f} | "
                               f"\nOperate. Cost: {episode_operating_cost:.2f} | "
                               f"Training. Loss: {loss:.2f}")

        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(path=f"./{args.directory}/ckpt/nyc4/a2c_gnn_test.pth")
            best_reward = episode_reward
        # Log KPIs
        log['train_reward'].append(episode_reward)
        log['train_served_demand'].append(episode_served_demand)
        log['train_reb_cost'].append(episode_rebalancing_cost)
        log['train_operate_cost'].append(episode_operating_cost)
        log['train_wait_cost'].append(episode_waitingtime_cost)
        model.log(log, path=f"./{args.directory}/rl_logs/nyc4/a2c_gnn_test.pth")

    # 绘制损失值图
    plt.plot(loss_history)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    print(loss_history)
    # 保存图像到文件
    plt.savefig('D:/Python/PycharmProjects/pythonProject/gnn_rl_MTNs/training_loss.png')

    plt.clf()  # 清除之前的图像
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.savefig('D:/Python/PycharmProjects/pythonProject/gnn_rl_MTNs/training_reward.png')


else:
    # Load pre-trained model
    model.load_checkpoint(path=f"./{args.directory}/ckpt/nyc4/a2c_gnn.pth")
    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {'test_reward': [],
           'test_served_demand': [],
           'test_reb_cost': [],
           'test_operate_cost': [],
           'test_wait_cost': []}
    for episode in epochs:
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_operating_cost = 0
        episode_waitingtime_cost = 0
        obs = env.reset()
        done = False
        k = 0
        while(not done):
            # use GNN-RL policy (Step 1 in paper)
            action1_rl, action2_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {env.station[i]: int(action1_rl[i] * dictsum(env.acc, env.time + 1)) for i in
                          range(len(env.station))}
            desiredAdd = {env.station[i]: int(action2_rl[i]) for i in range(len(env.station))}

            # take matching step (Step 2 in paper)
            modAction, rebAction, alphaAction, passengerAction = env.optimize(CPLEXPATH=args.cplexpath,
                                                                              PATH='scenario_nyc4',
                                                                              desiredAcc=desiredAcc,
                                                                              desiredAdd=desiredAdd)
            obs, paxreward, done, info = env.pax_step(modAction=modAction, rebAction=rebAction, alphaAction=alphaAction,
                                                      passengerAction=passengerAction)
            episode_reward += paxreward

            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_operating_cost += info['operating_cost']
            episode_waitingtime_cost += info['waitingtime_cost']
            k += 1
        # Send current statistics to screen
        epochs.set_description(f"Episode {episode + 1} | Reward: {episode_reward:.2f} |"
                               f" ServedDemand: {episode_served_demand:.2f} |\n"
                               f" Reb. Cost: {episode_rebalancing_cost:.2f} |"
                               f" Operate. Cost: {episode_operating_cost:.2f} |")

        # Log KPIs
        log['test_reward'].append(episode_reward)
        log['test_served_demand'].append(episode_served_demand)
        log['test_reb_cost'].append(episode_rebalancing_cost)
        log['test_operate_cost'].append(episode_operating_cost)
        log['test_wait_cost'].append(episode_waitingtime_cost)
        model.log(log, path=f"./{args.directory}/rl_logs/nyc4/a2c_gnn_test.pth")
        break




