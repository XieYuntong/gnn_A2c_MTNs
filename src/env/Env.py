from collections import defaultdict
import numpy as np
import subprocess
import os
import networkx as nx
from gnn_rl_MTNs.src.misc.utils import mat2str
from copy import deepcopy
import pandas as pd
import random
from numpy.random import uniform


class MTNs:
    # initialization
    def __init__(self, scenario, runcost=0.5, recost=0.2):  # updated to take scenario and beta (cost for rebalancing each minute)
        self.scenario = deepcopy(scenario)  # the scenario input is not modified by env
        self.G = scenario.G  # Road Graph: node - station, edge - connection of stations, node attr: 'accInit', edge attr: 'time'
        self.rebTime = self.scenario.rebTime
        self.selected_stations = self.scenario.selected_stations
        self.ltime = self.scenario.ltime
        self.line_time = self.scenario.line_time
        self.ordered_stops = self.scenario.ordered_stops
        self.headWay = self.scenario.headWay
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.tcost = 0.08  # time cost of waiting time TODO: decide the time cost
        self.modCapacity = 6
        self.transTime = self.scenario.transTime

        self.station = list(self.G)  # set of stations

        for i in self.station:
            self.depDemand[i] = defaultdict(float)

        self.busLine = scenario.busLine  # Input the info of busline: l-name of busline, edge-pair of station on each l

        self.acc = defaultdict(dict)  # number of mods within each station, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of rebalancing mods arriving at each station, key: i - region, t - time
        self.rebFlow = defaultdict(dict)  # number of rebalancing mods, key: (i,j) - (origin, destination), t - time
        self.podsFlow = defaultdict(dict)  # number of mods
        self.modFlow = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # number of dispatching mods, key:(i,l) - (origin, chosenline), t - time
        self.passengerFlow = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))  # number of passengers moving from i to j by bus l, key: (i,j), l, t
        self.modsNum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.totalPassenger = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        self.alpha = defaultdict(dict)  # Binary variable: whether bus line l from station i in t, key: (i,j),t
        # Binary variable: whether stations i to station j on busline l, key: (i,j,l)

        for i, j, t, d in scenario.tripAttr:  # trip attribute (origin, destination, time of request, demand, cost)
            self.demand[i, j][t] = d
            self.depDemand[i][t] += d

        self.reEdges = []  # set of rebalancing edges,key(i,j)
        self.num_stations = len(self.station)  # total number of stations
        self.modRoute = []  # set of mod route edge,key(i,l)
        self.travelEdge = []  # set of travel edge, key(i,j,l)

        for l in self.busLine.keys():
            for (i, j) in self.busLine[l]:
                if i in self.selected_stations:
                    self.modRoute.append((i, l))

        for l in self.busLine.keys():
            for (i, j) in self.busLine[l]:
                self.travelEdge.append((i, j, l))

        for i in self.G:
            self.reEdges.append((i, i))
            for e in self.rebTime.keys():
                self.reEdges.append(e)
        self.reEdges = list(set(self.reEdges))  # 去重并转化为list形式
        self.num_edge = [len(self.G.out_edges(n)) + 1 for n in self.station]  # number of edges leaving each region

        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j]

        for i in self.selected_stations:
            for j in self.selected_stations:
                self.rebFlow[i, j] = defaultdict(float)

        self.lcost = defaultdict(dict)
        for l in self.busLine.keys():
            for (i, j) in self.busLine[l]:
                self.lcost[i][l] = self.line_time[i][l] * runcost  # get the cost from i to the destination of line l

        self.rebalance_cost = defaultdict(dict)
        for i, j in self.reEdges:
            self.rebalance_cost[i][j] = 99999

        for i, j in self.transTime:
            self.rebalance_cost[i][j] = self.transTime[(i, j)] * recost  # get the rebalance cost from i to j

        for (i, l) in self.modRoute:
            self.alpha[i, l] = defaultdict(int)  # key: time step

        self.start_nodes = {}
        self.end_nodes = {}
        for l, stops in self.ordered_stops.items():
            self.start_nodes[l] = stops[0]
            self.end_nodes[l] = stops[-1]

        for n in self.G:
            self.acc[n] = defaultdict(float)
            self.acc[n][0] = self.G.nodes[n]['accInit']  # number of mods within each station initially
            self.dacc[n] = defaultdict(float)

        self.recost = recost * scenario.tstep  # 计算一个时间步长的 rebalancing cost

        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)

        # 随机选择 20 个 (i, l) 键
        random.seed(12)
        selected_keys = random.sample([(i, l) for l in self.ordered_stops.keys() for i in self.ordered_stops[l]], 20)
        # 使用选定的键设置 totalPassenger[i][l][0] 和 modsNum[i][l][0] 的值
        for i, l in selected_keys:
            self.totalPassenger[i][l][0] = 10
            self.modsNum[i][l][0] = 3

        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        # observation: current mod distribution, time step, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def optimize(self, CPLEXPATH=None, PATH='', platform='win64', desiredAcc=None, desiredAdd=None):
        t = self.time  # time step for matching and rebalancing
        busLine = [int(l) for l in self.busLine]
        demandAttr = [(int(i), int(j), int(self.demand[i, j][t])) for i, j in self.demand \
                      if t in self.demand[i, j]]  # 当前时间阶段的需求量
        pathAttr = [(int(i), int(j), int(l)) for l in self.busLine.keys() for i, j in self.busLine[l]]
        accTuple = [(int(n), int(self.acc[n][t])) for n in self.acc]
        edgeAttr = [(int(i), int(j), int(self.transTime[(i, j)])) for (i, j) in self.transTime.keys()]
        accRL = [(int(i), int(round(desiredAcc[i]))) for i in self.selected_stations]
        ModsaddAttrRL = [(int(i), int(round(desiredAdd[i]))) for i in self.selected_stations]
        routeCost = [(int(i), int(l), self.lcost[i][l]) for i, l in self.modRoute]
        travelTime = [(int(i), int(j), int(l), self.ltime[(i, j)][l]) for l in self.busLine.keys() for i, j in self.busLine[l]]
        headWay = [(int(l), self.headWay[l]) for l in self.busLine.keys()]
        busType = [(int(i), int(l), int(self.modsNum[i][l][t])) for i, l in self.modRoute]
        totalPassengerAttr = [(int(i), int(l), int(self.totalPassenger[i][l][t])) for i, l in self.modRoute]
        ReCostAttr = [(int(i), int(j), self.rebalance_cost[i][j]) for (i, j) in self.transTime.keys()]
        modRouteAttr = [(int(i), int(l)) for (i, l) in self.modRoute]

        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        optimizePath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/optimize/' + PATH + '/'
        if not os.path.exists(optimizePath):
            os.makedirs(optimizePath)
        # create files for output
        datafile = optimizePath + 'data_{}.dat'.format(t)
        resfile = optimizePath + 'res_{}.dat'.format(t)
        out_file = optimizePath + 'out_{}.dat'.format(t)

        with open(datafile, 'w') as file:
            file.write('path="' + resfile + '";\r\n')
            file.write('busLine=' + mat2str(busLine) + ';\r\n')  # 已有的公交线
            file.write('demandAttr=' + mat2str(demandAttr) + ';\r\n')  # 该时刻从i到j的需求量
            file.write('pathAttr=' + mat2str(pathAttr) + ';\r\n')  # 以数组<i,j,l>的形式将每条公交线路l途经的站点输入
            file.write('acclintTuple=' + mat2str(accTuple) + ';\r\n')  # 数组<i,n>形式输入i站现有mod数量
            file.write('edgeAttr=' + mat2str(edgeAttr) + ';\r\n')  # 输入节点路段数据，包括rebalancing行驶时间
            file.write('accRL=' + mat2str(accRL) + ';\r\n')  # 输入RL计算得到t时刻i站需要mod数量
            file.write('routeCostAttr=' + mat2str(routeCost) + ';\r\n')  # 输入从各个起点i出发行驶线路l的花费,key(i,l)
            file.write('travelTimeAttr=' + mat2str(travelTime) + ';\r\n')  # 乘客乘坐l号公交从i到j的时间数据,key(i,j,l)
            file.write('busType=' + mat2str(busType) + ';\r\n')  # 输入t时刻抵达i站的l线路的mods车队属性，<i,l,s,t>
            file.write('desiredModsaddAttr=' + mat2str(ModsaddAttrRL) + ';\r\n')  # 输入RL计算得到的t时刻l在i站需要加减mods数量
            file.write('totalPassengerAttr=' + mat2str(totalPassengerAttr) + ';\r\n')  # 输入t时刻抵达i站时l线路车上的乘客总量
            file.write('headWayAttr=' + mat2str(headWay) + ';\r\n')  # 输入各个线路的发车时距
            file.write('ReCost=' + mat2str(ReCostAttr) + ';\r\n')  # 输入rebalancing的花费
            file.write('modRouteAttr=' + mat2str(modRouteAttr) + ';\r\n')

        modfile = modPath + 'MTNmodel.mod'
        if CPLEXPATH is None:
            CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio221/opl/bin/x86-64_linux/"
        my_env = os.environ.copy()
        if platform == 'mac':
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        # output of cplex
        with open(out_file, 'w') as output_f:  # 执行cplex优化模型
            subprocess.check_call([CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()

        vehicleflow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'vehicleflow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, l, f = v.split(',')
                        vehicleflow[float(i), float(l)] = float(f)
        modAction = {(i, l): vehicleflow[i, l] if (i, l) in vehicleflow else 0 for l in self.busLine for i, j in self.busLine[l] }

        rebalanceflow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'rebflow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        rebalanceflow[float(i), float(j)] = float(f)
        rebAction = {(i, j): rebalanceflow[i, j] for (i, j) in rebalanceflow}

        alpha = defaultdict(int)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'Alpha':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, l, f = v.split(',')
                        alpha[float(i), float(l)] = int(f)
        alphaAction = {(i, l): alpha[i, l] if (i, l) in alpha else 0 for l in self.busLine for i, j in self.busLine[l]}

        passengerflow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'passengerflow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, l, f = v.split(',')
                        passengerflow[float(i), float(j), float(l)] = float(f)
        passengerAction = {(i, j, l): passengerflow[i, j, l] if (i, j, l) in passengerflow else 0 for l in self.busLine for i, j in self.busLine[l]}

        return modAction, rebAction, alphaAction, passengerAction


    def pax_step(self, modAction, rebAction, alphaAction, passengerAction):
        t = self.time
        self.reward = 0

        # t时刻到达i站的l线路的mods的数量
        for l, stops in self.ordered_stops.items():
            for i in stops:
                if i in self.selected_stations:
                    if i == self.start_nodes[l]:
                        self.modsNum[i][l][t] = 0
                    else:
                        pos = stops.index(i)
                        prev_station = stops[pos - 1]
                        prev_ltime = self.ltime[(prev_station, i)][l]
                        if i == self.end_nodes[l]:
                            self.modsNum[i][l][t] = self.modsNum[prev_station][l][t - prev_ltime] + \
                                                    self.modFlow[prev_station][l][t - prev_ltime]
                        else:
                            next_station = stops[pos + 1]
                            next_ltime = self.ltime[(next_station, i)][l]
                            self.modsNum[i][l][t] = self.modsNum[prev_station][l][t - prev_ltime] + \
                                                    self.modsNum[next_station][l][t - next_ltime] + \
                                                    self.modFlow[prev_station][l][t - prev_ltime] + \
                                                    self.modFlow[next_station][l][t - next_ltime]

        # t时刻到达i站的l线路的乘客的数量
        for l, stops in self.ordered_stops.items():
            for i in stops:
                if i in self.selected_stations:
                    if i == self.start_nodes[l]:
                        self.totalPassenger[i][l][t] = 0
                    else:
                        add1 = 0
                        add2 = 0
                        arrived1 = 0
                        arrived2 = 0
                        pos = stops.index(i)
                        prev_station = stops[pos - 1]
                        prev_ltime = self.ltime[(prev_station, i)][l]
                        for j in range(pos, len(stops)):
                            for i_prime in range(0, pos - 1):
                                arrived1 += self.passengerFlow[stops[i_prime]][prev_station][l][
                                    t - self.ltime[(stops[i_prime], prev_station)][l]]
                            add1 += self.passengerFlow[prev_station][j][l][t - prev_ltime] - arrived1
                        if i != self.end_nodes[l]:
                            next_station = stops[pos + 1]
                            next_ltime = self.ltime[(next_station, i)][l]
                            for m in range(0, pos + 1):
                                for m_prime in range(pos + 2, len(stops)):
                                    arrived2 += self.passengerFlow[stops[m_prime]][next_station][l][
                                        t - self.ltime[(stops[m_prime], next_station)][l]]
                                add2 += self.passengerFlow[next_station][m][l][t - next_ltime] - arrived2
                        if i == self.end_nodes[l]:
                            self.totalPassenger[i][l][t] = self.totalPassenger[prev_station][l][t - prev_ltime] + add1
                        else:
                            next_station = stops[pos + 1]
                            next_ltime = self.ltime[(next_station, i)][l]
                            self.totalPassenger[i][l][t] = self.totalPassenger[prev_station][l][t - prev_ltime] + add1 + \
                                                           self.totalPassenger[next_station][l][t - next_ltime] + add2

        self.info['served_demand'] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        self.info["waitingtime_cost"] = 0

        self.modAction = modAction
        self.rebAction = rebAction
        self.alphaAction = alphaAction
        self.passengerAction = passengerAction

        for i in self.selected_stations:
            self.acc[i][t + 1] = self.acc[i][t]

        for (i, j, l) in self.passengerAction.keys():
                # I moved the min operator above, since we want modFlow to be consistent with modAction
                self.servedDemand[i, l][t] = self.passengerAction[(i, j, l)]
                self.modFlow[i][l][t] = self.modAction[(i, l)]
                self.podsFlow[i, j][t + self.ltime[i, j][l]] = self.modAction[(i, l)]
                self.passengerFlow[i][j][l][t] = self.passengerAction[(i, j, l)]
                self.alpha[i, l][t] = self.alphaAction[(i, l)]
                self.info["waitingtime_cost"] += self.tcost * self.passengerAction[(i, j, l)] * (0.5 * self.headWay[l])
                self.info['served_demand'] += self.servedDemand[i, l][t]
                self.reward += self.passengerAction[(i, j, l)] * 0.05    # 乘客一分钟给的价格0.05

        for i in self.selected_stations:
            for l in self.busLine.keys():
                if (i, l) in self.modAction:
                    # update the number of vehicles
                    self.acc[i][t + 1] -= self.modAction[(i, l)]
                    self.info["operating_cost"] += self.lcost[i][l] * self.modAction[(i, l)] * self.alphaAction[(i, l)]
                    self.reward -= (self.modAction[(i, l)] * self.lcost[i][l])

        for i in self.selected_stations:
            for j in self.selected_stations:
                if i == j:
                    continue
                self.rebFlow[i, j][t + self.transTime[i, j]] = self.rebAction[(i, j)]
                self.acc[i][t + 1] -= self.rebAction[(i, j)]
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
                self.dacc[j][t + self.transTime[i, j]] += self.rebFlow[i, j][t + self.transTime[i, j]]
                self.info["rebalancing_cost"] += self.rebAction[(i, j)] * self.rebalance_cost[i][j]
                self.reward -= (self.rebAction[(i, j)] * self.rebalance_cost[i][j])
                self.info["operating_cost"] += self.rebAction[(i, j)] * self.rebalance_cost[i][j]

        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        # for acc, the time index would be t+1, but for demand, the time index would be t
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j]
        done = (self.tf == t + 1)  # if the episode is completed

        return self.obs, max(0, self.reward), done, self.info

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)  # number of mods within each station, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of rebalancing mods arriving at each station, key: i - region, t - time
        self.rebFlow = defaultdict(dict)  # number of rebalancing mods, key: (i,j) - (origin, destination), t - time
        self.podsFlow = defaultdict(dict)  # number of mods
        self.modFlow = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # number of dispatching mods, key:(i,l) - (origin, chosenline), t - time
        self.passengerFlow = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))  # number of passengers moving from i to j by bus l, key: (i,j), l, t
        self.modsNum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.totalPassenger = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.alpha = defaultdict(dict)  # Binary variable: whether bus line l from station i in t, key: (i,j),t

        self.reEdges = []
        for i in self.G:
            self.reEdges.append((i, i))
            for e in self.rebTime.keys():
                self.reEdges.append(e)
        self.reEdges = list(set(self.reEdges))

        self.demand = defaultdict(dict)  # demand
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.stationDemand = defaultdict(dict)

        for i, j, t, d in tripAttr:  # trip attribute (origin, destination, busline, time of request, demand, cost)
            self.demand[i, j][t] = d
            if t not in self.stationDemand[i]:
                self.stationDemand[i][t] = 0
            else:
                self.stationDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)

        for i in self.selected_stations:
            for j in self.selected_stations:
                self.rebFlow[i, j] = defaultdict(float)

        for (i, l) in self.modRoute:
            self.alpha[i, l] = defaultdict(int)  # key: time step

        for n in self.G:
            self.acc[n] = defaultdict(float)
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)

        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)

        # 随机选择 20 个 (i, l) 键
        random.seed(12)
        selected_keys = random.sample([(i, l) for l in self.ordered_stops.keys() for i in self.ordered_stops[l]], 20)
        # 使用选定的键设置 totalPassenger[i][l][0] 和 modsNum[i][l][0] 的值
        for i, l in selected_keys:
            self.totalPassenger[i][l][0] = 10
            self.modsNum[i][l][0] = 3

        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs


class Scenario:

    def __init__(self, tf=60, sd=None, ninit=5, unit_travel_time=1, tstep=1):
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            random.seed(self.sd)

        self.tstep = tstep
        self.unit_travel_time = unit_travel_time
        self.demand_input = defaultdict(dict)
        # 读取 travel_time.csv 中的数据，构建网络图
        self.G = nx.Graph()
        self.busLine = {}
        self.ltime = defaultdict(dict)

        # 读取 travel_time.csv 中的数据，构建网络图
        travel_time = pd.read_csv("travel_time_test.csv")
        travel_time.fillna(0, inplace=True)

        # 遍历 travel_time 数据集的每一行，添加到图中
        for i, row in travel_time.iterrows():
            self.G.add_edge(row["Haltestellen_Id"], row["Haltestellen_Id_2"], line=row["Line_id"])
            line = row["Line_id"]
            start_node, end_node = row["Haltestellen_Id"], row["Haltestellen_Id_2"]
            if line not in self.busLine:
                self.busLine[line] = []
            self.busLine[line].append((start_node, end_node))
            travel_time_l = travel_time[
                (travel_time["Haltestellen_Id"] == start_node) & (travel_time["Haltestellen_Id_2"] == end_node) & (
                        travel_time["Line_id"] == line)]
            if not travel_time_l.empty:
                travel_time_val = travel_time_l.iloc[0]["travel_time"]
                self.ltime[(start_node, end_node)][line] = int(round(travel_time_val / tstep))
                self.ltime[(start_node, start_node)][line] = 0
                # get the time spend from i to the j of line l
        for n in self.G.nodes():
            self.G.nodes[n]['accInit'] = 0  # 初始值为 0

        self.G = self.G.to_directed()

        self.travel_time_single = pd.read_csv("travel_time_single_test.csv")
        self.travel_time_single.fillna(0, inplace=True)
        self.busLine_single = {}
        for i, row in self.travel_time_single.iterrows():
            line = row["Line_id"]
            start_node, end_node = row["Haltestellen_Id"], row["Haltestellen_Id_2"]
            if line not in self.busLine_single:
                self.busLine_single[line] = []
            self.busLine_single[line].append((start_node, end_node))

        self.ordered_stops = defaultdict(list)

        for line, stops in self.busLine_single.items():
            for i, j in stops:
                if i not in self.ordered_stops[line]:
                    self.ordered_stops[line].append(i)
                if j not in self.ordered_stops[line]:
                    self.ordered_stops[line].append(j)

        # 读取文件
        stations_df = pd.read_csv('selected_stations.csv')
        transTime_df = pd.read_csv('transTime.csv')

        # 将数据存入selected_stations
        self.selected_stations = list(stations_df['Station ID'])

        # 将数据存入transTime
        self.transTime = defaultdict(dict)
        for i, row in transTime_df.iterrows():
            self.transTime[(row['from_station_id'], row['to_station_id'])] = row['travel_time']

        for n in self.selected_stations:
            self.G.nodes[n]['accInit'] = int(ninit)

        self.headWay = defaultdict(lambda: float('inf'))
        for l in self.busLine.keys():
            self.headWay[l] = int(10000)
        timetable_df = pd.read_csv("bustimeTable_test.csv", delimiter=",")
        busLineSet = set(self.busLine.keys())
        for i, row in timetable_df.iterrows():
            line_id = row["line"]
            interval = row["interval"]
            if line_id in busLineSet:
                self.headWay[line_id] = int(interval)

        self.line_time = defaultdict(dict)
        line_time_df = pd.read_csv("(i,line)time_modified_test.csv", delimiter=",")
        for _, row in line_time_df.iterrows():
            i, l, travel_time = row["Haltestellen_Id"], row["Line_id"], row["travel_time"]
            self.line_time[i][l] = int(round(travel_time / tstep))

        self.c = defaultdict(dict)
        self.tf = tf
        self.reEdges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        self.station = list(self.G)

        for i in self.station:
            for j in self.station:
                self.demand_input[i, j] = defaultdict(float)

        self.rebTime = defaultdict()
        for i, j in self.ltime.keys():
            travel_times = [self.ltime[(i, j)][l] for l in self.ltime[(i, j)]]
            if travel_times:
                self.rebTime[(i, j)] = round(min(travel_times))

        demand_df = pd.read_csv("demand.csv")
        # 遍历每一行数据，并将需求量取整后存储到 self.demand_input[(o, d)][t] 中
        for _, row in demand_df.iterrows():
            o, d, t, demand = row["origin"], row["destination"], row["time"], row["demand"]
            self.demand_input[(o, d)][t] = demand
            # why tf * 2：给还没计算的时间里的需求赋值

        self.tripAttr = self.get_random_demand()


    def get_random_demand(self, reset=False):
        # generate demand
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        # assuming static demand is already generated
        # reset = False means that the function is called when initializing the dem
        tripAttr = []
        # skip this when resetting the demand
        random.seed(self.sd)

        self.demand = defaultdict(dict)
        for i in self.station:
            for j in self.station:
                self.demand[i, j] = defaultdict(float)

        study_time_range = range(0, self.tf * 2)  # 假设研究时间范围是0到120分钟
        for i, j in self.reEdges:
            for t in study_time_range:
                self.demand[(i, j)][t] = int(random.uniform(0, 3))
                if random.random() < 0.2:
                    # 有 10% 的概率将需求量增加到 6
                    self.demand[(i, j)][t] += 3
                self.demand[(i, j)][t] = min(self.demand[(i, j)][t], 6)  # 最大需求量为 6
                tripAttr.append((i, j, t, self.demand[i, j][t]))

        return tripAttr