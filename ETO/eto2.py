import time
import random
import threading
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import defaultdict, deque
import os
import gc
import heapq
from itertools import combinations
import concurrent.futures
import hashlib
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
from collections import defaultdict


def evaluate_metrics_ep(component, workers, worker_names, assignments, worker_bind='mod'):
    """
    针对 EP 聚合的统一评估器：
    - PS Agg. Vol: 直达 PS 的 (t,e) 段 + 每个 (s,t) 聚成 1 段
    - In-net Agg. Vol: Σ_{(s,t)} (A_s(t)-1) * SEG
    - Total Comm. Vol: worker→(s|PS) + Σ_{(s,t)} 1·(s→PS)
    - Fallback: token 只要有任何 expert 输出直达 PS
    """
    SEG = int(component.moe_config.expert_output_size_mb * 1024 * 1024)

    # (1) worker -> (switch|PS)
    A = defaultdict(int)   # (s,t) -> expert count
    direct_ps_cnt = 0
    total_comm = 0
    activated_pairs = 0

    for e in component.experts:
        expert_id = int(e[1:]) if isinstance(e, str) else e
        w = worker_names[expert_id % len(worker_names)]
        for t in component.tokens:
            if component.token_routing.get((t, expert_id), {}).get('activated', 0) == 0:
                continue
            activated_pairs += 1
            node = assignments.get(t, {}).get(e, component.ps)
            if node == component.ps:
                total_comm += SEG * component.calculate_distance(w, component.ps)
                direct_ps_cnt += 1
            else:
                total_comm += SEG * component.calculate_distance(w, node)
                A[(node, t)] += 1

    # (2) switch -> PS
    ps_from_switch_bytes = 0
    in_net_saved_bytes = 0
    for (sw, t), n in A.items():
        ps_from_switch_bytes += SEG            # 每个 (s,t) 聚合成 1 段
        in_net_saved_bytes += (n - 1) * SEG    # 节省量
        total_comm += SEG * component.calculate_distance(sw, component.ps)

    ps_agg_bytes = direct_ps_cnt * SEG + ps_from_switch_bytes

    # (3) Fallback
    activated_tokens = 0
    # 旧口径：任意落 PS 记 fallback（保留，兼容下游）
    fallback_tokens_any = 0
    # 宽松口径：仅当全部落 PS 才记 fallback（新增）
    fallback_tokens_loose = 0
    # 完全在网内：所有激活 (t,e) 都在交换机
    full_innet_tokens = 0
    for t in component.tokens:
        token_activated = False
        any_to_ps = False
        any_to_switch = False

        for e in component.experts:
            expert_id = int(e[1:]) if isinstance(e, str) else e
            if component.token_routing.get((t, expert_id), {}).get('activated', 0) == 0:
                continue
            token_activated = True
            if assignments.get(t, {}).get(e, component.ps) == component.ps:
                any_to_ps = True
            else:
                any_to_switch = True

        if token_activated:
            activated_tokens += 1
            # 旧口径：有任意一条到 PS
            if any_to_ps:
                fallback_tokens_any += 1
            # 宽松口径：全部到 PS 才算 fallback
            if (not any_to_switch) and any_to_ps:
                fallback_tokens_loose += 1
            # 完全在网内：没有任何一条到 PS
            if (not any_to_ps) and any_to_switch:
                full_innet_tokens += 1

    fallback_rate = (fallback_tokens_any / activated_tokens) if activated_tokens > 0 else 0.0
    fallback_rate_loose = (fallback_tokens_loose / activated_tokens) if activated_tokens > 0 else 0.0
    full_innet_rate = (full_innet_tokens / activated_tokens) if activated_tokens > 0 else 0.0

    return {
        'agg_overhead': ps_agg_bytes,
        'in_network_agg': in_net_saved_bytes,
        'comm_overhead': total_comm,
        'fallback_rate': fallback_rate_loose
    }

# =============================================================
# ⚠️ 修改说明
# 1. 全部注释改为中文。
# 2. 在 run_simulation() 中按四组参数依次运行：
#    - num_experts: 32 → 64 → 128 → 256
#    - top_k      : 4  → 4  → 6   → 8
# 3. 新增指标 (6) In-network aggregation fallback rate：
#    - 定义：在窗口内未能在网内完全融合、最终在 PS 完成的 token 占比。
#    - 计算方式：基于每个 token 的完成情况（是否所有被激活的 (token, expert)
#      都在交换机上聚合；若其中任意一个落在 PS，则该 token 视为 fallback）。
#    - 在各 evaluate_* 方法中计算并返回 'fallback_rate'。
# =============================================================

# =============================================================
# 轻量级网络拓扑原语（不改动任何逻辑，只补充中文注释）
# =============================================================
class LightweightTopo:
    """一个基于 NetworkX 的最小拓扑抽象。"""

    def __init__(self):
        self._nodes = []  # 所有节点名
        self._switches = []  # 交换机列表
        self._hosts = []  # 主机列表
        self._links = []  # 链路列表 (node1, node2)
        self._node_info = {}  # 每个节点的元信息
        self.graph = nx.Graph()  # 底层无向图

    def addSwitch(self, name):
        """添加交换机，默认内存 64MB；返回交换机名。"""
        self._switches.append(name)
        self._nodes.append(name)
        self._node_info[name] = {'type': 'switch', 'memory': 64 * 1024 * 1024}
        self.graph.add_node(name, type='switch')
        return name

    def addHost(self, name, ip=None):
        """添加主机，可选 IP；返回主机名。"""
        self._hosts.append(name)
        self._nodes.append(name)
        self._node_info[name] = {'type': 'host', 'ip': ip}
        self.graph.add_node(name, type='host')
        return name

    def addLink(self, node1, node2):
        """在图中添加无向链路。"""
        self._links.append((node1, node2))
        self.graph.add_edge(node1, node2)

    # 访问器（逻辑原样保留）
    def nodes(self):
        return self._nodes

    def switches(self):
        return self._switches

    def hosts(self):
        return self._hosts

    def links(self):
        return self._links

    def nodeInfo(self, node):
        return self._node_info.get(node, {})


# =============================================================
# 拓扑：轻量级 Leaf-Spine
# =============================================================
class LeafSpineTopoLight(LightweightTopo):
    """构建 10×10 的 leaf-spine 结构，每个 leaf 连接 5 台主机。"""

    def __init__(self):
        super().__init__()
        spines = [self.addSwitch(f'spine{i}') for i in range(1, 11)]
        leaves = []

        for i in range(1, 11):
            leaf = self.addSwitch(f'leaf{i}')
            leaves.append(leaf)

            # spine 与该 leaf 全连接
            for spine in spines:
                self.addLink(spine, leaf)

            # leaf 下挂 10 台主机
            for j in range(1, 11):
                host = self.addHost(f'h{i}_{j}')
                self.addLink(host, leaf)

        # 参数服务器连接到第一个 spine
        ps = self.addHost('ps', ip='10.0.0.254')
        # self.addLink(ps, spines[0])  # PS连到第一个 spine
        for spine in spines:
            self.addLink(ps, spine)

# =============================================================
# 拓扑：轻量级 Fat-Tree
# =============================================================
class FatTreeTopoLight(LightweightTopo):
    """构建 k 叉 Fat-Tree（默认 k=8）。逻辑保持不变。"""

    def __init__(self, k=8):
        super().__init__()
        core_sw = [self.addSwitch(f'c{i}') for i in range(1, (k // 2) ** 2 + 1)]

        for p in range(k):
            agg_sw = [self.addSwitch(f'p{p}_a{i}') for i in range(1, k // 2 + 1)]
            edge_sw = [self.addSwitch(f'p{p}_e{j}') for j in range(1, k // 2 + 1)]

            for a in agg_sw:
                for e in edge_sw:
                    self.addLink(a, e)

            for i, c in enumerate(core_sw):
                idx = i // (k // 2)
                self.addLink(agg_sw[idx], c)

            for e in edge_sw:
                for h in range(1, k // 2 + 1):
                    edge_id = e.split('_')[-1][1:]
                    host = self.addHost(f'h{p}_{edge_id}_{h}')
                    self.addLink(host, e)

        ps = self.addHost('ps', ip='10.0.0.254')
        # self.addLink(ps, core_sw[0])
        for core in core_sw:
            self.addLink(ps, core)


# =============================================================
# MoE 配置（参数位置说明）
# =============================================================
class MoEConfig:
    """MoE 仿真配置包。"""

    def __init__(self, num_experts, top_k, num_tokens,
                 jitter_std: float = 0.0):  # ← 新增：抖动标准差（秒）
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_tokens = num_tokens
        self.expert_output_size_mb = 0.016  # 参数可调：专家输出单段大小（MB）
        self.switch_memory_mb = 64  # 参数可调：交换机内存（MB）
        self.average_sending_rate = 10 * 1024 * 1024 * 1024
        # === 新增参数（可实验调节）：到达/扰动抖动的标准差（单位：秒） ===
        # 说明：0.0（无抖动）、0.2、0.4、0.6 等
        self.jitter_std = jitter_std

    def generate_token_routing(self, tokens):
        """为每个 token 随机选择 top-k 专家，并生成门控权重（与原逻辑一致）。"""
        routing = {}
        for token in tokens:
            token_id = int(token[1:]) if isinstance(token, str) else token
            random.seed(token_id * 1000)
            activated_experts = random.sample(range(self.num_experts), self.top_k)
            gate_weights = np.random.dirichlet(np.ones(self.top_k))

            for expert_id in range(self.num_experts):
                if expert_id in activated_experts:
                    idx = activated_experts.index(expert_id)
                    routing[(token, expert_id)] = {
                        'activated': 1,
                        'gate_weight': gate_weights[idx]
                    }
                else:
                    routing[(token, expert_id)] = {
                        'activated': 0,
                        'gate_weight': 0.0
                    }
        return routing


# =============================================================
# 流量监控（计数器）
# =============================================================
class TrafficMonitor:
    """轻量级的传输计量（用于累计统计）。"""

    def __init__(self):
        self.worker_to_switch = defaultdict(float)
        self.worker_to_ps = defaultdict(float)
        self.switch_to_ps = defaultdict(float)
        self.link_utilization = defaultdict(list)
        self.timestamp = []
        self.time_series = defaultdict(list)
        self.lock = threading.Lock()

    def record_transmission(self, src, dst, size_mb, timestamp):
        """线程安全记录一次传输事件。"""
        with self.lock:
            if dst == 'ps':
                self.worker_to_ps[src] += size_mb
            else:
                self.worker_to_switch[(src, dst)] += size_mb

            self.time_series['timestamp'].append(timestamp)
            self.time_series[f'{src}_to_{dst}'].append(size_mb)

    def calculate_metrics(self):
        """聚合总量（与原逻辑一致）。"""
        ps_overhead = sum(self.worker_to_ps.values()) + sum(self.switch_to_ps.values())
        in_network = sum(self.worker_to_switch.values())
        total_comm = ps_overhead + in_network
        return ps_overhead, in_network, total_comm


# =============================================================
# 基线方案：仅 PS 聚合（PS-Only）
# =============================================================
class PS_Only_MoE:
    """所有专家输出直接发往参数服务器（PS）。"""

    def __init__(self, topo, experts, moe_config=None, traffic_monitor=None):
        self.topo = topo
        self.moe_config = moe_config or MoEConfig()
        self.traffic_monitor = traffic_monitor
        self.ps = 'ps'
        self.graph = self.topo.graph

        self.num_experts = self.moe_config.num_experts
        self.top_k = self.moe_config.top_k
        self.experts = experts
        self.tokens = []
        self.token_routing = {}

    def calculate_distance(self, node1, node2):
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def run_ps_only(self, workers, worker_names, experts, tokens):
        self.tokens = tokens
        self.token_routing = self.moe_config.generate_token_routing(tokens)
        return self.evaluate_ps_only_solution(workers, worker_names, experts, tokens)

    def evaluate_ps_only_solution(self, workers, worker_names, experts, tokens):
        """计算仅 PS 聚合的开销，并计算 fallback 率（恒为 1）。"""
        SEGMENT_SIZE = self.moe_config.expert_output_size_mb * 1024 * 1024
        total_ps_traffic = 0
        total_in_network_agg = 0
        total_communication_cost = 0

        # 统计 token 是否 fallback（PS-only 场景全部 fallback）
        activated_token_count = 0
        fallback_token_count = 0

        for token in tokens:
            token_activated = False
            for expert_id in range(self.num_experts):
                routing_info = self.token_routing.get((token, expert_id), {})
                if routing_info.get('activated', 0) == 0:
                    continue
                token_activated = True

                worker_name = worker_names[expert_id % len(worker_names)]
                dist_to_ps = self.calculate_distance(worker_name, self.ps)
                total_ps_traffic += SEGMENT_SIZE
                total_communication_cost += SEGMENT_SIZE * dist_to_ps

            if token_activated:
                activated_token_count += 1
                fallback_token_count += 1  # 全部在 PS 完成

        agg_overhead = total_ps_traffic
        in_network_agg = 0
        comm_overhead = total_communication_cost

        fallback_rate = (fallback_token_count / activated_token_count) if activated_token_count > 0 else 0.0

        print(f"\nok")
        print(f"PS aggregation overhead: {agg_overhead / (1024 * 1024):.2f} MB")
        print(f"In-network aggregation: {in_network_agg / (1024 * 1024):.2f} MB (always 0 for PS-only)")
        print(f"Communication overhead: {comm_overhead:.2f}")
        print(f"Fallback rate: {fallback_rate:.3f}")

        return {
            'agg_overhead': agg_overhead,
            'in_network_agg': in_network_agg,
            'comm_overhead': comm_overhead,
            'fallback_rate': fallback_rate
        }


# =============================================================
# ETO：基于线性规划的专家-Token 下沉到交换机
# =============================================================
class ETO_MoE:
    """线性规划 + 随机舍入进行聚合位置选择。"""

    def __init__(self, topo, experts, moe_config=None, traffic_monitor=None):
        self.topo = topo
        self.moe_config = moe_config or MoEConfig()
        self.traffic_monitor = traffic_monitor
        self.prog_sw = self._get_programmable_switches()
        self.ps = 'ps'
        self.graph = self.topo.graph

        self.num_experts = self.moe_config.num_experts
        self.top_k = self.moe_config.top_k
        self.experts = experts
        self.tokens = []
        self.token_routing = {}

        self.switch_memory_mb = self.moe_config.switch_memory_mb
        self.token_assignments = defaultdict(lambda: defaultdict(str))

    # def _get_programmable_switches(self):
    #     all_switches = self.topo.switches()
    #     num_prog = max(3, int(0.2 * len(all_switches)))
    #     prog_sw = random.sample(all_switches, min(num_prog, len(all_switches)))
    #     for sw in prog_sw:
    #         self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024
    #     return prog_sw

    def _get_programmable_switches(self):
        """版本3：简化版本，只设置内存不计算容量"""
        all_switches = self.topo.switches()
        tors = [s for s in all_switches if s.startswith('leaf') or ('_e' in s)]
        other_sw = [s for s in all_switches if s not in tors]

        # 选择80%的TOR交换机和20%的其他交换机
        num_tors = max(1, int(0.3 * len(tors))) if tors else 0
        num_other = max(1, int(0.6 * len(other_sw))) if other_sw else 0

        prog_tors = random.sample(tors, min(num_tors, len(tors))) if tors else []
        prog_other = random.sample(other_sw, min(num_other, len(other_sw))) if other_sw else []
        prog = prog_tors + prog_other

        for sw in prog:
            self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024

        return prog

    def calculate_distance(self, node1, node2):
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def solve_lp_eto(self, workers, tokens, experts):
        prob = pulp.LpProblem("ETO_FOCUS", pulp.LpMaximize)
        all_nodes = self.prog_sw + [self.ps]

        # 变量
        x = pulp.LpVariable.dicts(
            "x",
            [(t, s) for t in tokens for s in all_nodes],
            lowBound=0, upBound=1, cat='Continuous'
        )

        y = {}
        for e in experts:
            for t in tokens:
                expert_id = int(e[1:]) if isinstance(e, str) else e
                if self.token_routing.get((t, expert_id), {}).get('activated', 0) == 1:
                    for s in all_nodes:
                        y[(e, t, s)] = pulp.LpVariable(
                            f"y_{e}_{t}_{s}", lowBound=0, upBound=1, cat='Continuous'
                        )

        m = self.moe_config.expert_output_size_mb * 1024 * 1024

        # 目标
        objective_terms = []
        for t in tokens:
            for s in self.prog_sw:
                expert_sum_terms = []
                for e in experts:
                    expert_id = int(e[1:]) if isinstance(e, str) else e
                    if self.token_routing.get((t, expert_id), {}).get('activated', 0) == 1:
                        if (e, t, s) in y:
                            expert_sum_terms.append(y[(e, t, s)])
                if expert_sum_terms:
                    expert_sum = pulp.lpSum(expert_sum_terms)
                    distance = self.calculate_distance(s, self.ps)
                    objective_terms.append((expert_sum - x[(t, s)]) * distance * m)
        prob += pulp.lpSum(objective_terms)

        # 约束
        all_nodes = self.prog_sw + [self.ps]
        for t in tokens:
            prob += pulp.lpSum(x[(t, s)] for s in all_nodes) >= 1

        for e in experts:
            for t in tokens:
                expert_id = int(e[1:]) if isinstance(e, str) else e
                r_te = self.token_routing.get((t, expert_id), {}).get('activated', 0)
                if r_te == 1:
                    y_vars = [y[(e, t, s)] for s in all_nodes if (e, t, s) in y]
                    if y_vars:
                        prob += pulp.lpSum(y_vars) == r_te

        for e in experts:
            for t in tokens:
                for s in all_nodes:
                    if (e, t, s) in y:
                        prob += y[(e, t, s)] <= x[(t, s)]

        for s in self.prog_sw:
            capacity = self.topo.nodeInfo(s).get('memory', self.switch_memory_mb * 1024 * 1024)
            prob += pulp.lpSum(x[(t, s)] * m for t in tokens) <= capacity

        prob.solve(pulp.PULP_CBC_CMD(
            msg=0,
            threads=2,
            options=[
                "presolve on",
                "strong 0",
                "cuts off",
                "primalS steep",
                "dualS steep",
                "node 1"
            ],
            timeLimit=600,
            gapRel=0.01
        ))

        status = pulp.LpStatus[prob.status]
        print(f"状态: {status}")

        y_full = pulp.LpVariable.dicts(
            "y",
            [(e, t, s) for e in experts for t in tokens for s in all_nodes],
            lowBound=0, upBound=0
        )
        for key, var in y.items():
            y_full[key] = var
        for e in experts:
            for t in tokens:
                for s in all_nodes:
                    if (e, t, s) not in y:
                        dummy_var = pulp.LpVariable(f"dummy_{e}_{t}_{s}", lowBound=0, upBound=0)
                        dummy_var.varValue = 0
                        y_full[(e, t, s)] = dummy_var
        return x, y_full, status

    def partition_knapsacks(self, items, k):
        if k <= 0 or not items:
            return []
        knapsacks = [[] for _ in range(k)]
        sums = [0] * k
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        for item in sorted_items:
            min_idx = sums.index(min(sums))
            knapsacks[min_idx].append(item)
            sums[min_idx] += item[1]
        return knapsacks

    def balanced_partition(self, items, k):
        """
        将items分成k个knapsack，使得各个knapsack的和尽可能平衡
        items: [(switch, value)] 列表
        k: 分区数量
        """
        if k <= 0 or not items:
            return [[] for _ in range(k)]

        # 按值降序排序（重要：先处理大值）
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # 初始化k个空knapsack和它们的当前和
        knapsacks = [[] for _ in range(k)]
        sums = [0.0] * k

        # 贪心算法：总是将当前item分配给当前和最小的knapsack
        for item in sorted_items:
            switch, value = item
            min_idx = np.argmin(sums)  # 找到当前和最小的knapsack
            knapsacks[min_idx].append(item)
            sums[min_idx] += value

        return knapsacks

    def knapsack_randomized_rounding(self, x, y, workers, tokens, experts):
        m = self.moe_config.expert_output_size_mb * 1024 * 1024

        # token_switches = {}
        # residual = {}
        # for s in self.prog_sw:
        #     capacity = self.topo.nodeInfo(s).get('memory', self.switch_memory_mb * 1024 * 1024)
        #     residual[s] = capacity // m
        #
        # for t in tokens:
        #     k_t = int(sum(x[(t, s)].varValue for s in self.prog_sw if x[(t, s)].varValue))
        #     items = [(s, x[(t, s)].varValue) for s in self.prog_sw if x[(t, s)].varValue and x[(t, s)].varValue > 0]
        #     if not items:
        #         continue
        #     knapsacks = self.partition_knapsacks(items, k_t)
        #     selected = set()
        #     for knapsack in knapsacks:
        #         if not knapsack:
        #             continue
        #         total_sum = sum(item[1] for item in knapsack)
        #         if total_sum == 0:
        #             continue
        #         probabilities = [item[1] / total_sum for item in knapsack]
        #         switches = [item[0] for item in knapsack]
        #         chosen_switch = np.random.choice(switches, p=probabilities)
        #         selected.add(chosen_switch)
        #         residual[chosen_switch] -= 1
        #     token_switches[t] = selected
        # 初始化剩余容量
        residual = {}
        for s in self.prog_sw:
            capacity = self.topo.nodeInfo(s).get('memory', self.switch_memory_mb * 1024 * 1024)
            residual[s] = capacity // m

        token_switches = {}

        for t in tokens:
            # 计算 k(t) = ceil(sum_{s} x_t^s)
            x_sum = sum(x[(t, s)].varValue for s in self.prog_sw if x[(t, s)].varValue > 1e-5)
            k_t = max(1, int(np.ceil(x_sum)))  # 至少为1

            # 只考虑有正值和剩余容量的交换机
            items = [(s, x[(t, s)].varValue) for s in self.prog_sw
                     if x[(t, s)].varValue > 1e-5 and residual[s] > 0]

            if not items:
                token_switches[t] = set()
                continue

            # 使用平衡分区算法
            knapsacks = self.balanced_partition(items, k_t)
            selected = set()

            for knapsack in knapsacks:
                if not knapsack:
                    continue

                # 过滤出有剩余容量的交换机
                available_switches = [item for item in knapsack if residual[item[0]] > 0]

                if not available_switches:
                    # 所有容量耗尽，这个knapsack无法选择任何交换机
                    continue

                total_sum = sum(item[1] for item in available_switches)
                probabilities = [item[1] / total_sum for item in available_switches]
                switches = [item[0] for item in available_switches]

                chosen_switch = np.random.choice(switches, p=probabilities)
                selected.add(chosen_switch)
                residual[chosen_switch] -= 1

            token_switches[t] = selected

        token_assignments = defaultdict(lambda: defaultdict(str))
        for e in experts:
            for t in tokens:
                if self.token_routing.get((t, int(e[1:]) if isinstance(e, str) else e), {}).get('activated', 0) == 0:
                    continue
                available_switches = token_switches.get(t, set())
                if not available_switches:
                    token_assignments[t][e] = self.ps
                    continue
                switch_probs = {}
                total_pn = 0.0
                for s in available_switches:
                    x_gs = x[(t, s)].varValue if x[(t, s)].varValue else 0
                    if x_gs > 1e-5:
                        y_wgs = y[(e, t, s)].varValue if y[(e, t, s)].varValue else 0
                        pn_s = y_wgs / x_gs if x_gs > 0 else 0
                    else:
                        pn_s = 0.0
                    switch_probs[s] = pn_s
                    total_pn += pn_s
                ps_prob = max(0, 1.0 - total_pn)
                switch_probs[self.ps] = ps_prob
                choices = list(switch_probs.keys())
                probabilities = [switch_probs[s] for s in choices]
                if sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    chosen_node = np.random.choice(choices, p=probabilities)
                else:
                    chosen_node = self.ps
                token_assignments[t][e] = chosen_node
        return token_assignments

    def run_eto(self, workers, worker_names, experts, tokens):
        self.tokens = tokens
        self.token_routing = self.moe_config.generate_token_routing(tokens)
        x, y, status = self.solve_lp_eto(workers, tokens, experts)
        if status != 'Optimal':
            print(f"LP solving failed: {status}")
            assignments = {t: {e: self.ps for e in experts} for t in tokens}
            return self.evaluate_eto_solution(workers, experts, tokens, assignments)
        assignments = self.knapsack_randomized_rounding(x, y, workers, tokens, experts)
        return self.evaluate_eto_solution(workers, experts, tokens, assignments)

    def evaluate_eto_solution(self, workers, experts, tokens, assignments):
        # 保持现有签名不变，内部转到统一评估器
        self.tokens = tokens
        self.experts = experts
        return evaluate_metrics_ep(self, workers, list(workers.keys()), assignments, worker_bind='mod')


# =============================================================
# DSA：基于事件的优先级聚合与抢占
# =============================================================
class DSA_MoE:
    """ESA/DSA：在首个可编程交换机处，若有槽则接纳累加；若满则按优先级抢占最低槽（受害者部分和以 1 段上送 PS）；否则在该交换机转发到 PS。满 top_k 就地聚合并释放槽。"""

    def __init__(self, topo, experts, moe_config=None, traffic_monitor=None):
        self.topo = topo
        self.moe_config = moe_config or MoEConfig()
        self.traffic_monitor = traffic_monitor
        self.ps = 'ps'
        self.graph = self.topo.graph

        self.num_experts = self.moe_config.num_experts
        self.top_k = self.moe_config.top_k
        self.experts = experts
        self.tokens = []
        self.token_routing = {}

        self.priority_bits = 8
        self.timeout = 4.0

        # —— 先建所有容量/数据结构（_get_programmable_switches 里会用到）——
        self.SEGMENT_SIZE = int(self.moe_config.expert_output_size_mb * 1024 * 1024)
        self.switch_capacity = {}                                   # sw -> 槽位上限
        self.switch_used_slots = defaultdict(int)                   # sw -> 已用槽位
        self.switch_token_counts = defaultdict(lambda: defaultdict(int))  # sw -> token -> 已累加片段数
        self.switch_heap = {}                                       # sw -> [(priority, token), ...] 最小堆
        self.switch_slot_prio = defaultdict(dict)                   # sw -> token -> priority
        self._spath_cache = {}
        self.token_assignments = defaultdict(lambda: defaultdict(str))
        self._evicted_tokens = defaultdict(set)                     # sw -> {token}：统计被逐出的 token（用于 switch->PS 计费）

        # —— 最后再探测可编程交换机（初始化容量 & 堆）——
        self.prog_sw = self._get_programmable_switches()

    # def _get_programmable_switches(self):
    #     all_sw = self.topo.switches()
    #     tors = [s for s in all_sw if s.startswith('leaf') or ('_e' in s)]
    #     prog = tors if tors else random.sample(all_sw, max(2, int(0.2 * len(all_sw))))
    #     for sw in prog:
    #         self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024
    #         memB = self.topo.nodeInfo(sw).get('memory', self.moe_config.switch_memory_mb * 1024 * 1024)
    #         self.switch_capacity[sw] = max(1, int(memB // self.SEGMENT_SIZE))
    #     self.switch_heap = {sw: [] for sw in prog}
    #     return prog
    def _get_programmable_switches(self):
        """版本2：在版本1基础上增加switch_heap初始化"""
        all_sw = self.topo.switches()
        tors = [s for s in all_sw if s.startswith('leaf') or ('_e' in s)]
        other_sw = [s for s in all_sw if s not in tors]

        # 选择80%的TOR交换机和20%的其他交换机
        num_tors = max(1, int(0.3 * len(tors))) if tors else 0
        num_other = max(1, int(0.6 * len(other_sw))) if other_sw else 0

        prog_tors = random.sample(tors, min(num_tors, len(tors))) if tors else []
        prog_other = random.sample(other_sw, min(num_other, len(other_sw))) if other_sw else []
        prog = prog_tors + prog_other

        for sw in prog:
            self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024
            memB = self.topo.nodeInfo(sw).get('memory', self.moe_config.switch_memory_mb * 1024 * 1024)
            self.switch_capacity[sw] = max(1, int(memB // self.SEGMENT_SIZE))

        # 额外初始化switch_heap
        self.switch_heap = {sw: [] for sw in prog}
        return prog

    def calculate_distance(self, node1, node2):
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def _shortest_path(self, u, v):
        key = (u, v)
        if key in self._spath_cache: return self._spath_cache[key]
        try:
            p = nx.shortest_path(self.graph, u, v)
        except nx.NetworkXNoPath:
            p = None
        self._spath_cache[key] = p
        return p

    def _first_prog_on_path(self, worker):
        p = self._shortest_path(worker, self.ps)
        if not p or len(p) <= 2: return None
        for node in p[1:-1]:
            if node in self.prog_sw:
                return node
        return None

    def calculate_priority_dsa(self, token, expert_id, layer_id=None, remaining_time=None, comm_comp_ratio=None):
        token_id = int(token[1:]) if isinstance(token, str) else token
        if layer_id is None:
            experts_per_layer = 32
            layer_id = (expert_id // experts_per_layer) + 1
        if remaining_time is None:
            total_tokens = len(self.tokens) if self.tokens else 512
            progress_ratio = token_id / total_tokens
            remaining_time = max(10, 100 * (1 - progress_ratio))
        if comm_comp_ratio is None:
            if expert_id < 134: comm_comp_ratio = 2.0
            elif expert_id < 192: comm_comp_ratio = 1.0
            else: comm_comp_ratio = 0.5
        num_layers = 2
        priority = (1.0 / remaining_time) * (num_layers / layer_id) * comm_comp_ratio
        return int(min(max(priority * 100, 1), 255))

    def _evict_min_if_needed(self, sw, new_token, new_prio):
        cap = self.switch_capacity[sw]
        if self.switch_used_slots[sw] < cap:
            return True
        heap = self.switch_heap[sw]
        prmap = self.switch_slot_prio[sw]
        while heap and prmap.get(heap[0][1], None) != heap[0][0]:
            heapq.heappop(heap)
        if not heap:
            self.switch_used_slots[sw] = min(self.switch_used_slots[sw], cap - 1)
            return True
        lowest_pri, victim_token = heap[0]
        if new_prio > lowest_pri and victim_token != new_token:
            heapq.heappop(heap)
            # 受害 token 的部分和上送（评估阶段通过 _evicted_tokens 计 1 段 switch->PS）
            self._evicted_tokens[sw].add(victim_token)
            self.switch_used_slots[sw] -= 1
            prmap.pop(victim_token, None)
            self.switch_token_counts[sw][victim_token] = 0
            return True
        return False

    def allocate_dsa(self, workers, worker_names):
        self.token_routing = self.moe_config.generate_token_routing(self.tokens)
        self.token_assignments.clear()
        self._evicted_tokens.clear()

        events = []
        for token in self.tokens:
            base = np.random.normal(loc=0.0, scale=self.moe_config.jitter_std)
            for expert_id in range(self.num_experts):
                if self.token_routing.get((token, expert_id), {}).get('activated', 0) == 0:
                    continue
                worker = worker_names[(expert_id) % len(worker_names)]
                events.append({
                    'time': base + expert_id * 0.0005,
                    'token': token,
                    'expert_id': expert_id,
                    'expert_name': f'e{expert_id}',
                    'worker': worker,
                    'priority': self.calculate_priority_dsa(token, expert_id)
                })
        events.sort(key=lambda x: x['time'])

        for ev in events:
            token = ev['token']; expert = ev['expert_name']; worker = ev['worker']; prio = ev['priority']
            sw = self._first_prog_on_path(worker)
            if sw is None:
                self.token_assignments[token][expert] = self.ps
                continue

            if self.switch_token_counts[sw].get(token, 0) > 0:
                self.switch_token_counts[sw][token] += 1
                self.token_assignments[token][expert] = sw
            else:
                if self.switch_used_slots[sw] < self.switch_capacity[sw] or self._evict_min_if_needed(sw, token, prio):
                    if self.switch_token_counts[sw].get(token, 0) == 0:
                        self.switch_used_slots[sw] += 1
                        self.switch_slot_prio[sw][token] = prio
                        heapq.heappush(self.switch_heap[sw], (prio, token))
                    self.switch_token_counts[sw][token] += 1
                    self.token_assignments[token][expert] = sw
                else:
                    self.token_assignments[token][expert] = self.ps

            if self.token_assignments[token][expert] != self.ps and self.switch_token_counts[sw][token] >= self.top_k:
                # 就地聚合→释放槽；聚合结果的上送在评估阶段按 (sw, token) 记 1 段
                self.switch_token_counts[sw][token] = 0
                self.switch_used_slots[sw] -= 1
                self.switch_slot_prio[sw].pop(token, None)
                # 注意：此处不立刻增加计费，保持评估阶段统一处理

        return self.token_assignments

    def run_dsa(self, workers, worker_names, experts, tokens):
        self.tokens = tokens
        self.allocate_dsa(workers, worker_names)
        return self.evaluate_dsa_solution(workers, worker_names)

    def evaluate_dsa_solution(self, workers, worker_names):
        # DSA 的 worker 绑定沿用 (expert_id//8) 规则，传 'dsa'
        return evaluate_metrics_ep(self, workers, worker_names, self.token_assignments, worker_bind='dsa')


# =============================================================
# ATP：自适应阈值策略
# =============================================================
class AdaptiveATP:
    """ATP：沿最短路，首个可编程交换机上若已有该 token 槽或仍有空槽则接纳累加；否则继续向上游找可编程交换机；若全无接纳则在首个可编程交换机转发到 PS；若路径无可编程则直达 PS。"""

    def __init__(self, topo, experts, tokens, moe_config=None, traffic_monitor=None):
        self.topo = topo
        self.moe_config = moe_config or MoEConfig()
        self.traffic_monitor = traffic_monitor
        self.ps = 'ps'
        self.graph = self.topo.graph

        self.experts = experts
        self.tokens = tokens
        self.num_experts = self.moe_config.num_experts
        self.top_k = self.moe_config.top_k
        self.token_routing = {}

        # 片段大小 & 槽位/占用结构（必须先建好，_get_programmable_switches 要用）
        self.SEGMENT_SIZE = int(self.moe_config.expert_output_size_mb * 1024 * 1024)
        self.switch_capacity = {}                                  # sw -> 槽位上限
        self.switch_used_slots = defaultdict(int)                  # sw -> 已用槽位
        self.switch_token_counts = defaultdict(lambda: defaultdict(int))  # sw -> token -> 已累加片段数

        # 路径缓存 & 分配结果
        self._spath_cache = {}
        self.token_assignments = defaultdict(lambda: defaultdict(str))

        # 放到最后再探测可编程交换机（需要上面的结构已存在）
        self.prog_sw = self._get_programmable_switches()

    # def _get_programmable_switches(self):
    #     all_sw = self.topo.switches()
    #     tors = [s for s in all_sw if s.startswith('leaf') or ('_e' in s)]
    #     prog = tors if tors else random.sample(all_sw, max(2, int(0.2 * len(all_sw))))
    #     for sw in prog:
    #         self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024
    #         memB = self.topo.nodeInfo(sw).get('memory', self.moe_config.switch_memory_mb * 1024 * 1024)
    #         self.switch_capacity[sw] = max(1, int(memB // self.SEGMENT_SIZE))
    #     return prog

    def _get_programmable_switches(self):
        """版本1：优先选择TOR交换机，设置内存并计算容量"""
        all_sw = self.topo.switches()
        tors = [s for s in all_sw if s.startswith('leaf') or ('_e' in s)]
        other_sw = [s for s in all_sw if s not in tors]

        # 选择80%的TOR交换机和20%的其他交换机
        num_tors = max(1, int(0.3 * len(tors))) if tors else 0
        num_other = max(1, int(0.6 * len(other_sw))) if other_sw else 0

        prog_tors = random.sample(tors, min(num_tors, len(tors))) if tors else []
        prog_other = random.sample(other_sw, min(num_other, len(other_sw))) if other_sw else []
        prog = prog_tors + prog_other

        for sw in prog:
            self.topo.nodeInfo(sw)['memory'] = 64 * 1024 * 1024
            memB = self.topo.nodeInfo(sw).get('memory', self.moe_config.switch_memory_mb * 1024 * 1024)
            self.switch_capacity[sw] = max(1, int(memB // self.SEGMENT_SIZE))
        return prog

    def calculate_distance(self, node1, node2):
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def _shortest_path(self, u, v):
        key = (u, v)
        if key in self._spath_cache: return self._spath_cache[key]
        try:
            p = nx.shortest_path(self.graph, u, v)
        except nx.NetworkXNoPath:
            p = None
        self._spath_cache[key] = p
        return p

    def _first_prog_on_path(self, worker):
        p = self._shortest_path(worker, self.ps)
        if not p or len(p) <= 2: return None
        for node in p[1:-1]:
            if node in self.prog_sw:
                return node
        return None

    def _scan_first_available_on_path(self, worker, token):
        p = self._shortest_path(worker, self.ps)
        if not p or len(p) <= 2: return None
        for node in p[1:-1]:
            if node in self.prog_sw:
                if self.switch_token_counts[node].get(token, 0) > 0:  # 已有该 token 槽
                    return node
                if self.switch_used_slots[node] < self.switch_capacity[node]:  # 还有空槽
                    return node
        return None

    def run_atp(self, workers, worker_names, experts, tokens):
        self.tokens = tokens
        self.token_routing = self.moe_config.generate_token_routing(tokens)
        self.token_assignments.clear()

        for token in tokens:
            for expert_id in range(self.num_experts):
                if self.token_routing.get((token, expert_id), {}).get('activated', 0) == 0:
                    continue
                expert = f'e{expert_id}'
                worker = worker_names[expert_id % len(worker_names)]

                sw = self._scan_first_available_on_path(worker, token)
                if sw is not None:
                    if self.switch_token_counts[sw].get(token, 0) == 0:
                        self.switch_used_slots[sw] += 1
                    self.switch_token_counts[sw][token] += 1
                    self.token_assignments[token][expert] = sw
                    if self.switch_token_counts[sw][token] >= self.top_k:
                        self.switch_token_counts[sw][token] = 0
                        self.switch_used_slots[sw] -= 1
                else:
                    self.token_assignments[token][expert] = self.ps

        return self.evaluate_atp_solution(workers, worker_names)

    def evaluate_atp_solution(self, workers, worker_names):
        # assignments 已经在 run_atp 中构建： self.token_assignments[token][f'e{expert_id}'] = node
        return evaluate_metrics_ep(self, workers, worker_names, self.token_assignments, worker_bind='mod')


# =============================================================
# 可视化
# =============================================================

def visualize_comparison(results, topo_name):
    """绘制各算法三项指标的对比柱状图。"""
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)
    categories = ['Comm Overhead', 'PS Overhead', 'In-Network Agg']
    algorithms = list(results.keys())
    x = np.arange(len(categories))
    width = 0.2
    for i, algo in enumerate(algorithms):
        values = [
            results[algo]['comm_overhead'] / (1024 ** 2),
            results[algo]['agg_overhead'] / (1024 ** 2),
            results[algo]['in_network_agg'] / (1024 ** 2)
        ]
        ax1.bar(x + i * width, values, width, label=algo)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value (MB)')
    ax1.set_title(f'Algorithm Comparison - {topo_name}')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = plt.subplot(1, 2, 2)
    ratios = []
    for algo in algorithms:
        total = results[algo]['agg_overhead'] + results[algo]['in_network_agg']
        ratio = results[algo]['in_network_agg'] / total * 100 if total > 0 else 0
        ratios.append(ratio)
    bars = ax2.bar(algorithms, ratios)
    ax2.set_ylabel('In-Network Aggregation Ratio (%)')
    ax2.set_title('In-Network Aggregation Efficiency')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{ratio:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    return fig


# =============================================================
# 仿真驱动（循环四组参数）
# =============================================================

def run_simulation(topo_type):
    """运行指定拓扑的完整仿真流程"""
    try:
        gc.collect()

        configs = [(32, 2), (64, 4), (128, 6), (256, 8)]
        # num_experts, top_k = 256, 8
        # jitter_list = [0.0, 0.5, 1.0, 2.0]
        js = 0.0
        # 分别存储三个指标的结果
        all_results = {
            'PS_agg': {'ETO': [], 'DSA': [], 'ATP': [], 'PS-only': []},
            'In_net_agg': {'ETO': [], 'DSA': [], 'ATP': [], 'PS-only': []},
            'fallback_rate': {'ETO': [], 'DSA': [], 'ATP': [], 'PS-only': []},
            'Total_comm': {'ETO': [], 'DSA': [], 'ATP': [], 'PS-only': []}
        }

        for num_experts, top_k in configs:
        # for js in jitter_list:
            print(f"\n=== 运行配置: num_experts={num_experts}, top_k={top_k} ===")
            moe_config = MoEConfig(num_experts=num_experts, top_k=top_k, num_tokens=8000, jitter_std=js)

            if topo_type == 'leaf':
                topo = LeafSpineTopoLight()
                # topo = ScalableLeafSpineTopo(num_workers=moe_config.num_experts)
                topo_name = "Leaf-Spine"
            else:
                topo = FatTreeTopoLight(k=8)
                topo_name = "Fat-Tree"

            workers = {h: h for h in topo.hosts() if h != 'ps'}

            if topo_type == 'leaf':
                worker_names = [f'h{i}_{j}' for i in range(1, 11) for j in range(1, 11)]
            else:
                worker_names = [f'h{pod}_{edge}_{host}' for pod in range(8) for edge in range(1, 5) for host in
                                range(1, 5)]

            worker_names = worker_names[:len(workers)]
            tokens = [f't{i}' for i in range(moe_config.num_tokens)]
            experts = [f'e{i}' for i in range(moe_config.num_experts)]

            # 运行各算法
            # PS-only
            ps_only = PS_Only_MoE(topo, experts, moe_config)
            ps_only_metrics = ps_only.run_ps_only(workers, worker_names, experts, tokens)
            all_results['PS_agg']['PS-only'].append(
                round(ps_only_metrics['agg_overhead'] / (1024 ** 3), 3))  # 转换为GB并保留3位小数
            all_results['In_net_agg']['PS-only'].append(round(ps_only_metrics['in_network_agg'] / (1024 ** 3), 3))
            all_results['fallback_rate']['PS-only'].append(round(ps_only_metrics['fallback_rate'] * 100, 3))
            all_results['Total_comm']['PS-only'].append(round(ps_only_metrics['comm_overhead'] / (1024 ** 3), 3))

            # ETO
            # eto = ETO_MoE(topo, experts, moe_config)
            # eto_metrics = eto.run_eto(workers, worker_names, experts, tokens)
            # all_results['PS_agg']['ETO'].append(round(eto_metrics['agg_overhead'] / (1024 ** 3), 3))
            # all_results['In_net_agg']['ETO'].append(round(eto_metrics['in_network_agg'] / (1024 ** 3), 3))
            # all_results['fallback_rate']['ETO'].append(round(eto_metrics['fallback_rate'] * 100, 3))
            # all_results['Total_comm']['ETO'].append(round(eto_metrics['comm_overhead'] / (1024 ** 3), 3))

            # DSA
            dsa = DSA_MoE(topo, experts, moe_config)
            dsa_metrics = dsa.run_dsa(workers, worker_names, experts, tokens)
            all_results['PS_agg']['DSA'].append(round(dsa_metrics['agg_overhead'] / (1024 ** 3), 3))
            all_results['In_net_agg']['DSA'].append(round(dsa_metrics['in_network_agg'] / (1024 ** 3), 3))
            all_results['fallback_rate']['DSA'].append(round(dsa_metrics['fallback_rate'] * 100, 3))
            all_results['Total_comm']['DSA'].append(round(dsa_metrics['comm_overhead'] / (1024 ** 3), 3))

            # ATP
            atp = AdaptiveATP(topo, experts, tokens, moe_config)
            atp_metrics = atp.run_atp(workers, worker_names, experts, tokens)
            all_results['PS_agg']['ATP'].append(round(atp_metrics['agg_overhead'] / (1024 ** 3), 3))
            all_results['In_net_agg']['ATP'].append(round(atp_metrics['in_network_agg'] / (1024 ** 3), 3))
            all_results['fallback_rate']['ATP'].append(round(atp_metrics['fallback_rate'] * 100, 3))
            all_results['Total_comm']['ATP'].append(round(atp_metrics['comm_overhead'] / (1024 ** 3), 3))

        return all_results

    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    os.environ['PYTHONOPTIMIZE'] = '2'

    print("\nLeaf-Spine拓扑仿真:")
    results_leaf = run_simulation('leaf')

    print("\nFat-Tree拓扑仿真:")
    results_fat = run_simulation('fat-tree')

    # 使用实际的仿真结果
    if results_leaf and results_fat:
        fig10_leaf_spine = {
            "ETO": results_leaf['PS_agg']['ETO'],
            "DSA": results_leaf['PS_agg']['DSA'],
            "ATP": results_leaf['PS_agg']['ATP'],
            "PS-only": results_leaf['PS_agg']['PS-only']
        }

        fig10_fat_tree = {
            "ETO": results_fat['PS_agg']['ETO'],
            "DSA": results_fat['PS_agg']['DSA'],
            "ATP": results_fat['PS_agg']['ATP'],
            "PS-only": results_fat['PS_agg']['PS-only']
        }

        fig11_leaf_spine = {
            "ETO": results_leaf['In_net_agg']['ETO'],
            "DSA": results_leaf['In_net_agg']['DSA'],
            "ATP": results_leaf['In_net_agg']['ATP'],
            "PS-only": results_leaf['In_net_agg']['PS-only']
        }

        fig11_fat_tree = {
            "ETO": results_fat['In_net_agg']['ETO'],
            "DSA": results_fat['In_net_agg']['DSA'],
            "ATP": results_fat['In_net_agg']['ATP'],
            "PS-only": results_fat['In_net_agg']['PS-only']
        }

        fig12_leaf_spine = {
            "ETO": results_leaf['Total_comm']['ETO'],
            "DSA": results_leaf['Total_comm']['DSA'],
            "ATP": results_leaf['Total_comm']['ATP'],
            "PS-only": results_leaf['Total_comm']['PS-only']
        }

        fig12_fat_tree = {
            "ETO": results_fat['Total_comm']['ETO'],
            "DSA": results_fat['Total_comm']['DSA'],
            "ATP": results_fat['Total_comm']['ATP'],
            "PS-only": results_fat['Total_comm']['PS-only']
        }

        fig13_leaf_spine = {
            "ETO": results_leaf['fallback_rate']['ETO'],
            "DSA": results_leaf['fallback_rate']['DSA'],
            "ATP": results_leaf['fallback_rate']['ATP'],
            "PS-only": results_leaf['fallback_rate']['PS-only']
        }

        fig13_fat_tree = {
            "ETO": results_fat['fallback_rate']['ETO'],
            "DSA": results_fat['fallback_rate']['DSA'],
            "ATP": results_fat['fallback_rate']['ATP'],
            "PS-only": results_fat['fallback_rate']['PS-only']
        }

        print("\n=== 最终结果 ===")
        print("fig10_leaf_spine =", fig10_leaf_spine)
        print("fig10_fat_tree =", fig10_fat_tree)
        print("fig11_leaf_spine =", fig11_leaf_spine)
        print("fig11_fat_tree =", fig11_fat_tree)
        print("fig12_leaf_spine =", fig12_leaf_spine)
        print("fig12_fat_tree =", fig12_fat_tree)
        print("fig13_leaf_spine =", fig13_leaf_spine)
        print("fig13_fat_tree =", fig13_fat_tree)
    else:
        print("仿真失败，无法获取结果")
