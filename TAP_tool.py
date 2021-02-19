import time
import math
import torch
import cvxpy as cp
import networkx as nx
from queue import Queue
from cvxpylayers.torch import CvxpyLayer


class User(object):
    
    def __init__(self, n):
        self.node_number = n
        self.ds = None
        self.dt = None
        self.q = None # Tensor of Denmand of O-D pairs
        self.q_path = None
        self.demand_number = None # Number of O-D pairs
        self.od = dict() # od[source] = list of sinks
        self.od_serial = torch.zeros(n, n, dtype=torch.int64)
        # od_serial[source, sink] = O-D serial number
        self.source = None # Tensor of souce nodes
        
        self.edge_number = None # number of edges
        self.F = None # edge cost funcion
        self.time = None # edge cost
        self.dtime = None
        self.x = None # edge flow
        self.x_od = dict()
        self.es = None # tensor of source nodes of edges
        self.et = None # tensor of sink nodes of edges
        self.intF = None # integral of cost function
        self.dF = None # derivative of cost function
        
        self.adjacency = dict() # adjacency[node] = tensor of adjancen nodes
        self.edge_serial = None 
        # edge_serial[source, sink] = edge serial number
        self.f = dict()
        self.p_vec = None
        self.f_vec = None
        self.pool = dict()
        self.path = dict()
        self.path_vec = [list(), list()]
        self.path_mat = None
        self.cost = dict()
        self.cost_vec = None
        self.rgap = dict()
        self.indicator = dict()
    
        
    def set_demand(self, ds, dt, q):
        self.q = q
        self.ds = ds
        self.dt = dt
        self.demand_number = len(q)
        self.od_serial[ds, dt] = torch.arange(self.demand_number)
        
        for k, i in enumerate(ds):
            if not int(i) in self.od:
                self.od[int(i)] = list()
            self.od[int(i)].append(int(dt[k]))
        
        self.source = torch.tensor(list(set(self.od)), dtype=torch.int64)
        for i in self.source:
            self.od[int(i)] = torch.tensor(self.od[int(i)], dtype=torch.int64)            
        
    def set_edge(self, es, et, F, intF=None, dF=None):
        m = len(es)
        self.edge_number = m
        self.F = F
        self.time = torch.zeros(m)
        self.x = torch.zeros(m)
        self.es = es
        self.et = et
        self.update_topology()
        if intF:
            self.potential = lambda x: sum(intF(x))
        if dF:
            self.dF = dF
            
    def update_topology(self):
        n = self.node_number
        m = self.edge_number
        adjacency = dict()
        for i in torch.arange(n):
            adjacency[int(i)] = list()
        for k, i in enumerate(self.es):
            adjacency[int(i)].append(int(self.et[k]))
        for i in torch.arange(n):
            neighbor = adjacency[int(i)]
            self.adjacency[int(i)] = torch.tensor(neighbor, dtype=torch.int64)
        self.edge_serial = torch.zeros(n, n, dtype=torch.int64)
        self.edge_serial[self.es, self.et] = torch.arange(m)
        
    def shortest_path_tree(self, source):
        n = self.node_number
        dist = torch.zeros(n, dtype=torch.double) + math.inf
        dist[source] = 0
        prev = torch.zeros(n, dtype=torch.int64) - 1
        que = Queue(maxsize=n)
        que.put(source)
        que_content = torch.zeros(n, dtype=torch.bool)
        que_content[source] = True
        while not que.empty():
            i = que.get()
            que_content[i] = False
            for j in self.adjacency[int(i)]:
                k = self.edge_serial[i, j]
                with torch.enable_grad():
                    alt = dist[i] + self.time[k]
                if dist[j] > alt:
                    with torch.enable_grad():
                        dist[j] = alt
                    prev[j] = i
                    if not que_content[j]:
                        que.put(j)
                        que_content[j] = True
        return prev, dist
    
    def initialize_path(self):
        m = self.edge_number
        es = self.es
        et = self.et
        ds = self.ds
        dt = self.dt
        G = nx.DiGraph()
        E = [(int(es[k]), int(et[k]), self.time[k]) for k in range(m)]
        G.add_weighted_edges_from(E)
        shortest_path = dict(nx.all_pairs_bellman_ford_path(G))
        flow = torch.zeros(m, dtype=torch.double)
        self.f_vec = 1.0 * self.q
        self.cost_vec = torch.zeros_like(self.q)
        for k in range(len(self.q)):
            path = shortest_path[int(ds[k])][int(dt[k])]
            edges = self.edge_serial[es[path[:-1]], et[path[1:]]]
            self.cost_vec[k] = torch.sum(self.time[edges])
            self.pool[k] = [k]
            flow[edges] += self.q[k]
            for a in edges:
                self.path_vec[0].append(int(a))
                self.path_vec[1].append(k)
        self.x = flow
        
    def update_path(self):
        m = self.edge_number
        es = self.es
        et = self.et
        ds = self.ds
        dt = self.dt
        G = nx.DiGraph()
        E = [(int(es[k]), int(et[k]), self.time[k]) for k in range(m)]
        G.add_weighted_edges_from(E)
        shortest_path = dict(nx.all_pairs_bellman_ford_path(G))
        greedy_cost = 0
        n_path = len(self.f_vec)
        new_cost = list()
        l = 0
        for k in range(len(self.q)):
            path = shortest_path[int(ds[k])][int(dt[k])]
            edges = self.edge_serial[es[path[:-1]], et[path[1:]]]
            c_min = torch.sum(self.time[edges])
            greedy_cost += self.q[k] * c_min
            path_k = self.pool[k]
            if torch.min(torch.abs(self.cost_vec[path_k] - c_min)) < 1e-10:
                continue
            self.pool[k].append(n_path + l)
            new_cost.append(c_min)
            for a in edges:
                self.path_vec[0].append(int(a))
                self.path_vec[1].append(n_path + l)
            l += 1
        self.cost_vec = torch.cat((self.cost_vec, torch.tensor(new_cost)))
        zero_vector = torch.zeros(len(new_cost), dtype=torch.double)
        self.f_vec = torch.cat((self.f_vec, zero_vector))
        self.vectorization()

        return greedy_cost
    
    def vectorization(self):
        m = self.edge_number
        l = len(self.f_vec)
        i = torch.tensor(self.path_vec, dtype=torch.int64)
        v = torch.ones(len(self.path_vec[0]), dtype=torch.double)
        self.path_mat = torch.sparse.FloatTensor(i, v, torch.Size([m, l]))
        self.q_path = torch.zeros(l, dtype=torch.double)
        for k in range(len(self.q)):
            self.q_path[self.pool[k]] = self.q[k]
            
#    def vectorization(self):
#        m = self.edge_number
#        i_edge = list()
#        i_path = list()
#        l = 0
#        for source in self.source:
#            for sink in self.od[int(source)]:
#                k = self.od_serial[source, sink]
#                self.pool[int(k)] = l + torch.arange(len(self.f[int(k)]))
#                for pair in self.path[int(k)]:
#                    i_edge.append(pair[0])
#                    i_path.append(l + pair[1])
#                l += len(self.f[int(k)])
#        i = torch.tensor([i_edge, i_path], dtype=torch.int64)
#        v = torch.ones(len(i_edge), dtype=torch.double)
#        self.path_mat = torch.sparse.FloatTensor(i, v, torch.Size([m, l]))
#        self.f_vec = torch.zeros(l, dtype=torch.double)
#        self.p_vec = torch.zeros(l, dtype=torch.double)
#        self.q_path = torch.zeros(l, dtype=torch.double)
#        self.cost_vec = torch.zeros(l, dtype=torch.double)
#        for source in self.source:
#            for sink in self.od[int(source)]:
#                k = self.od_serial[source, sink]
#                self.q_path[self.pool[int(k)]] = self.q[int(k)]
#                self.f_vec[self.pool[int(k)]] = self.f[int(k)]
#                self.p_vec[self.pool[int(k)]] = self.f[int(k)] / self.q[int(k)]
#                self.cost_vec[self.pool[int(k)]] = self.cost[int(k)]
#    
    def remove_path(self):
        for source in self.source:
            for sink in self.od[int(source)]:
                k = self.od_serial[source, sink]
                k_min = torch.nonzero(self.f[int(k)], as_tuple=True)[0]
                renumber = torch.zeros(len(self.f[int(k)]), dtype=torch.int64)
                renumber[k_min] = torch.arange(len(k_min)) + 1
                self.f[int(k)] = self.f[int(k)][k_min]
                self.cost[int(k)] = self.cost[int(k)][k_min]
                path = list()
                for pair in self.path[int(k)]:
                    if renumber[pair[1]] > 0:
                        path.append([pair[0], renumber[pair[1]] - 1])
                self.path[int(k)] = path
          
    def projection_eu(self, o, d):
        r = self.r
        k = self.od_serial[o, d]
        if len(self.f[int(k)]) == 1:
            return self.f[int(k)]
        
        flow = cp.Variable(len(self.f[int(k)]))
        b = cp.Parameter(len(self.f[int(k)]))
        constraints = [flow >= 0, sum(flow) == self.q[k]]
        objective = cp.Minimize(cp.pnorm(flow - b, p=2))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        b_tch = self.f[int(k)] - r * self.cost[int(k)]
        cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[flow])
        solution,  = cvxpylayer(b_tch)
        solution = solution.clamp(0)
        return solution
    
    def projection_kl(self, o, d):
        r = self.r
        k = self.od_serial[o, d]
        if len(self.f[int(k)]) == 1:
            return self.f[int(k)]
        p = self.f[int(k)] / self.q[k]
        p *= torch.exp(-r * self.cost[int(k)])
        solution = p / sum(p) * self.q[k]
        return solution
    
    def newton_move(self, o, d):
        k = self.od_serial[o, d]
        if len(self.f[int(k)]) == 1:
            return
        cost = self.assign_path_cost(o, d)
        dcost = self.assign_path_dcost(o, d)
        f = self.f[int(k)]
        c = cost - dcost * f
        order = c.sort()[1]
        B = 1 / dcost[order[0]]
        C = c[order[0]] / dcost[order[0]]
        w = (self.q[k] + C) / B
        f_new = torch.zeros_like(f)
        h = 1
        while h < len(self.f[int(k)]) and c[order[h]] < w:
            C += c[order[h]] / dcost[order[h]]
            B += 1 / dcost[order[h]]
            w = (self.q[k] + C) / B
            h += 1
        f_new = torch.zeros_like(f)
        for l in torch.arange(h):
            f_new[order[l]] = (w - c[order[l]]) / dcost[order[l]]
        self.x -= self.assign_edge_flow(o, d)
        self.f[int(k)] = f_new
        self.x += self.assign_edge_flow(o, d)
        
    def no_regret_learning(self):
        r = self.r
        self.p_vec *= torch.exp(-r * self.cost_vec)
        p_sum = torch.zeros_like(self.p_vec)
        for o in self.source:
            for d in self.od[int(o)]:
                k = self.od_serial[o, d]
                path_k = self.pool[int(k)]
                p = self.p_vec[path_k]
                p_sum[path_k] = torch.sum(p)
        self.p_vec /= p_sum
                
        self.f_vec = self.p_vec * self.q_path
        self.x = sparse_product(self.path_mat, self.f_vec)
    
    def assign_edge_flow(self, o, d):
        m = self.edge_number
        k = self.od_serial[o, d]
        flow = torch.zeros(m, dtype=torch.double)
        for pair in self.path[int(k)]:
            flow[pair[0]] += self.f[int(k)][pair[1]]
        return flow
    
    def assign_path_cost(self, o, d):
        k = self.od_serial[o, d]
        cost = torch.zeros(len(self.f[int(k)]), dtype=torch.double)
        for pair in self.path[int(k)]:
            cost[pair[1]] += self.time[pair[0]]
        return cost
    
    def assign_path_dcost(self, o, d):
        k = self.od_serial[o, d]
        dcost = torch.zeros(len(self.f[int(k)]), dtype=torch.double)
        for pair in self.path[int(k)]:
            dcost[pair[1]] += self.dtime[pair[0]]
        return dcost


class Network(object):
    
    def __init__(self, es, et):
        self.edge = torch.stack((es, et))
        n = torch.max(self.edge) + 1
        m = len(es)
        self.node_number = n
        self.edge_number = m
        adjacency = dict()
        self.adjacency = dict()
        for i in torch.arange(n):
            adjacency[int(i)] = list()
        for k, i in enumerate(es):
            adjacency[int(i)].append(int(et[k]))
        for i in torch.arange(n):
            neighbor = adjacency[int(i)]
            self.adjacency[int(i)] = torch.tensor(neighbor, dtype=torch.int64)
        self.edge_serial = torch.zeros(n, n, dtype=torch.int64)
        self.edge_serial[es, et] = torch.arange(m)
        
    def set_users(self, users):
        m = self.edge_number
        self.users = users
        self.class_number = len(self.users)
        with torch.no_grad():
            for u in range(len(self.users)):
                es = self.users[u].es
                et = self.users[u].et
                self.users[u].feasible_edge = self.edge_serial[es, et]
            self.x_sum = torch.zeros(m, dtype=torch.double)
            self.x = torch.zeros(self.class_number, m, dtype=torch.double)
        
        with torch.enable_grad():
            for u in range(len(self.users)):
                fe = self.users[u].feasible_edge
                self.users[u].time = self.users[u].F(self.x[:, fe])
                
    def update(self):
        self.update_edge_flow()
        self.update_edge_cost()
        self.update_path_cost()
        
    def update_edge_flow(self):
        m = self.edge_number
        x = torch.zeros(self.class_number, m, dtype=torch.double)
        for u in range(len(self.users)):
            fe = self.users[u].feasible_edge
            x[u, fe] = self.users[u].x
        x_sum = sum(x)
        self.x_sum = x_sum
        self.x = x
        
    def update_edge_cost(self):
        for u in self.active_users:
            fe = self.users[u].feasible_edge
            self.users[u].time = self.users[u].F(self.x[:, fe])
                
    def update_path_cost(self):
        for u in self.active_users:
            self.users[u].cost_vec = sparse_product(self.users[u].path_mat.t(), self.users[u].time)    
                
    def adjust_r(self):
        accelerate = True
        for u in self.active_users:
            f_original = self.users[u].f.clone()
            self.users[u].eta = 10 / torch.max(self.users[u].cost)
            adjust_number = 0
            while adjust_number < 10:
                adjust_number += 1
                self.users[u].f = self.users[u].projection_kl()
                self.users[u].x = self.users[u].path.double() @ self.users[u].f
                self.update()
                
                y = torch.zeros_like(self.users[u].x)
                for o in self.users[u].source:
                    for d in self.users[u].od[int(o)]:
                        k = self.users[u].od_serial[o, d]
                        pk = self.users[u].indicator == k
                        pathk = self.users[u].path[:, pk]
                        costk = self.users[u].cost[pk]
                        loc = torch.min(costk, 0)[1]
                        y += self.users[u].q[k] * pathk[:, loc]
                d = y - self.users[u].x
                g = -torch.dot(d, self.users[u].time)
                self.users[u].f = f_original.clone()
                self.users[u].x = self.users[u].path.double() @ self.users[u].f
                self.update()
                if g > self.gap[-1][u]:
                    self.users[u].eta *= 0.5
                else:
                    break
            else:
                accelerate = False
                break
        return accelerate
                        
    def greedy_method(self, active_users, epsilon=1e-3, delta=0, display=False):
        self.active_users = active_users
        self.r = 0.5
        for u in self.active_users:
            self.users[u].r = self.r
        self.gap = list()
        
        m = self.edge_number
        self.x_sum = torch.zeros(m, dtype=torch.double)
        self.x = torch.zeros(self.class_number, m, dtype=torch.double)
        self.update_edge_cost()     
        for u in self.active_users:
            self.users[u].initialize_path()
        self.update()
        
        iter_number = 0
        rg_old = math.inf

        while iter_number < 1000:
            greedy_cost = 0
            actual_cost = 0
            for u in self.active_users:
                greedy_cost += self.users[u].update_path()
                actual_cost += torch.dot(self.users[u].time, self.users[u].x)
            rg = 1 - greedy_cost / actual_cost
            g = actual_cost - greedy_cost
            if display:
                print('iter:', iter_number, 'relative gap:', rg, 'gap', g)     
            
            self.gap.append(rg)            
            if rg < epsilon:
                break

            if rg > delta:
                if rg / rg_old > 0.999:
                    self.r = max(0.8 * self.r, 1e-5)
                    for u in range(len(self.users)):
                        self.users[u].r = self.r
                rg_old = rg
                for u in self.active_users:
                    self.users[u].x = torch.zeros(m, dtype=torch.double)
                    for o in self.users[u].source:
                        for d in self.users[u].od[int(o)]:
                            k = self.users[u].od_serial[o, d]
                            self.users[u].f[int(k)] = self.users[u].projection_eu(o, d)
                            self.users[u].x += self.users[u].assign_edge_flow(o, d)
                self.update()
            else:
                for u in self.active_users:
                    for o in self.users[u].source:
                        sink = self.users[u].od[int(o)]
                        for d in sink:
                            fe = self.users[u].feasible_edge
                            self.users[u].dtime = self.users[u].dF(self.x[:, fe])
                            self.users[u].newton_move(o, d)
                            self.update_edge_flow()
                            self.users[u].time = self.users[u].F(self.x[:, fe])
#                l = 0
#                FC_old = math.inf
#                while l < 50:
#                    FC = 0
#                    for u in self.active_users:
#                        for o in self.users[u].source:
#                            for d in self.users[u].od[int(o)]:
#                                k = self.users[u].od_serial[o, d]
#                                c_max = torch.max(self.users[u].cost[int(k)])
#                                c_min = torch.min(self.users[u].cost[int(k)])
#                                self.users[u].rgap[int(k)] = c_max - c_min
#                                if self.users[u].rgap[int(k)] > 0.5 * rg:
#                                    FC += 1
#                                    fe = self.users[u].feasible_edge
#                                    self.users[u].dtime = self.users[u].dF(self.x[:, fe])
#                                    self.users[u].newton_move(o, d)
#                    self.update()                
#                    if FC == FC_old:
#                        break
#                    FC_old = FC
#                    l += 1
                
            iter_number += 1                
            
            for u in self.active_users:
                self.users[u].remove_path()
        self.update_path_cost()
                
        return self.x_sum
    
    def hybrid_method(self, active_users, epsilon=1e-3, delta=0, display=False):
        self.active_users = active_users
        self.r = 0.5
        for u in self.active_users:
            self.users[u].r = self.r
        self.gap = list()
        
        m = self.edge_number
        self.x_sum = torch.zeros(m, dtype=torch.double)
        self.x = torch.zeros(self.class_number, m, dtype=torch.double)
        self.update_edge_cost()     
        for u in self.active_users:
            self.users[u].initialize_path()
            self.users[u].vectorization()
            
        self.update()
        
        iter_number = 0
        rg_old = math.inf

        while iter_number < 1000:
            greedy_cost = 0
            actual_cost = 0
            for u in self.active_users:
                greedy_cost += self.users[u].update_path()
                actual_cost += torch.dot(self.users[u].time, self.users[u].x)
            rg = 1 - greedy_cost / actual_cost
            g = actual_cost - greedy_cost
            if display:
                print('iter:', iter_number, 'relative gap:', rg, 'gap', g)     
            
            self.gap.append(rg)            
            if rg < epsilon:
                break

            if rg > delta:
                if rg / rg_old > 0.999:
                    self.r = max(0.8 * self.r, 1e-5)
                    for u in range(len(self.users)):
                        self.users[u].r = self.r
                rg_old = rg
                for u in self.active_users:
                    self.users[u].x = torch.zeros(m, dtype=torch.double)
                    for o in self.users[u].source:
                        for d in self.users[u].od[int(o)]:
                            k = self.users[u].od_serial[o, d]
                            self.users[u].f[int(k)] = self.users[u].projection_eu(o, d)
                            self.users[u].x += self.users[u].assign_edge_flow(o, d)
                self.update()
            else:
                for u in self.active_users:
                    for o in self.users[u].source:
                        sink = self.users[u].od[int(o)]
                        for d in sink:
                            fe = self.users[u].feasible_edge
                            self.users[u].dtime = self.users[u].dF(self.x[:, fe])
                            self.users[u].newton_move(o, d)
                            self.update_edge_flow()
                            self.users[u].time = self.users[u].F(self.x[:, fe])
                            
            iter_number += 1                
            
            for u in self.active_users:
                self.users[u].remove_path()
        self.update_path_cost()
                
        return self.x_sum
    
    def entropic_learning(self, active_users, epsilon=1e-3, display=False):
        self.active_users = active_users
        self.r = 0.5
        for u in self.active_users:
            self.users[u].r = self.r
        self.gap = list()
        
        m = self.edge_number
        self.x_sum = torch.zeros(m, dtype=torch.double)
        self.x = torch.zeros(self.class_number, m, dtype=torch.double)
        self.update_edge_cost()
        self.update_path_cost()
        for u in self.active_users:
            self.users[u].x = torch.zeros(m, dtype=torch.double)
            for o in self.users[u].source:
                for d in self.users[u].od[int(o)]:
                    k = self.users[u].od_serial[o, d]
                    n_path = len(self.users[u].f[int(k)])
                    l_min = torch.argmin(self.users[u].cost[int(k)])
                    if n_path > 1:
                        p = torch.zeros(n_path, dtype=torch.double) + 0.05 / (n_path - 1)
                        p[l_min] = 0.95
                    else:
                        p = torch.ones(1, dtype=torch.double)
                    self.users[u].f[int(k)] = p * self.users[u].q[int(k)]
                    self.users[u].x += self.users[u].assign_edge_flow(o, d)
        self.update()
        
        for u in self.active_users:
            self.users[u].vectorization()
        
        iter_number = 0
        rg_old = math.inf
        
        while iter_number < 100:
            greedy_cost = 0
            actual_cost = 0
            for u in self.active_users:
                for o in self.users[u].source:
                    for d in self.users[u].od[int(o)]:
                        k = self.users[u].od_serial[o, d]
                        path_k = self.users[u].pool[int(k)]
                        c_min = torch.min(self.users[u].cost_vec[path_k])
                        greedy_cost += c_min * self.users[u].q[int(k)]
                actual_cost += torch.dot(self.users[u].time, self.users[u].x)
            rg = 1 - greedy_cost / actual_cost
            g = actual_cost - greedy_cost
            if display:
                print('iter:', iter_number, 'relative gap:', rg, 'gap', g)     
            
            self.gap.append(rg)            
            if rg < epsilon:
                break

            if rg / rg_old >= 1:
                self.r = max(0.8 * self.r, 1e-5)
                for u in range(len(self.users)):
                    self.users[u].r = self.r
            rg_old = rg
            for u in self.active_users:                
                self.users[u].no_regret_learning()
                
            self.update_edge_flow()
            self.update_edge_cost()
            for u in self.active_users:
                self.users[u].cost_vec = sparse_product(self.users[u].path_mat.t(), self.users[u].time)
      
            iter_number += 1
                
        return self.x_sum
    

def sparse_product(spmatrix, vector):
    return torch.mm(spmatrix, vector.unsqueeze(1)).squeeze()


with open('sf_edge.txt', 'r') as f:
    link_data = f.readlines()
    link_data = [line.split() for line in link_data]
    
with open('sf_demand.txt', 'r') as f:
    od_data = f.readlines()
    od_data = [line.split() for line in od_data]

# Preprocess
s_node = [link[0] for link in link_data]
t_node = [link[1] for link in link_data]
node = list(set(s_node + t_node))
node.sort()

node_number = dict(zip(node, range(len(node))))
s_number = torch.tensor([node_number[name] for name in s_node], dtype=torch.int64)
t_number = torch.tensor([node_number[name] for name in t_node], dtype=torch.int64)

cap = torch.tensor([float(link[4]) for link in link_data], dtype=torch.double)
tfree = torch.tensor([float(link[2]) / float(link[3])  for link in link_data], dtype=torch.double)

a_node = [od[0] for od in od_data]
a_number = torch.tensor([node_number[name] for name in a_node], dtype=torch.int64)
b_node = [od[1] for od in od_data]
b_number = torch.tensor([node_number[name] for name in b_node], dtype=torch.int64)
demand = torch.tensor([float(od[2]) for od in od_data], dtype=torch.double)

network = Network(s_number, t_number)

Time = lambda x: tfree * (1 + 0.15 * (x / cap) ** 4)

F = lambda x: tfree * (1 + 0.15 * (sum(x) / cap) ** 4)
dF = lambda x: 0.15 * 4 * (sum(x) / cap) ** 3 / cap
intF = lambda x: tfree * (sum(x) + 0.15 * cap / 5 * (sum(x) / cap) ** 5)

user = User(network.node_number)
user.set_demand(a_number, b_number, demand)
user.set_edge(s_number, t_number, F, intF=intF, dF=dF)

network.set_users([user])
active_users = [0]
tic = time.time()
network.hybrid_method(active_users, epsilon=1e-4, delta=1, display=True)
toc = time.time()
time1 = toc - tic
T = torch.dot(network.x_sum, Time(network.x_sum))
#
#cap.requires_grad_()
#tic = time.time()
#network.entropic_learning(active_users, epsilon=1e-4, display=True)
#toc = time.time()
#time2 = toc - tic
#T = torch.dot(network.x_sum, Time(network.x_sum))
#T.backward()
#print(cap.grad)
