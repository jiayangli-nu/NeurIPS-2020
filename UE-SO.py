import math
import torch
import networkx as nx
from queue import Queue


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
        self.dcost_vec = None
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
        dist = torch.zeros(n, dtype=torch.float) + math.inf
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
        flow = torch.zeros(m, dtype=torch.float)
        self.f_vec = 1.0 * self.q
        self.cost_vec = torch.zeros_like(self.q)
        for k in range(len(self.q)):
            path = shortest_path[int(ds[k])][int(dt[k])]
            edges = self.edge_serial[path[:-1], path[1:]]
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
        update = list()
        for k in range(len(self.q)):
            path = shortest_path[int(ds[k])][int(dt[k])]
            edges = self.edge_serial[path[:-1], path[1:]]
            c_min = torch.sum(self.time[edges])
            greedy_cost += self.q[k] * c_min
            path_k = self.pool[k]
            if torch.min(torch.abs(self.cost_vec[path_k] - c_min)) < 1e-10:
                continue
            update.append(k)
            self.pool[k].append(n_path + l)
            new_cost.append(c_min)
            for a in edges:
                self.path_vec[0].append(int(a))
                self.path_vec[1].append(n_path + l)
            l += 1
        self.cost_vec = torch.cat((self.cost_vec, torch.tensor(new_cost)))
        zero_vector = torch.zeros(len(new_cost), dtype=torch.float)
        self.f_vec = torch.cat((self.f_vec, zero_vector))
        self.form_path_matrix()

        return greedy_cost, update
    
    def form_path_matrix(self):
        m = self.edge_number
        l = len(self.f_vec)
        i = torch.tensor(self.path_vec, dtype=torch.int64)
        v = torch.ones(len(self.path_vec[0]), dtype=torch.float)
        self.path_mat = torch.sparse.FloatTensor(i, v, torch.Size([m, l]))
        self.q_path = torch.zeros(l, dtype=torch.float)
        for k in range(len(self.q)):
            self.q_path[self.pool[k]] = self.q[k]
        self.p_vec = self.f_vec / self.q_path
   
    def remove_path(self):
        self.f_vec[self.p_vec < 1e-4] = 0
        k_min = torch.nonzero(self.f_vec, as_tuple=True)[0]
        renumber = torch.zeros(len(self.f_vec), dtype=torch.int64)
        renumber[k_min] = torch.arange(len(k_min)) + 1
        self.f_vec = self.f_vec[k_min]
        self.p_vec = self.p_vec[k_min]
        self.cost_vec = self.cost_vec[k_min]
        path = [list(), list()]
        for i in range(len(self.path_vec[0])):
            if renumber[self.path_vec[1][i]] > 0:
                path[0].append(self.path_vec[0][i])
                path[1].append(int(renumber[self.path_vec[1][i]] - 1))
        self.path_vec = path
        for k in range(len(self.q)):
            path_k = list()
            for l in self.pool[k]:
                if renumber[l] > 0:
                    path_k.append(int(renumber[l] - 1))
            self.pool[k] = path_k
               
    def softmin_distribution(self, k):
        path_k = self.pool[k]
        cost = self.cost_vec[path_k]
        gamma = 0.01 
        p = torch.exp(-gamma * cost)
        p /= torch.sum(p)
        self.p_vec[path_k] = p
        self.f_vec[path_k] = p * self.q[k]
        
    def adjustment_distribution(self, k):
        path_k = self.pool[k]
        f = self.f_vec[path_k]
        f += 0.05 * self.q[k]
        p = f / torch.sum(f)
        self.p_vec[path_k] = p
        self.f_vec[path_k] = p * self.q[k]

    def newton_move(self, k):
        path_k = self.pool[k]
        if len(path_k) == 1:
            return
        cost = self.cost_vec[path_k]
        dcost = self.dcost_vec[path_k]
        dcost[dcost == 0] = 1e-10
        f = self.f_vec[path_k]
        c = cost - dcost * f
        order = c.sort()[1]
        B = 1 / dcost[order[0]]
        C = c[order[0]] / dcost[order[0]]
        w = (self.q[k] + C) / B
        
        h = 1
        while h < len(f) and c[order[h]] < w:
            C += c[order[h]] / dcost[order[h]]
            B += 1 / dcost[order[h]]
            w = (self.q[k] + C) / B
            h += 1
        f_new = torch.zeros_like(f)
        for l in torch.arange(h):
            f_new[order[l]] = (w - c[order[l]]) / dcost[order[l]]
        self.f_vec[path_k] = f_new
        
    def no_regret_learning(self):
        r = self.r
        self.p_vec *= torch.exp(-r * self.cost_vec)
        p_sum = torch.zeros_like(self.p_vec)
        for k in range(len(self.q)):
            path_k = self.pool[k]
            p_sum[path_k] = torch.sum(self.p_vec[path_k])
        self.p_vec /= p_sum          
        self.f_vec = self.p_vec * self.q_path
        self.x = sparse_product(self.path_mat, self.f_vec)
        

class Network(object):
    
    def __init__(self, es, et):
        self.edge = torch.stack((es, et))
        n = torch.max(self.edge) + 1
        m = len(es)
        self.node_number = n
        self.edge_number = m
        adjacency = dict()
        self.adjacency = dict()
        self.update = dict()
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
        for u in range(len(self.users)):
            es = self.users[u].es
            et = self.users[u].et
            self.users[u].feasible_edge = self.edge_serial[es, et]
        self.x_sum = torch.zeros(m, dtype=torch.float)
        self.x = torch.zeros(self.class_number, m, dtype=torch.float)
        
        for u in range(len(self.users)):
            fe = self.users[u].feasible_edge
            self.users[u].time = self.users[u].F(self.x[:, fe])
        
    def update_edge_flow(self):
        m = self.edge_number
        x = torch.zeros(self.class_number, m, dtype=torch.float)
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
            
    def update_edge_dcost(self):
        for u in self.active_users:
            fe = self.users[u].feasible_edge
            self.users[u].dtime = self.users[u].dF(self.x[:, fe])
                
    def update_path_cost(self):
        for u in self.active_users:
            path = self.users[u].path_mat
            time = self.users[u].time
            self.users[u].cost_vec = sparse_product(path.t(), time)
            
    def update_path_dcost(self):
        for u in self.active_users:
            path = self.users[u].path_mat
            dtime = self.users[u].dtime
            self.users[u].dcost_vec = sparse_product(path.t(), dtime)  
    
    def hybrid_method(self, active_users, epsilon=1e-3):
        self.active_users = active_users
        self.gap = list()
        
        m = self.edge_number
        self.x = torch.zeros(self.class_number, m, dtype=torch.float)
        self.update_edge_cost()     
        for u in self.active_users:
            self.users[u].initialize_path()
            self.users[u].form_path_matrix()    
        self.update_edge_flow()
        self.update_edge_cost()
        self.update_path_cost()
        delta = 1e-2
        iter_number = 0
        while iter_number < 1000:
            """Calculate Reletive Gap"""
            greedy_cost = 0
            actual_cost = 0
            n_update = 0
            for u in self.active_users:
                cost, update = self.users[u].update_path()
                n_update += len(update)
                greedy_cost += cost
                actual_cost += torch.dot(self.users[u].time, self.users[u].x)
                self.update[u] = update
            rg = 1 - greedy_cost / actual_cost
            print('iter:', iter_number, 'relative gap:', rg)
            if rg < epsilon:
                break
            
            delta = min(delta, rg)
            if n_update == 0:
                delta = max(0.1 * delta, 0.8 * epsilon)
            else:
                #print('find', n_update, 'new paths')
                for u in self.active_users:
                    for k in self.update[u]:
                        self.users[u].adjustment_distribution(k)
                for u in self.active_users:
                    path = self.users[u].path_mat
                    f = self.users[u].f_vec
                    self.users[u].x = sparse_product(path, f)
                self.update_edge_flow()
                self.update_edge_cost()
                self.update_path_cost()
            
            """Entropic Learning"""
            self.r = 1
            for u in self.active_users:
                self.users[u].r = self.r
            irg = math.inf    
            while irg > delta:
                """Calculate Restricted Reletive Gap"""
                greedy_cost = 0
                actual_cost = 0
                for u in self.active_users:
                    for k in range(len(self.users[u].q)):
                        path_k = self.users[u].pool[k]
                        c_min = torch.min(self.users[u].cost_vec[path_k])
                        greedy_cost += c_min * self.users[u].q[k]
                    time = self.users[u].time
                    x = self.users[u].x
                    actual_cost += torch.dot(time, x)
                irg = 1 - greedy_cost / actual_cost
                for u in self.active_users:                
                    self.users[u].no_regret_learning()     
                self.update_edge_flow()
                self.update_edge_cost()
                self.update_path_cost()
            
            """Remove Path"""
            for u in self.active_users:
                self.users[u].remove_path()
            
            iter_number += 1
            
        for u in self.active_users:
            self.users[u].remove_path()
            self.users[u].form_path_matrix()    
                
        return self.x
    
    def entropic_learning(self, active_users, epsilon=1e-3, display=False):
        self.active_users = active_users
        with torch.no_grad():
            m = self.edge_number
            for u in self.active_users:
                self.x[u, :] = torch.zeros(m, dtype=torch.float)
            self.update_edge_cost()
            self.update_path_cost()
            for u in self.active_users:
                l = len(self.users[u].p_vec)
                self.users[u].p_vec = torch.zeros(l, dtype=torch.float)
                self.users[u].f_vec = torch.zeros(l, dtype=torch.float)
                for k in range(len(self.users[u].q)):
                    self.users[u].softmin_distribution(k)
            for u in self.active_users:
                path = self.users[u].path_mat
                f = self.users[u].f_vec
                self.users[u].x = sparse_product(path, f)
            self.update_edge_flow()
            self.update_edge_cost()
            self.update_path_cost()
        
        self.r = 1
        for u in self.active_users:
            self.users[u].r = self.r
        rg = math.inf
        iter_number = 0
        while rg > epsilon:
            """Calculate Restricted Reletive Gap"""
            with torch.no_grad():
                greedy_cost = 0
                actual_cost = 0
                for u in self.active_users:
                    for k in range(len(self.users[u].q)):
                        path_k = self.users[u].pool[k]
                        c_min = torch.min(self.users[u].cost_vec[path_k])
                        greedy_cost += c_min * self.users[u].q[k]
                    time = self.users[u].time
                    x = self.users[u].x
                    actual_cost += torch.dot(time, x)
                rg = 1 - greedy_cost / actual_cost
                if rg < 0:
                    print('error')
                if display:
                    print(rg)
            for u in self.active_users:                
                self.users[u].no_regret_learning()     
            self.update_edge_flow()
            self.update_edge_cost()
            self.update_path_cost()
                
            iter_number += 1
            if iter_number > 50:
                break
                
        return self.x
    

def sparse_product(spmatrix, vector):
    mvector = torch.mm(spmatrix, vector.unsqueeze(1))
    if len(mvector) > 1:
        return mvector.squeeze()
    else:
        return mvector.squeeze(0)


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

cap = torch.tensor([float(link[4]) for link in link_data], dtype=torch.float)
tfree = torch.tensor([float(link[2]) / float(link[3])  for link in link_data], dtype=torch.float)

a_node = [od[0] for od in od_data]
a_number = torch.tensor([node_number[name] for name in a_node], dtype=torch.int64)
b_node = [od[1] for od in od_data]
b_number = torch.tensor([node_number[name] for name in b_node], dtype=torch.int64)
demand = torch.tensor([float(od[2]) for od in od_data], dtype=torch.float)

network = Network(s_number, t_number)

F = lambda x: tfree * (1 + 0.15 * (sum(x) / cap) ** 4)

alpha = 0.8
leader = User(network.node_number)
leader.set_demand(a_number, b_number, alpha * demand)
leader.set_edge(s_number, t_number, F)

follower = User(network.node_number)
follower.set_demand(a_number, b_number, (1 - alpha) * demand)
follower.set_edge(s_number, t_number, F)

network.set_users([leader, follower])
active_users = [0, 1]
network.hybrid_method(active_users, epsilon=1e-4)
T = torch.dot(network.x_sum, F(network.x))
print(T)
leader.p_vec.requires_grad_()
gamma = 1
descent_number = 0
while descent_number < 200:
    #print(leader.x, follower.x)

    leader.f_vec = leader.p_vec * leader.q_path
    path = leader.path_mat
    f = leader.f_vec
    leader.x = sparse_product(path, f)
    active_users = [1]
    network.entropic_learning(active_users, epsilon=5 * 1e-4, display=False)
    network.active_users = [0, 1]
    network.update_edge_flow()
    network.update_edge_cost()
    network.update_path_cost()
    T_leader = torch.dot(leader.x, leader.time)
    T_follower = torch.dot(follower.x, follower.time)
    T = T_leader + T_follower
    T_leader.backward()
    print(T_leader)
    
    with torch.no_grad():
        #print(leader.p_vec.grad)
        leader.p_vec *= torch.exp(-gamma * leader.p_vec.grad)
        p_sum = torch.zeros_like(leader.p_vec)
        for k in range(len(leader.q)):
            path_k = leader.pool[k]
            p_sum[path_k] = torch.sum(leader.p_vec[path_k])
        leader.p_vec /= p_sum
    leader.p_vec.grad.zero_()
    descent_number += 1
