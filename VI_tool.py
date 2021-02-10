import math
import torch
import cvxpy as cp
from queue import Queue
from cvxpylayers.torch import CvxpyLayer


class User(object):
    
    def __init__(self, n):
        self.node_number = n
        self.q = None # Tensor of Denmand of O-D pairs
        self.demand_number = None # Number of O-D pairs
        self.od = dict() # od[source] = list of sinks
        self.od_serial = torch.zeros(n, n, dtype=torch.int64)
        # od_serial[source, sink] = O-D serial number
        self.source = None # Tensor of souce nodes
        
        self.edge_number = None # number of edges
        self.F = None # edge cost funcion
        self.time = None # edge cost
        self.x = None # edge flow
        self.es = None # tensor of source nodes of edges
        self.et = None # tensor of sink nodes of edges
        self.intF = None # integral of cost function
        self.F_x = None # derivative of cost function
        
        self.adjacency = dict() # adjacency[node] = tensor of adjancen nodes
        self.edge_serial = None 
        # edge_serial[source, sink] = edge serial number
        
        self.Ma = None
   
    def set_demand(self, ds, dt, q):
        self.q = q
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
            self.F_x = dF
        
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
    
    def tree_travel(self, source):
        m = self.edge_number
        n = self.node_number
        prev, dist = self.shortest_path_tree(source)
        flow = torch.zeros(m, dtype=torch.double)
        path = torch.zeros(m, n, dtype=torch.int8).bool()
        for sink in self.od[int(source)]:
            t = sink
            while t != source:
                s = prev[t]
                k = self.edge_serial[s, t]
                flow[k] += self.q[self.od_serial[source, sink]]
                path[k, sink] = True
                t = s

        return flow, path, dist
    
    def initialize_path(self):
        all_or_nothing = [self.tree_travel(i) for i in self.source]
        with torch.enable_grad():
            path_pool = [harvest[1] for harvest in all_or_nothing]
            path_indicator = [None for _ in all_or_nothing]
            path_cost = [harvest[2] for harvest in all_or_nothing]
            for k, i in enumerate(self.source):
                sinks = self.od[int(i)]
                path_indicator[k] = self.od_serial[i, sinks]
                path_pool[k] = path_pool[k][:, sinks]
                path_cost[k] = path_pool[k].t().double() @ self.time
            
            self.path = torch.cat(path_pool, 1)
            self.indicator = torch.cat(path_indicator)
            M = torch.eq(self.indicator.unsqueeze(0), torch.arange(self.demand_number).unsqueeze(1)).double()
            self.M = M
            self.f = self.q[self.indicator]
            self.x = self.path.double() @ self.f
    
    def update_path(self, source):
        sinks = self.od[int(source)]
        ks = self.od_serial[source, sinks]
        flow, path, dist = self.tree_travel(source)
        new_path = torch.zeros(self.node_number, dtype=torch.bool)
            
        for i in torch.arange(len(sinks)):
            k, sink = [ks[i], sinks[i]]
            compare = torch.eq(path[:, sink].unsqueeze(1), self.path[:, self.indicator == k])
            new_path[sink] = not torch.any(torch.all(compare, 0))
        
        if sum(new_path) > 0:
            self.path = torch.cat((self.path, path[:, new_path]), 1)
            with torch.enable_grad():
                self.cost = torch.cat((self.cost, dist[new_path]))
                self.f = torch.cat((self.f, torch.zeros(new_path.sum(), dtype=torch.double)))
            self.indicator = torch.cat((self.indicator, self.od_serial[source, new_path]))
            M = torch.eq(self.indicator.unsqueeze(0), torch.arange(self.demand_number).unsqueeze(1)).double()
            self.M = M
        return flow
        
    def remove_path(self):
        k_nonzero = torch.nonzero(self.f).squeeze()
        self.path = torch.index_select(self.path, 1, k_nonzero)
        self.cost = torch.index_select(self.cost, 0, k_nonzero)
        self.f = torch.index_select(self.f, 0, k_nonzero)
        self.indicator = torch.index_select(self.indicator, 0, k_nonzero)
        M = torch.eq(self.indicator.unsqueeze(0), torch.arange(self.demand_number).unsqueeze(1)).double()
        self.M = M
          
    def projection_eu(self):
        r = self.r
        
        flow = cp.Variable(len(self.f))
        M = self.M
        b = cp.Parameter(len(self.f))
        q = self.q
        constraints = [flow >= 0, M @ flow == q]
        if self.Ma is not None:
            self.Ma = torch.zeros_like(self.f)
            self.Ma[0] = 1
            constraints = [flow >= 0, M @ flow == q, self.Ma @ flow == self.qa]
        objective = cp.Minimize(cp.pnorm(flow - b, p=2))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        b_tch = self.f - r * self.cost
        cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[flow])
        solution,  = cvxpylayer(b_tch)
        solution = solution.clamp(0)
        return solution
    
    def projection_kl(self):
        r = self.eta
        q = self.q
        b = self.f * torch.exp(-r * self.cost)
        M = self.M
        solution = b / sum(M * (M @ b).unsqueeze(1)) * sum(M * q.unsqueeze(1))
        return solution
    
    def projection_kld(self, o, d):
        k = self.od_serial[o, d]
        pk = self.indicator == k
        pn = pk.sum()
        if pn == 1:
            return
        r = self.r
        q = self.q[k]
        b = self.f[pk] * torch.exp(-r * self.cost[pk])
        solution = b / sum(b) * q
        self.f[pk] = solution
    
    def newton_move(self, o, d):
        k = self.od_serial[o, d]
        pk = self.indicator == k
        pn = pk.sum()
        if pn == 1:
            return
        
        paths = self.path[:, pk]
        flow = self.f[pk]
        x_prev = torch.mm(paths.double(), flow.unsqueeze(1)).squeeze()
        c1 = self.cost[pk]
        cm, lm = torch.min(c1, 0)
        lo = torch.arange(pn)
        lo = torch.cat((lo[:lm], lo[lm + 1:]))
        diffc = c1[lo] - cm
        same = paths[:, lm].unsqueeze(1) * paths[:, lo]
        a = (paths[:, lm].unsqueeze(1) ^ same) ^ (paths[:, lo] ^ same)
        c2 = self.F_x(self.x_global)
        flow[lo] -= 0.95 * diffc / torch.mm(a.t().double(), c2.unsqueeze(1)).squeeze()
        flow = flow.clamp(0)
        flow[lm] = self.q[k] - flow[lo].sum()
        x_new = torch.mm(self.path[:, pk].double(), flow.unsqueeze(1)).squeeze()
        self.f[pk] = flow
        self.x = self.x - x_prev + x_new
        
    def mirror(self):
        r = self.r
        
        flow = cp.Variable(len(self.f))
        M = self.M
        b = cp.Parameter(len(self.f))
        q = self.q
        constraints = [flow >= 0, M @ flow == q.detach()]
        objective = cp.Minimize(cp.pnorm(flow - b, p=2))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        b_tch = self.f - r * self.f.grad
        cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[flow])
        solution,  = cvxpylayer(b_tch)
        solution = solution.clamp(0)
        return solution


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
        with torch.enable_grad():
            m = self.edge_number
            x = torch.zeros(self.class_number, m, dtype=torch.double)
            for u in range(len(self.users)):
                fe = self.users[u].feasible_edge
                x[u, fe] = self.users[u].x
            x_sum = sum(x)
            self.x_sum = x_sum
            self.x = x
        
    def update_edge_cost(self):
        with torch.enable_grad():
            for u in self.active_users:
                fe = self.users[u].feasible_edge
                self.users[u].time = self.users[u].F(self.x[:, fe])
                
    def update_path_cost(self):
        with torch.enable_grad():
            for u in self.active_users:
                self.users[u].cost = self.users[u].path.t().double() @ self.users[u].time
                
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
                        
    def user_equilibrium(self, active_users, epsilon=1e-3, delta=0, display=False, number=0):
        self.active_users = active_users
        self.r = 0.5
        self.eta = 0.5
        for u in self.active_users:
            self.users[u].r = self.r
            self.users[u].eta = self.eta
        self.gap = list()
        m = self.edge_number
        with torch.no_grad():
            self.x_sum = torch.zeros(m, dtype=torch.double)
            self.x = torch.zeros(self.class_number, m, dtype=torch.double)
            self.update_edge_cost()     
            for u in self.active_users:
                self.users[u].initialize_path()
            self.update()
        
        iter_number = 0
        gap_old = math.inf

        with torch.no_grad():
            while iter_number < 1000:
                g = torch.zeros(len(self.active_users))
                rg = torch.zeros(len(self.active_users))
                for u in self.active_users:
                    y = torch.zeros_like(self.users[u].x)
                    for o in self.users[u].source:
                        y += self.users[u].update_path(o)
                    d = y - self.users[u].x
                    fe = self.users[u].feasible_edge
                    ub = self.users[u].potential(self.x[:, fe])
                    g[u] = -torch.dot(d, self.users[u].time)
                    rg[u] = g[u] / ub
                    self.users[u].x_global = self.x[:, fe]
                if display:
                    print('iter:', iter_number, 'T:', torch.dot(self.x_sum, Time(self.x_sum)), 'relitive gap:', rg)     
                
                self.gap.append(g)
                g = sum(g) / len(g)
                rg = sum(rg) / len(rg)
    
                if rg > delta and g / gap_old > 0.999:
                    self.r = max(0.8 * self.r, 1e-5)
                    for u in range(len(self.users)):
                        self.users[u].r = self.r
                gap_old = g
                if rg < epsilon:
                    break
              
                for u in self.active_users:
                    if g > delta:
                        self.users[u].f = self.users[u].projection_eu()
                        self.users[u].x = self.users[u].path.double() @ self.users[u].f
                    else:
                        for o in self.users[u].source:
                            sink = self.users[u].od[int(o)]
                            for d in sink:
                                self.users[u].newton_move(o, d)
                        
                self.update()
                iter_number += 1
                
                
        for u in self.active_users:
            self.users[u].remove_path()
            
        '''the projection phase'''
        if number > 0:
            with torch.no_grad():
                self.adjust_r()     
        with torch.enable_grad():
            projection_number = 0
            while projection_number < number:                
                for u in active_users:
                    self.users[u].f = self.users[u].projection_kl()
                    self.users[u].x = self.users[u].path.double() @ self.users[u].f
                                     
                self.update_edge_flow()
                self.update_edge_cost()
                self.update_path_cost()                    
                projection_number += 1
            
        return self.x_sum


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
dF = lambda x: 0.15 * 4 * (sum(x) / cap) ** 4 / cap
intF = lambda x: tfree * (sum(x) + 0.15 * cap / 5 * (sum(x) / cap) ** 5)

user = User(network.node_number)
user.set_demand(a_number, b_number, demand)
user.set_edge(s_number, t_number, F, intF=intF, dF=dF)


network.set_users([user])
active_users = [0]
network.user_equilibrium(active_users, epsilon=0.001, delta=100, number=0, display=True)
T = torch.dot(network.x_sum, Time(network.x_sum))
print(T)
