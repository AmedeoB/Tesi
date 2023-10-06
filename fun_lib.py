import numpy as np
import random

def get_nodes(l, dictionary):
    '''
    A function that returns a tuple (n1,n2) containing
    the nodes linked by link l and saved in a dictionary
    with structure {(n1,n2) = l}
    '''
    values = list(dictionary.values())
    index = values.index(l)
    nodes = list(dictionary.keys())[index]
    nodes = nodes.replace("(", "")
    nodes = nodes.replace(")", "")
    return tuple(map(int, nodes.split(', ')))



class Proxymanager():
    '''
    A class to manage all program constant for debug, time multipliers, 
    savers, and lagrange multipliers
    '''
    def __init__(self, IDLE_PC: int):
        self.DEBUG = True            # Debug boolean
        self.SAVE_DICT = False
        self.LOAD_DICT = False
        self.TIME_MULT1 = 1           # CQM Time multiplier 1 for BQM in VM problem
        self.TIME_MULT2 = 1           # CQM Time multiplier 2 for BQM in path problem
        self.LAGRANGE_MUL = IDLE_PC*10   # Lagrange multiplier for cqm -> bqm conversion



class Proxytree():
    '''
    A class to manage all the constants of the tree and its structures
    '''
    def __init__(self):
        self.DEPTH = 3

        self.SERVER_C = 10               # Server capacity
        self.LINK_C = 5*self.DEPTH            # Link capacity
        self.LINK_C_DECREASE = 2
        self.IDLE_PC = 10*self.DEPTH          # Idle power consumption
        self.IDLE_PC_DECREASE = 5        # Idle power consumption
        self.DYN_PC = 2*self.DEPTH            # Dynamic power consumption
        self.DYN_PC_DECREASE = 1         # Dynamic power consumption
        # self.REQ_AVG = 8                 # Average flow request           (LEGACY: using randint and server capacity)
        self.DATAR_AVG = 4               # Average data rate per flow

        self.SERVERS = pow(2, self.DEPTH)                                 # Server number
        self.SWITCHES = sum(pow(2,i) for i in range(self.DEPTH))          # Switch number
        self.VMS = self.SERVERS                                           # VM number per server
        self.FLOWS = self.VMS//2 if self.VMS%2==0 else self.VMS//2+1                # Flow number
        self.NODES = self.SERVERS + self.SWITCHES                              # Total Nodes
        self.LINKS = 0
        self.init_links()

        self.server_capacity = [self.SERVER_C for i in range(self.SERVERS)]           # Capacity of each server
        self.link_capacity = []
        self.idle_powcons = []           # Idle power consumption of each node
        self.dyn_powcons = []            # Dynamic power of each node
        self.init_link_capacity()
        self.init_idle_dyn_powcons()
        self.cpu_util = [random.randint(self.SERVER_C//2 , self.SERVER_C-1) for _ in range(self.VMS)] # CPU utilization of each VM
        self.data_rate = [random.randint(self.DATAR_AVG-1 , self.DATAR_AVG+1) for _ in range(self.FLOWS)]      # Data rate of flow f on link l 
        
        self.link_dict = {}
        self.adjancy_list = [[0 for j in range(self.NODES)] for i in range(self.NODES)] 
        self.init_link_dict_adj_list()

        self.src_dst = [[0 for j in range(2)] for i in range(self.FLOWS)]
        self.initi_src_dst()


    def init_links(self):
        for i in range(self.DEPTH-1):
            self.LINKS += 2**i * 2**(i+1)
        self.LINKS += 2*(2**(self.DEPTH-1))
    
    def init_link_capacity(self):
        # Switch links
        for lvl in range(self.DEPTH-1):
            for _ in range(2**(2*lvl+1)):
                self.link_capacity.append(self.LINK_C)
            self.LINK_C -= self.LINK_C_DECREASE
        # Server links
        for _ in range(2**self.DEPTH):
            self.link_capacity.append(self.LINK_C)
        
    def init_idle_dyn_powcons(self):
        for lvl in range(self.DEPTH+1):
            for _ in range(2**lvl):
                self.idle_powcons.append(self.IDLE_PC)
                self.dyn_powcons.append(self.DYN_PC)
            self.IDLE_PC -= self.IDLE_PC_DECREASE
            self.DYN_PC -= self.DYN_PC_DECREASE

    def init_link_dict_adj_list(self):
        link_counter = 0
        # Create all sw-sw links
        for lvl in range(self.DEPTH-1):
            if lvl == 0:
                for i in range(1,3):
                    self.adjancy_list[0][i] = 1
                    self.adjancy_list[i][0] = 1
                    self.link_dict[str((0,i))] = link_counter
                    self.link_dict[str((i,0))] = link_counter
                    link_counter += 1
            else:
                first_sw = 2**(lvl) - 1
                last_sw = first_sw * 2
                for father in range(first_sw, last_sw + 1):
                    first_son = 2**(lvl+1) - 1
                    last_son = first_son * 2
                    for son in range(first_son, last_son + 1):
                        self.adjancy_list[father][son] = 1
                        self.adjancy_list[son][father] = 1
                        self.link_dict[str((father,son))] = link_counter
                        self.link_dict[str((son,father))] = link_counter
                        link_counter += 1
        
        # Last layer first and last switch
        ll_firstsw = 2**(self.DEPTH-1) - 1
        ll_lastsw = ll_firstsw * 2
        
        # Create all sw-s links
        for father in range(ll_firstsw, ll_lastsw + 1):
            for i in range(2):
                son = father * 2 + 1 + i

                self.adjancy_list[father][son] = 1
                self.adjancy_list[son][father] = 1
                self.link_dict[str((father,son))] = link_counter
                self.link_dict[str((son,father))] = link_counter
                link_counter += 1

    def initi_src_dst(self):
        index_list = [i for i in range(self.VMS)]
        random.shuffle(index_list)
        for i in range(self.FLOWS):
            for j in range(2):
                self.src_dst[i][j] = index_list[i*2 + j]


