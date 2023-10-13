import numpy as np
import random

def get_nodes(l, dictionary):
    '''
    A function that returns a tuple (n1,n2) containing
    the nodes linked by link l and saved in a dictionary
    with structure {(n1,n2) = l}

    Args:
        - l (int): the link index
        - dictionary (dict): the dictionary containing
        the couple
    
    Returns:
        - Tuple(int, int): nodes indexes
    '''
    
    values = list(dictionary.values())
    index = values.index(l)
    nodes = list(dictionary.keys())[index]
    nodes = nodes.replace("(", "")
    nodes = nodes.replace(")", "")
    
    return tuple(map(int, nodes.split(', ')))



def printSection(section_name: str):
    '''
    A standard printing formatting for section separation. 
    Automatically converts text to uppercase.

    Args:
        - section_name (str): the name of the section
    '''
    
    print("\n\n\n")
    print("####################### " + section_name + " ###########################")
    print("\n")



class Proxytree():
    
    '''
    A class to manage all the constants of the tree and its structures.
    Takes in several values:
    - depth: the tree depth
    - server_c: the servers capacity
    - link_c: the links capacity * 
    - idle_pc: idle powcons of nodes *
    - dyn_pc: dynamic powcons of nodes *
    - datar_avg: the average datarate of flows
    * multiplied by tree levels, changes every level
    '''

    def __init__(self, depth, server_c, link_c, idle_pc, dyn_pc, datar_avg):
        self.DEPTH = depth

        self.SERVER_C = server_c               # Server capacity
        self.LINK_C = link_c*self.DEPTH            # Link capacity
        self.LINK_C_DECREASE = 2
        self.IDLE_PC = idle_pc*self.DEPTH          # Idle power consumption
        self.IDLE_PC_DECREASE = 5        # Idle power consumption
        self.DYN_PC = dyn_pc*self.DEPTH            # Dynamic power consumption
        self.DYN_PC_DECREASE = 1         # Dynamic power consumption
        # self.REQ_AVG = 8                 # Average flow request           (LEGACY: using randint and server capacity)
        self.DATAR_AVG = datar_avg               # Average data rate per flow

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
        start_link_c = self.LINK_C
        # Switch links
        for lvl in range(self.DEPTH-1):
            for _ in range(2**(2*lvl+1)):
                self.link_capacity.append(start_link_c)
            start_link_c -= self.LINK_C_DECREASE
        # Server links
        for _ in range(2**self.DEPTH):
            self.link_capacity.append(start_link_c)
        

    def init_idle_dyn_powcons(self):
        start_idle_pc = self.IDLE_PC
        start_dyn_pc = self.DYN_PC
        for lvl in range(self.DEPTH+1):
            for _ in range(2**lvl):
                self.idle_powcons.append(start_idle_pc)
                self.dyn_powcons.append(start_dyn_pc)
            start_idle_pc -= self.IDLE_PC_DECREASE
            start_dyn_pc -= self.DYN_PC_DECREASE


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


class Proxymanager():

    '''
    A class to manage all program constant for  time multipliers, 
    savers, and lagrange multipliers.

    Args:
        - proxytree (Proxytree): the tree structure to compute lagrange 
        multipliers
        - save_cqm_dict (bool, optional, default=False): boolean 
        to save CQM results
        - load_cqm_dict (bool, optional, default=False): boolean 
        to load CQM results
        - time_mul_vm (int, optional, default=0): time multiplier 
        for VM BQM solver
        - time_mul_path (int, optional, default=0): time multiplier 
        for path BQM solver
        - lag_mul_vm (int, optional, default=0): multiplier to compute 
        lagrange multiplier for VM BQM
        - lag_mul_path (int, optional, default=0): multiplier to 
        compute lagrange multiplier for path BQM
    '''

    def __init__(self, proxytree: Proxytree, save_cqm_dict = False, 
                load_cqm_dict = False, time_mul_vm = 0, time_mul_path = 0, 
                lag_mul_vm = 0, lag_mul_path = 0):
                
        self.SAVE_DICT = save_cqm_dict
        self.LOAD_DICT = load_cqm_dict


        # BQM Time multiplier for VM assignment solver
        if time_mul_vm:
            self.VM_CUSTOM_TIME = True
        else:
            self.VM_CUSTOM_TIME = False
        self.VM_TIME_MULT = time_mul_vm 

        # BQM Time multiplier for Path planner solver
        if time_mul_path:
            self.PATH_CUSTOM_TIME = True 
        else:
            self.PATH_CUSTOM_TIME = False
        self.PATH_TIME_MULT = time_mul_path           
        

        # Lagrange multiplier for CQM to BQM conversion for
            # VM problem | calculated from server idle powcons
        if lag_mul_vm:
            self.VM_CUSTOM_LAGRANGE = True 
        else:
            self.VM_CUSTOM_LAGRANGE = False 
        self.VM_LAGRANGE_MUL = int(proxytree.idle_powcons[-1] * lag_mul_vm)

        # Lagrange multiplier for CQM to BQM conversion for
            # path problem | calculated from server idle powcons
        if lag_mul_path:
            self.PATH_CUSTOM_LAGRANGE = True 
        else:
            self.PATH_CUSTOM_LAGRANGE = False 
        self.PATH_LAGRANGE_MUL = int(proxytree.idle_powcons[0] * lag_mul_path)   # Lagrange multiplier for cqm -> bqm path problem conversion | calculated from root switch idle powcons
