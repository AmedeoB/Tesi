from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel, Binary, quicksum, cqm_to_bqm 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
import dwave.inspector
from dwave.preprocessing import roof_duality
from scipy.stats import norm
import numpy as np
from random import seed
from random import randint
from datetime import datetime
import re


# Init random seed
seed()

"""
MODEL VARIABLES DICTIONARY
    M           int         number of servers
    N           int         number of VM
    K           int         number of switches
    F           int         number of flows (server-server paths, = M/2)
    L           int         number of links (graph links)

    Cs          1D list     capacity of each server
    pi_idle     1D list     idle power consumption fo each node
    pi_dyn      1D list     maximum dynamic power of each node 
    adj_node    2D list     node's adjancy list
    C           1D list     capacity of each link
    src_dst     2D list     list of server communicating through a path

    si          1D list     server status, 1 on, 0 off
    swk         1D list     switch status, 1 on, 0 off
    v           2D list     VM status per server, 1 on, 0 off
    u_v         2D array    CPU utilization of each VM
    d           2D array    data rate of flow f on link l

    rho         dictionary  
    on          dictionary


"""