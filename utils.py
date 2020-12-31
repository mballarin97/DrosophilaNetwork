# Network analysis
import networkx as nx
from scipy import sparse
from fast_pagerank import pagerank_power
# Data handling
import pandas as pd
# Visualization
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig
import cairocffi as cairo
# Miscellaneous
import time
import os
import numpy as np
from tqdm import tqdm


def read_datasets(path='exported-traced-adjacencies'):
    """
        Read the datasets 'traced_neurons.csv', 'traced-total-connections.csv' and 
        'traced-roi-connections.csv'.
        Returns: the three datasets neurons, synapses, roi as pandas dataframes.
    """
    neurons_dataframe = pd.read_csv(os.path.join(path, 'traced-neurons.csv'))
    synapses_dataframe = pd.read_csv(os.path.join(path, 'traced-total-connections.csv'))
    roi_dataframe = pd.read_csv(os.path.join(path, 'traced-roi-connections.csv'))
    return neurons_dataframe, synapses_dataframe, roi_dataframe

