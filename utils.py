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


def read_datasets(path='exported-traced-adjacencies', prepare=True):
    """
        Read the datasets 'traced_neurons.csv', 'traced-total-connections.csv' and 
        'traced-roi-connections.csv'.
        Param path: path string where the files are stored
        Param prepare: if True, directly prepares the dataset in final form
        Returns: the three datasets neurons, synapses, roi as pandas dataframes.
    """
    neurons_dataframe = pd.read_csv(os.path.join(path, 'traced-neurons.csv'))
    synapses_dataframe = pd.read_csv(os.path.join(path, 'traced-total-connections.csv'))
    roi_dataframe = pd.read_csv(os.path.join(path, 'traced-roi-connections.csv'))
    if prepare:
        # zip synapese
        synapses_dataframe['synaps'] = list(zip(synapses_dataframe.bodyId_pre, synapses_dataframe.bodyId_post))
        roi_dataframe['synaps'] = list(zip(roi_dataframe.bodyId_pre, roi_dataframe.bodyId_post))
        # clean roi and reformat dataframe
        roi = clean_roi(roi_dataframe.to_numpy())
        roi_dataframe = pd.DataFrame(roi, columns=list(roi_dataframe.columns))

    
    return neurons_dataframe, synapses_dataframe, roi_dataframe


def clean_roi(roi):
    indexes_to_delete = []
    out_index = 0
    length = len(roi)
    while out_index < length:
        j = 1
        while(out_index+j < length and roi[out_index][4] == roi[out_index + j][4]):
            indexes_to_delete.append(out_index+j)
            j = j + 1
        out_index = out_index + j

    roi = np.delete(roi, obj=indexes_to_delete, axis=0)
    return roi