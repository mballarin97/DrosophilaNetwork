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

# key is superset, value is list of roi
_roi_supersets = {
    'AL': ['AL(L)', 'AL(R)'],
    'CX': ['FB', 'EB', 'PB', 'NO', 'AB(L)', 'AB(R)'],
    'GNG': ['GNG'],
    'INP': ['CRE(L)', 'CRE(R)', 'SCL(L)', 'SCL(R)', 'IB', 'ICL(L)', 'ICL(R)', 'ATL(L)', 'ATL(R)', 'IB'],
    'LH': ['LH(R)'],
    'LX': ['BU(L)', 'BU(R)','LAL(L)', 'LAL(R)'],
    'MB': ['CA(R)','PED(R)',"a'L(L)", "a'L(R)", 'aL(L)', 'aL(R)', "b'L(L)", "b'L(R)", 'bL(L)', 'bL(R)', 'gL(L)', 'gL(R)'],
    'OL': ['AME(R)','ME(R)','LO(R)', 'LOP(R)'],
    'PENP': ['SAD','CAN(R)','FLA(R)'],
    'SNP': ['SIP(L)', 'SIP(R)', 'SLP(R)', 'SMP(L)', 'SMP(R)', 'SPS(L)', 'SPS(R)'],
    'VLNP': ['AOTU(R)','AVLP(R)','PVLP(R)','PLP(R)','WED(R)'],
    'VMNP': ['VES(L)', 'VES(R)','EPA(L)', 'EPA(R)','GOR(L)', 'GOR(R)','SPS(L)', 'SPS(R)','IPS(R)'],
    'NotPrimary': ['NotPrimary']
}



def read_datasets(path='exported-traced-adjacencies', prepare=True, roi_supersets=None):
    """
        Read the datasets 'traced_neurons.csv', 'traced-total-connections.csv' and 
        'traced-roi-connections.csv'.
        Param path: path string where the files are stored
        Param prepare: if True, directly prepares the dataset in final form
        Returns: the three datasets neurons, synapses, roi as pandas dataframes.
    """
    neurons_dataframe = pd.read_csv(os.path.join(path, 'traced-neurons.csv'))
    synapses_dataframe = pd.read_csv(os.path.join(path, 'traced-total-connections.csv'))
    if prepare:
        # zip synapese
        synapses_dataframe['synaps'] = list(zip(synapses_dataframe.bodyId_pre, synapses_dataframe.bodyId_post))

    if os.path.exists(os.path.join(path, 'treated_roi.csv')):
        roi_dataframe = pd.read_csv(os.path.join(path, 'treated_roi.csv'))
    else:
        roi_dataframe = pd.read_csv(os.path.join(path, 'traced-roi-connections.csv'))
        if prepare:
            # zip synapese
            roi_dataframe['synaps'] = list(zip(roi_dataframe.bodyId_pre, roi_dataframe.bodyId_post))
            # clean roi and reformat dataframe
            roi = _clean_roi(roi_dataframe.to_numpy())
            roi_dataframe = pd.DataFrame(roi, columns=list(roi_dataframe.columns))
            roi_dataframe['superset'] = _map_roi_to_superset(roi_dataframe['roi'].tolist(), roi_supersets)

    return neurons_dataframe, synapses_dataframe, roi_dataframe


def _clean_roi(roi):
    """
        Clean the roi dataset by taking the first roi (associated with highest weight).
        Param roi: numpy array containig the roi dataset
        Returns the cleaned roi array
    """
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
    
    
def _map_roi_to_superset(roi_list, roi_supersets=None):
    # if no mapping was provided, use the default one
    if not roi_supersets: roi_supersets = _roi_supersets
    def find_attr(roi):
        for key, value in roi_supersets.items():
            if roi in value:
                return key
    res = list(map(find_attr, roi_list))
    return res       