import numpy as np
import os 
import suite2p
import bottleneck as bn
import seaborn as sns

## DOC: https://suite2p.readthedocs.io/en/latest/settings.html
## DOC: https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_tiffs_or_batch.ipynb


#%% wrapper tho load data more convieniently
def loadS2p(pathfile , typeF = 'F.npy'):
    data = np.load(  os.path.join(pathfile, typeF) , allow_pickle=True  )
    return data

def loadOps(pathfile):
    ops = np.load(  os.path.join(pathfile, "ops.npy") ,allow_pickle=True )
    return ops.item()

def loadAllS2p(pathfile):
    F = loadS2p(pathfile , 'F.npy')
    Fneu = loadS2p(pathfile , 'Fneu.npy')
    spks = loadS2p(pathfile , 'spks.npy')
    stat = loadS2p(pathfile , 'stat.npy')
    ops = loadOps(pathfile)
    iscell = loadS2p(pathfile , 'iscell.npy')

    return F,Fneu,spks, stat, ops, iscell

#%% preprocessing function utils
def make_ops():
    ops = suite2p.default_ops()
    ops['diameter']= 10
    ops['tau'] = 0.7
    ops['fs']= 50
    return ops

def run_s2p(pathfile):
    ops = make_ops()
    ops['data_path'] = [str(pathfile)]
    suite2p.run_s2p(ops)


#%% Miscellanious helpers
    
def filterCell(iscell,data):
    # could be directly implemented when we load the data
    # I let is here for the moment

    if len(data.shape) ==1:
        return data[iscell[:,0]==1]
    else:
        return data[iscell[:,0]==1,:]
#%% Plotting function 

# %%
