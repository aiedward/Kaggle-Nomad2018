
# coding: utf-8

# In[29]:


import glob, os, sys
import pandas as pd
import numpy as np


# In[43]:


def mean_submit(input_d, output_f):
    os.chdir(input_d)       
    files = (os.path.join(input_d, f) for f in os.listdir())
    dfs = (pd.read_csv(f) for f in files)
    mats = (df.as_matrix() for df in dfs)
    sub_stack = np.dstack(tuple(mats))
    reduce_mean = np.mean(sub_stack, axis=2)
    submission = pd.DataFrame(reduce_mean, columns=['id', 'formation_energy_ev_natom', "bandgap_energy_ev"])
    submission = submission.drop('id', 1)
    submission.insert(0, 'id', range(1, 601))
    submission.to_csv(output_f, index=False)
    
input_directory = sys.argv[1]
output_file = sys.argv[2]          
            
mean_submit(input_directory, output_file)

