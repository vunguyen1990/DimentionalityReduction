
# coding: utf-8

# In[4]:


# import 


# In[20]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
from time import time


# In[17]:

digits_df = pd.read_csv('train.csv')






# In[18]:

y = digits_df['label']
X = digits_df[1:]
n_samples, n_features = X.shape
n_neighbors = 30


# In[ ]:

# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle,
               "Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))

