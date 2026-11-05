"""
Generation of coordinatates for the different geometries
"""
import numpy as np

x = []
y = []
r = []
i=0
for i in range(5):
    x= np.append(x,np.random.uniform(low=0.15,high=0.5))
    y= np.append(y,np.random.uniform(low=0.1,high=0.3))
    r= np.append(r,np.random.uniform(low=0.02,high=0.08))

print(x,y,r)