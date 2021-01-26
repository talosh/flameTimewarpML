from configparser import Interpolation
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

from pprint import pprint

ctr =np.array( [(1 , 1), (1.397513, 2.066667), (8, 12), (8 + 0.384456, 12 - 0.666667)])
x=ctr[:,0]
y=ctr[:,1]

# uncomment both lines for a closed curve
#x=np.append(x,[x[0]])  
#y=np.append(y,[y[0]])

l=len(x)  

'''
t=np.linspace(0,1,l-2,endpoint=True)
t=np.append([0,0,0],t)
t=np.append(t,[1,1,1])
'''

t=np.linspace(0,1,l-2,endpoint=True)
t=np.append([0,0],t)
t=np.append(t,[1,1])

# interpolation=[t,[x,y],2]
# u3=np.linspace(1,8,8,endpoint=True)
interpolation, u = interpolate.splprep([x,y], k=3,s=0)

# pprint (interpolation)
# interpolation, u = interpolate.splrep(x, y, k=2, s=0)
# u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)
u3=np.linspace(0,1,200,endpoint=True)
out = interpolate.splev(u3, interpolation)
# pprint (u3)
pprint (out)

plt.plot(x[0:2],y[0:2],'k--',label='tangents',marker='o',markerfacecolor='red')
plt.plot(x[2:4],y[2:4],'k--',label='tangents',marker='o',markerfacecolor='red')

# plt.plot(x,y,'ro',label='Control points only')
plt.plot(out[0],out[1],'b',linewidth=2.0,label='B-spline curve')
plt.legend(loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('Cubic B-spline curve evaluation')
plt.show()