from configparser import Interpolation
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

from pprint import pprint

ctr =np.array( [(1 , 1), (1 + 0.671710, 1 + 7.143724), (8 - 1.798965, 12.152957),  (8, 12.152957)])
x=ctr[:,0]
y=ctr[:,1]

# uncomment both lines for a closed curve
#x=np.append(x,[x[0]])  
#y=np.append(y,[y[0]])

l=len(x)  

t=np.linspace(0,1,l-2,endpoint=True)
t=np.append([0,0,0],t)
t=np.append(t,[1,1,1])

'''
t=np.linspace(0,1,l-2,endpoint=True)
t=np.append([0,0],t)
t=np.append(t,[1,1])
'''
# interpolation=[t,[x,y],2]
tck = [t, [x,y], 3]
# u3=np.linspace(1,8,8,endpoint=True)

# pprint (interpolation)
# interpolation, u = interpolate.splrep(x, y, k=2, s=0)
# u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)
frame = 2
normalized_frame = ( frame - 1 )/ (x.max() - x.min())
# normalized_frame = 0.33
pprint (normalized_frame)
u3 = np.array( [normalized_frame] )

'''
1 = 0
2 = 0.231
3 = 0.371
4 = 0.491
5 = 0.606
6 = 0.722
7 = 0.847
8 = 1
'''

def find_value(array, value):
    frames = array[0]
    n = [abs(i-value) for i in frames]
    idx = n.index(min(n))
    return array[1][idx]

u3=np.linspace(0,1, int(x.max() - x.min())*100, endpoint=True)
out = interpolate.splev(u3, tck)

value = find_value(out, 3)
pprint (value)

plt.plot(x[0:2],y[0:2],'k--',label='tangents',marker='o',markerfacecolor='red')
plt.plot(x[2:4],y[2:4],'k--',label='tangents',marker='o',markerfacecolor='red')

# plt.plot(x,y,'ro',label='Control points only')
plt.plot(out[0], out[1],'b',linewidth=2.0,label='B-spline curve')
plt.legend(loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('Cubic B-spline curve evaluation')
plt.show()