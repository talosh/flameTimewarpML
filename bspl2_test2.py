from configparser import Interpolation
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

from pprint import pprint

ctr =np.array( [(0 , 0), (0.1, 0), (0.9, 1),  (1, 1)])
x=ctr[:,0]
y=ctr[:,1]

'''
tck = interpolate.splrep(x, y, k=3, s=0)
xnew = np.array( [3] )
ynew = interpolate.splev(xnew, tck, der=0)
pprint (ynew)
'''

interp = interpolate.CubicSpline(x, y)

xnew=np.linspace(x.min(),x.max(), 100,endpoint=True)
# out = interpolate.splev(xnew, tck, der=0)

out = interp(xnew)

plt.plot(x[0:2],y[0:2],'k--',label='tangents',marker='o',markerfacecolor='red')
plt.plot(x[2:4],y[2:4],'k--',label='tangents',marker='o',markerfacecolor='red')

# plt.plot(x,y,'ro',label='Control points only')
plt.plot(xnew,out,'b',linewidth=2.0,label='B-spline curve')
plt.legend(loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('Cubic B-spline curve evaluation')
plt.show()