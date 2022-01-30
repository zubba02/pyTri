import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


data = np.genfromtxt('BATHY.csv', delimiter=',')

interpolator_mv = scipy.interpolate.NearestNDInterpolator((data[:,0], data[:,1]), data[:,2])

x_min,x_max = data[:,0].min(),data[:,0].max()
y_min,y_max = data[:,1].min(),data[:,1].max()

x_range = np.linspace(x_min,x_min,250)
y_range = np.linspace(y_min,y_max,250)

gridxy = np.mgrid[x_min:x_max:250j, y_min:y_max:250j].T
gridxy = np.reshape(gridxy, (250*250, 2))

x_coord = []
y_coord = []

for i,xy in enumerate(gridxy):
    x_coord.append(xy[0])
    y_coord.append(xy[1])

len_x = np.shape(x_coord)

coords = np.column_stack([x_coord, y_coord])

bathy = []
abS = []

for i,xy in enumerate (coords):
    print (i,len_x,interpolator_mv(xy[0], xy[1]))
    bathy.append(interpolator_mv(xy[0], xy[1]))
    abS.append(np.abs(interpolator_mv(xy[0], xy[1])))

all_data_mesh = np.column_stack([abS, x_coord, y_coord])

all_data = np.column_stack([x_coord, y_coord, bathy])
np.savetxt('clippedbathy_250_250.csv',bathy, fmt='%f')
np.savetxt('clippedbathy_with_coord_250_250.csv',all_data, fmt='%f', delimiter=',')

np.savetxt('clippedbathy_with_coord_250_250_mesh.csv',all_data_mesh, fmt='%f', delimiter=',')

