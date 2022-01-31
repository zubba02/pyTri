
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
import cmocean
import os
from PIL import Image
import glob


# In[2]:


F_N = 'STR_10_ALL'


# In[3]:


swan_out = sio.loadmat('{}.mat'.format(F_N))
sorted(swan_out.keys())


# In[4]:


val_2d = np.array(swan_out['Watlev_001909_999'])
#np.shape(val_2d)


# In[5]:


#plt.imshow(val_2d, interpolation='none')
#plt.show()


# In[6]:


'''methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
'''
'''plt.figure(figsize = (10,10))
#plt.imshow(val_2d, cmap=plt.get_cmap('jet'),aspect='auto', interpolation='none')
plt.imshow(val_2d, cmap='cmo.balance', aspect='auto', interpolation='quadric')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('Shuaib')
plt.show()'''


# In[7]:


'''ve = 0.05

dx, dy = 0.01,0.01

ls = LightSource(azdeg=180, altdeg=50)

plt.figure(figsize = (10,10))
plt.imshow(ls.hillshade(val_2d, vert_exag=ve, dx=dx, dy=dy), cmap='cmo.balance', interpolation='quadric', aspect='auto')
plt.gca().invert_yaxis()
plt.colorbar()'''


# In[8]:


'''os.mkdir('Watlev')

for i in sorted(swan_out.keys()):
    #print (i)
    if i[0] == 'W':
        val_2d = np.array(swan_out['{}'.format(i)])
        plt.figure(figsize = (10,10))
        #plt.imshow(val_2d, cmap=plt.get_cmap('jet'),aspect='auto', interpolation='none')
        plt.imshow(val_2d, cmap='cmo.balance', aspect='auto', interpolation='quadric', vmin=-1.0, vmax=1.0,)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Watlev')
        plt.savefig('Watlev/{}'.format(i))
        plt.show()
        
 
# Create the frames
frames = []
imgs = glob.glob("Watlev/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('Watlev/Watlev.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)'''


# In[15]:


'''os.mkdir('VMag')

for i in sorted(swan_out.keys()):
    #print (i)
    if i[0:4] == 'Vmag':
        val_2d = np.array(swan_out['{}'.format(i)])
        plt.figure(figsize = (10,10))
        #plt.imshow(val_2d, cmap=plt.get_cmap('jet'),aspect='auto', interpolation='none')
        plt.imshow(val_2d, cmap='cmo.speed', aspect='auto', interpolation='quadric', vmin=0.0, vmax=2.5,)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Vmag')
        plt.savefig('Vmag/{}'.format(i))
        plt.show()'''
        
 
# Create the frames
frames = []
imgs = glob.glob("Vmag/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('VMag/VMag.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=0)
 

