######################################################################################    
###################                    pyTri                       ###################
###################                                                ###################
###################  SHUAIB RASHEED  Email : zubba1989@gmail.com   ###################
###################                                                ###################
###################                                                ###################
######################################################################################       

'''
    This program provides a python interface to Triangle [https://www.cs.cmu.edu/~quake/triangle.html], 
    to generate triangular meshes for geophysical domains based on the bathymetry for use in Hydrodynamic models such as Swash, Swan and Thetis.

    python THETIS_T3.py --file_name --bathy_file_name --num_iterrations[5] --elem_start_size[10000] --elem_end_size[100] --init_ratio[0.80] --final_ratio[2.50]

'''                                         

import numpy as np
import subprocess
from subprocess import call
import scipy.interpolate
import click
import time
import sys
import matplotlib.pyplot as plt


@click.command()

@click.option('--file_name', prompt='Initial .poly file name', help='The name of the .poly file.', default='TEST')

@click.option('--bathy_file_name', prompt='Initial bathymetry file name', help='The name of the bathymetry file, this should include a header as the first line will be skipped!. The file format should be : | bathy,x,y |', default='bath.csv')

@click.option('--num_iterrations', prompt='Number of iterations', help='Number of iterations to increase the mesh resolution (Default:5)', default=5)

@click.option('--elem_start_size', prompt='Initial General Elemental Size :', help='Size of Triangular Element at initial iteration (m) (Default:10000)', default=10000)

@click.option('--elem_end_size', prompt='Final General Element Size :', help='Size of Triangular Element at final iteration (m) (Default:100)', default=100)

@click.option('--init_ratio', prompt='Initial depth ratio :', help='Initial Alpha for setting the mesh size (Default:0.80). The mesh size of the initial iteration will be set using the ratio between this value and the depth', default=0.80)

@click.option('--final_ratio', prompt='Final depth ratio :', help='Final Alpha for setting the mesh size (Default:2.5). The mesh size of the final iteration will be set using the ratio between this value and the depth', default=2.50)



def iter_mesh (file_name, bathy_file_name, num_iterrations, elem_start_size, elem_end_size, init_ratio, final_ratio):

    '''
    This program provides a python interface to Triangle [https://www.cs.cmu.edu/~quake/triangle.html], to generate triangular meshes for geophysical domains based on the bathymetry for use in Hydrodynamic models such as Swash, Swan and Thetis.

    python THETIS_T3.py --file_name --bathy_file_name --num_iterrations[5] --elem_start_size[10000] --elem_end_size[100] --init_ratio[0.80] --final_ratio[2.50]

    '''
    ##################

    #RUN SOME TESTS!

    print ('RUNNING SOME INITIAL TESTS!')

    try:
        bathy_values = np.loadtxt('{}'.format(bathy_file_name), skiprows=1, delimiter=',')
    except:
        print ('Bathymetry file {} is missing, or cannot be read. Please use the file extension as well in specifying the file. The file format should be : | bathy,x,y |'.format(bathy_file_name))
        sys.exit()

    try:
        rc = subprocess.call(["triangle"])
        assert rc == 0,"Triangle is not available!"
    except AssertionError:
        print ('Triangle is Not Installed! Please install Triangle https://www.cs.cmu.edu/~quake/triangle.html before proceeding!')
        raise
        sys.exit()

    try:
        assert elem_start_size>elem_end_size,"Element start size ideally larger than the elemantal end size"
        assert init_ratio<final_ratio, "Initial Alpha size should be ideally smaller than the end size"
    except AssertionError:
        print ('We have a slight issue! Please check if elemental sizes and Alpha values are ok!')
        raise
        sys.exit()


    print ('########### All Tests Passed ! ###########')

    ###################

    print ('READING INITIAL POLY FILE')

    subprocess.call('triangle -pcqL {}.poly'.format(file_name), shell=True)

    print ('RUNNING INTIAL ITERATION')

    subprocess.call('triangle -rpa2000L {}.1'.format(file_name), shell=True)

    print ('CREATING INTERPOLATOR')

    bathy_values = np.loadtxt('{}'.format(bathy_file_name), skiprows=1, delimiter=',')
    bathy = bathy_values[:,0]
    x = bathy_values[:,1]
    y = bathy_values[:,2]
    intp = scipy.interpolate.LinearNDInterpolator((x,y), bathy)

    print ('FINISHED CREATING INTERPOLATOR')

    end_iter = int(num_iterrations)

    arreas = np.linspace(elem_start_size, elem_end_size, num_iterrations)

    ALPHAS = np.linspace(init_ratio, final_ratio, num_iterrations)

    for z in range(2,end_iter,1):

        print ('CREATING ELE AND NODE FILES FOR ITERATION NUMBER {}'.format(z))

        ele = np.genfromtxt('{}.{}.ele'.format(file_name,z), skip_header=1)

        nod_e = np.genfromtxt('{}.{}.node'.format(file_name,z), skip_header=1)

        NUM_TRIANGLE = int(ele[-1][0])

        ALPHA = ALPHAS[z-2]

        print ('REFINING USING Alpha = h/A = {}'.format(ALPHA))

        with open('{}.{}.area'.format(file_name,z), 'w') as f:

            f.write('{}\n'.format(NUM_TRIANGLE))

            for i in range (1,int(ele[-1][0])+1):

                print ('IT NUM {}/{} EVALUATING TRIANGLE NUMBER {}/{}'.format(z,end_iter, i,NUM_TRIANGLE))

                v1, v2, v3 = int(ele[i-1][1]), int(ele[i-1][2]), int(ele[i-1][3])

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : VERTICES INDEX OF THE TRIANGLE ARE : {}, {}, {}'.format(z,end_iter, i,NUM_TRIANGLE, v1,v2,v3))

                n1x, n1y = nod_e[v1-1][1], nod_e[v1-1][2]

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : THE COORDINATES OF THE VERTICE INDEX {} is {},{}'.format(z,end_iter, i,NUM_TRIANGLE, v1,n1x ,n1y))

                n2x, n2y = nod_e[v2-1][1], nod_e[v2-1][2]

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : THE COORDINATES OF THE VERTICE INDEX {} is {},{}'.format(z,end_iter, i,NUM_TRIANGLE, v2,n2x ,n2y))

                n3x, n3y = nod_e[v3-1][1], nod_e[v3-1][2]

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : THE COORDINATES OF THE VERTICE INDEX {} is {},{}'.format(z,end_iter, i,NUM_TRIANGLE, v3,n3x ,n3y))  

                center_x, center_y = round((n1x + n2x + n3x) / 3, 2), round((n1y + n2y + n3y) / 3, 2)

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : THE COORDINATES OF CENTROID OF TRIANGLE is {},{}'.format(z,end_iter, i,NUM_TRIANGLE, center_x ,center_y)) 
                
                depth = intp((center_x,center_y))

                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : THE DEPTH AT CENTROID OF TRIANGLE is {}'.format(z,end_iter, i,NUM_TRIANGLE, depth))   
        
                ref = depth/ALPHA
                
                print ('IT NUM {}/{}, TRIANGLE NUMBER {}/{} : TRIANGLE WILL BE REFINED USING A RATIO OF {}'.format(z,end_iter, i,NUM_TRIANGLE, ref))
        
                #f.write(\n)

                f.write('{}  {}\n'.format(i, ref))

                print ('COMPLETED IT NUM {}/{} EVALUATING TRIANGLE NUMBER {}/{}'.format(z,end_iter, i,NUM_TRIANGLE))

        curr_area = arreas[z]

        subprocess.call('triangle -rpaa{}L {}.{}'.format(curr_area,file_name,z), shell=True)

        print ('COMPLETED REFINMENT OF ALL {} TRIANGLES FOR IT NUM {}/{} '.format(NUM_TRIANGLE,z,end_iter))

        print ('KILL THE SCRIPT TO TERMINATE AT THIS STAGE! OTHER WISE NEXT ITERATION WILL RESUME IN 10 SECONDS')

        time.sleep(10)

    print ('ALL ITERATIONS COMPLETED SUCCESSFULLY!')

    print ('PREPARING IMAGES OF MESHES IN PNG FORMAT')

    for k in range(1,end_iter+1,1):

        print ('PLOTTING MESH IMAGE OF ITERATION NUMBER {}'.format(k))

        ele = np.genfromtxt('{}.{}.ele'.format(file_name,k), skip_header=1)

        nod_e = np.genfromtxt('{}.{}.node'.format(file_name,k), skip_header=1)

        NUM_TRIANGLE = int(ele[-1][0])

        for w in range (1,NUM_TRIANGLE+1):

            v1, v2, v3 = int(ele[w-1][1]), int(ele[w-1][2]), int(ele[w-1][3])

            n1x, n1y = nod_e[v1-1][1], nod_e[v1-1][2]

            n2x, n2y = nod_e[v2-1][1], nod_e[v2-1][2]

            n3x, n3y = nod_e[v3-1][1], nod_e[v3-1][2]

            plt.plot([n1x,n2x,n3x,n1x], [n1y,n2y,n3y,n1y], '-k', linewidth=0.1)

            plt.gca().set_aspect('equal')


        RES = np.linspace(500, 5000, end_iter)[k-1]

        plt.savefig('ITERATION_{}.png'.format(k), bbox_inches='tight', dpi=RES)

        plt.clf()

    print ('COMPLETED SUCCESSFULLY')


if __name__ == '__main__':
    iter_mesh()


