# pyTri
This program provides a python interface to Triangle [https://www.cs.cmu.edu/~quake/triangle.html], to generate triangular meshes for geophysical domains based on the bathymetry for use in Hydrodynamic models such as Swash, Swan and Thetis.

Usage: pyTri_m.py [OPTIONS] 

  This program provides a python interface to Triangle
  [https://www.cs.cmu.edu/~quake/triangle.html], to generate triangular
  meshes for geophysical domains based on the bathymetry for use in
  Hydrodynamic models such as Swash, Swan and Thetis.

  python pyTri_m.py --file_name --bathy_file_name --num_iterrations[5]
  --elem_start_size[10000] --elem_end_size[100] --init_ratio[0.80]
  --final_ratio[2.50] 
  
Options: 
  --file_name [TEXT] :          The name of the .poly file. 
  
  --bathy_file_name [TEXT] :     The name of the bathymetry file, this should include a header as the first line will be skipped!. The file format should be : | bathy,x,y | 
                              
                              
  --num_iterrations [INTEGER] : Number of iterations to increase the mesh resolution (Default:5) 
                             
  --elem_start_size [INTEGER] : Size of Triangular Element at initial iteration (m) (Default:10000) 
                             
  --elem_end_size [INTEGER] :   Size of Triangular Element at final iteration (m) (Default:100) 
                             
  --init_ratio [FLOAT] :        Initial Alpha for setting the mesh size (Default:0.80). The mesh size of the initial iteration will be set using the ratio between this value and the depth
                             
  --final_ratio [FLOAT] :       Final Alpha for setting the mesh size (Default:2.5). The mesh size of the final iteration will be set using the ratio between this value and the depth 
                             
  --help
