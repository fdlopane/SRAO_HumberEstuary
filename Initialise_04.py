# -*- coding: utf-8 -*-
"""
Initialise
Initialisation module

Functions:
    + Generate_Availability 
        @ It opens specific constraints rasters.
        @ Then, by running through them, identifies sites which are developable.
		@ It creates a new .tif file in which the available cells have a value
		@ of 1 and not available cells have a value of 0.
	+ Generate_Lookup
		@ It opens a specific availability raster.
		@ Then, by running through it, identifies sites which are developable.
        @ These are saved into a lookup list, which is saved as txt and returned.
	+ Generate_WarehousePlan
		@ Function creates a Warehouse_Plan based the availability Lookup.
		@ It generates a solution with a random number of warehouses
		@ between 1 and Warehouses_Max
	+ Generate_Proposed_Sites
		@ Function that takes a Warehouse plan as an input and returns
		@ a list of the coordinates of the proposed sites
"""
# Import modules:

"""
# sys.path.append:
# works as it will search the anaconda directory (which is the default python installation)
# for any libraries you try to import, and when it doesn’t find them, it will then search the python27 directory
# as we have added this as a path to look in. The only downside to this is that is a version of a library is in
# the anaconda directory, it will use that, rather than the version you have coped over.
import sys
sys.path.append('C:/Python27/Lib/site-packages')
"""

import random as rndm
import numpy as np
import pandas as pd
import rasterIO
import time
import os.path
import sys
import fiona
import math
from osgeo import gdal
from shutil import copyfile
import scipy
# from scipy import misc
from pathlib import Path

# import Calc_fdist_values_preprocess_02 as network_analysis_preproess
import Constraints

# Borders	      = raster where: ¦1 = inside borders       ¦ 0 = outside borders
# Road_Buffer     = raster where: ¦1 = inside buffer        ¦ 0 = outside buffer
# Green_areas     = raster where: ¦1 = Green_areas          ¦ 0 = not Green_areas
# Surface_water   = raster where: ¦1 = surface water        ¦ 0 = land



def Generate_Availability(Data_Folder):
	# This function takes as inputs several constraints rasters and then generates
	# an availability raster and saves it as a Gtiff file.
	
	# Constraints rasters:
	border_file_name   = "Border.tif"
	buffer_file_name   = "1km_roads.tif"
	Green_areas_file_name = "Green_areas.tif"
	water_file_name    = "surface_water.tif"
	
	time_Gen_Aval = time.asctime()
	
	print "Generate Availability function starts at: " , time_Gen_Aval
	print
	print "Opening input raster files..."
	print
	
	# File to be created:
	available_file = "Available.tif"

	print "File pointers creation..."
	
	# Import the constraints rasters to identify sites for the lookup
	border_pointer    	 = rasterIO.opengdalraster(Data_Folder+border_file_name)
	buffer_pointer    	 = rasterIO.opengdalraster(Data_Folder+buffer_file_name)
	Green_areas_pointer	 = rasterIO.opengdalraster(Data_Folder+Green_areas_file_name)
	water_pointer   	 = rasterIO.opengdalraster(Data_Folder+water_file_name)

	print "File pointers created."
	print
	
	# Creation of rasters:
	print "raster creation..."
	
	border_raster      = rasterIO.readrasterband(border_pointer,1,0,False)
	print "border raster created"
	buffer_raster      = rasterIO.readrasterband(buffer_pointer,1,0,False)
	print "buffer raster created"
	Green_areas_raster    = rasterIO.readrasterband(Green_areas_pointer,1,0,False)
	print "Green_areas raster created"
	water_raster       = rasterIO.readrasterband(water_pointer,1,0,False)
	print "water raster created"
	print
	
	print "readrastermeta creation..."	
	border_driver, border_XSize, border_YSize, border_proj_wkt, border_geo_t_params = rasterIO.readrastermeta(border_pointer)
	print "readrastermeta created."
	print
	
	# Initialise the availability raster with all zeros
	# __raster_name__.shape returns a tuple with the X and Y dimensions of the raster
	# dtype = np.int it is optional, I assign the integer type to save space and keep it light
	print "availability raster initialisation"
	print
	availability_raster = np.zeros(border_raster.shape, dtype = np.int)
	
	time_avl_ras = time.asctime()
	print "Creation of availability raster, starts at: " , time_avl_ras
	
	for x in range(0,border_XSize):
		for y in range(0,border_YSize):
			# print "x = ", x, "y = ", y
			if border_raster[y,x] == 1:
				if buffer_raster[y,x] == 1:
					if Green_areas_raster[y,x] != 1:
							if water_raster[y,x] != 1:
								# overwrite the availability_raster (all zeros) putting ones in proper positions:
								availability_raster[y,x] = 1

	# use the rasterIO writerasterbands function to create a new GTiff file starting from the availability_raster just created
	# epsg code of british national grid: http://spatialreference.org/ref/epsg/osgb-1936-british-national-grid/			
	rasterIO.writerasterbands(Data_Folder+available_file, 'GTiff', border_XSize, border_YSize, border_geo_t_params, 27700, None, availability_raster)
	
	time_avl_ras_end = time.asctime()
	print "Availability raster created at: " , time_avl_ras_end
	print


def Generate_Lookup(Data_Folder, Results_Folder, shapefilefile):
	# Creates a list of all the available sites.
	
	# if centroids shapefile exists, use it, otherwise create it:
	if os.path.isfile(os.path.join(Data_Folder, 'Available_centroids.shp')):
		available_centroids_file = 'Available_centroids.shp'
	else:
		if os.path.isfile(os.path.join(Data_Folder, 'Available.tif')):
			print
			print "Available centroid shape file is missing. You must create it!"
			quit()
		else:
			Generate_Availability(Data_Folder)
			print
			print "Available centroid shape file is missing. You must create it!"
			quit()
		
	'''else:
		if os.path.isfile(os.path.join(Data_Folder, 'Available.tif')):
			network_analysis_preproess.Convert_raster_to_points(Data_Folder)
			available_centroids_file = 'Available_centroids.shp'
		else:
			Generate_Availability(Data_Folder)
			network_analysis_preproess.Convert_raster_to_points(Data_Folder)
			available_centroids_file = 'Available_centroids.shp'
	'''

	centroids = fiona.open(Data_Folder+shapefilefile)
	
	# Create a list of all the points:
	Lookup = []
	for p in centroids:
		x_p = p['properties']['X']
		y_p = p['properties']['Y']
		# x_p = p['properties']['POINT_X']
		# y_p = p['properties']['POINT_Y']
		
		x_p = int(math.trunc(x_p)) # convert to integer rounding down
		y_p = int(math.trunc(y_p))
		
		Lookup.append([(x_p),(y_p)])
	
	#save to a txt file in Results Folder so other modules can load it
	np.savetxt(os.path.join(Results_Folder, "lookup.txt"), Lookup, delimiter=',', newline='\n') 	
	
	print
	return Lookup


def Generate_Lookup_local(Data_Folder, Results_Folder):
	# file is the basic availabilty raster
	File = "Available.tif"
	
	# Import the Availability Raster to identify sites for the Lookup_local
	file_pointer        = rasterIO.opengdalraster(Data_Folder+File)  
	Availability_Raster = rasterIO.readrasterband(file_pointer,1) 
	driver, XSize, YSize, proj_wkt, geo_t_params = rasterIO.readrastermeta(file_pointer)
	
	Lookup_local = [] # Array to hold the location of available sites   
	
	# Investigate for all x and y combinations in the file
	for x in range(0,XSize):
		for y in range(0,YSize):
			# If the yx location yeilds a available site
			if Availability_Raster[y,x]==1:
				# format it and append it to the Lookup_local list
				yx = (y,x)
				Lookup_local.append(yx)
	
	# save to a txt file so other modules can load it
	np.savetxt(os.path.join(Results_Folder, "lookup_local.txt"), Lookup_local, delimiter=',', newline='\n')      

	# return the Lookup_local list
	return Lookup_local

def Generate_Lookup_RurSubUrb(Data_Folder, Results_Folder, shapefilefile):
	# Creates a list of all the available sites with indication of Rural/Suburban/Urban location
	
	# if centroids_RurSubUrb shapefile exists, use it, otherwise create it:
	if os.path.isfile(os.path.join(Data_Folder, 'Available_centroids_RurSubUrb.shp')):
		available_centroids_file = 'Available_centroids_RurSubUrb.shp'
	else:
		print "Available_centroids_RurSubUrb does not exist! Create it in ArcGIS!"
	
	centroids_RurSubUrb = fiona.open(Data_Folder+shapefilefile)
	
	# Create a list of all the points:
	Lookup_RurSubUrb = []
	
	for p in centroids_RurSubUrb:
		x_p = p['properties']['X']
		y_p = p['properties']['Y']
		# x_p = p['properties']['POINT_X']
		# y_p = p['properties']['POINT_Y']
		RSU = p['properties']['R1_S2_U3']
		
		x_p = int(math.trunc(x_p)) # convert to integer rounding down
		y_p = int(math.trunc(y_p))
		RSU = int(math.trunc(RSU))
		
		Lookup_RurSubUrb.append([(x_p),(y_p),(RSU)])
	
	#save to a txt file in Results Folder so other modules can load it
	np.savetxt(os.path.join(Results_Folder, "Lookup_RurSubUrb.txt"), Lookup_RurSubUrb, delimiter=',', newline='\n') 	
	
	print
	return Lookup_RurSubUrb
	

def Generate_DevPlan(Development_Plan, Data_Folder, External_Results_Folder):
     # Produce development plan with 
     
     file_pointer = rasterIO.opengdalraster(Data_Folder+'Empty_plan.tif')  
     DevPlan      = np.double(np.copy(rasterIO.readrasterband(file_pointer,1)))
     
     file_pointer = rasterIO.opengdalraster(Data_Folder+'border.tif')     
     Boundary     = np.double(np.copy(rasterIO.readrasterband(file_pointer,1)))
     
     # Upload the Lookup_local table from the generated file.
     Lookup_local = (np.loadtxt(External_Results_Folder+"lookup_local.txt",delimiter=",")).tolist()
     
     # for each site in the Development Plan (with the same length as the Lookup_local)
     for t in range(0, len(Lookup_local)-1):
         
         # find the sites yx location
         j, i = tuple(Lookup_local[t])
         
         # Add the proposed development to the development plan
         #print Development_Plan[j]
         DevPlan[int(j), int(i)] = Development_Plan[t]
         
     # multiplying it to try stop it being square in the raster
     return np.multiply(DevPlan, Boundary)


def Generate_WarehousePlan(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder):
	# Function creates a Warehouse_Plan based the availability Lookup
	# generates a solution with a random number of warehouses between 1 and Warehouses_Max
	
	# sys.stdout.write("Generation of warehouse plan...\r")
		
	# Handles preventing the code hanging
	check_sum   = 0 # stores the previous Agg_Wahr to indicate if its changed
	check_count = 0 # counts the number of iterations the Agg_Wahr remains unchanged

	Warehouse_Plan = [0]*No_Available # Stores proposed warehouses sites
	Agg_War        = 0				  # Aggregate number of warehouses assigned 
    
	# Upload the lookup table from the generated file.
	# Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Assign a random number of warehouses between 1 and Warehouses_Max:
	Number_of_warehouses = rndm.randint(Warehouses_Min, Warehouses_Max)
	
	# Ensure enough warehouses are assigned
	while Agg_War < Number_of_warehouses:
		# Select a random available site
		j = rndm.randint(0,No_Available-1)

		# Check if the site hasn't already been designated.
		if Warehouse_Plan[j] == 0:
			Warehouse_Plan[j] = 1
			Agg_War += 1

		# Prevents the code hanging:
		if check_sum == Agg_War:
			# if the Agg_War was the same on the last iteration the count is increased
			check_count += 1
		else:
			# if Agg_War is different reset count and take the new Agg_War 
			check_count = 0
			check_sum = Agg_War
            
		# If the iteration has gone through with no change return false   
		if check_count > 100000:
			print "Caught hanging in Generate_WarehousePlan"           
			return False
	# sys.stdout.write("Generation of warehouse plan Completed.\r")
	
	if sum(Warehouse_Plan) == 0:
		raise ValueError('Sum of warehouse plan = 0. No sites allocated in Generate_WarehousePlan function.')
	
	# print sum(Warehouse_Plan)
	# print "Generate_WarehousePlan"
	
	return Warehouse_Plan


def Generate_WarehousePlan_Cluster_Ranking(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist):
	# Function creates a Warehouse_Plan based the availability Lookup length
	# generates a solution with a random number of warehouses between 1 and Warehouses_Max
	
	# Upload the lookup table from the generated file.
	Lookup_list = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Transform the lookup list into a pandas DataFrame
	Lookup = pd.DataFrame(Lookup_list, columns=['X','Y'])
	
	# Handles preventing the code hanging
	check_sum   = 0 # stores the previous Agg_Wahr to indicate if its changed
	check_count = 0 # counts the number of iterations the Agg_Wahr remains unchanged

	Warehouse_Plan = [0]*No_Available # Stores proposed warehouses sites
	Agg_War        = 0				  # Aggregate number of warehouses assigned 
    
	# Assign a random number of warehouses between 1 and Warehouses_Max:
	Number_of_warehouses = rndm.randint(Warehouses_Min, Warehouses_Max)
	
	probability = rndm.random()
	
	if probability > 0.8 :
		# Ensure enough warehouses are assigned
		while Agg_War < Number_of_warehouses:
			# Select a random available site
			j = rndm.randint(0,No_Available-1)

			# Check if the site hasn't already been designated and that it is far enough form already assigned sites.
			if Warehouse_Plan[j] == 0:
				Warehouse_Plan[j] = 1
				# Ensure that there is enough space between the proposed site and the already assigned sites.
				if Constraints.Check_Distance(Warehouse_Plan, Lookup_list, Min_Warh_dist) == True:
					# assign the value of 1 to the chosen site:
					Agg_War += 1
				else:
					Warehouse_Plan[j] = 0

				# Prevents the code hanging:
				if check_sum == Agg_War:
					# if the Agg_War was the same on the last iteration the count is increased
					check_count += 1
				else:
					# if Agg_War is different reset count and take the new Agg_War 
					check_count = 0
					check_sum = Agg_War
					
				# If the iteration has gone through with no change return false   
				if check_count > 100000:
					print "Caught hanging in Generate_WarehousePlan"           
					return False
	else:
		for n in range(1,Number_of_warehouses+1):
			# Open ranked cells file (ranked on the basis of proximity to assets clusters centroids)
			file = Results_Folder + 'Ranked_cells/' + 'CL' + str(Number_of_warehouses) + 'cl' + str(n)
			
			# Create a pandas df from the .csv file
			df = pd.read_csv(file, names=['X_Cell','Y_Cell','X_CL_centroid','Y_CL_centroid','Dist'])
			
			# Keep just the first two columns containing the coordinates of the cells
			df = df.loc[:,['X_Cell','Y_Cell']]
			
			# Select a random available site
			k = rndm.randint(0,len(df.index)-1)
			
			# Select the k-th row of the df
			site = df.loc[k,:]
			site_x = site[0]
			site_y = site[1]
			
			# Find index of Lookup df row that is equal to 'site'
			LL = Lookup[[all([X, Y]) for X, Y in zip(Lookup.X == site_x, Lookup.Y == site_y)]].index.tolist()
			j = LL[0] # transform the list into a single element

			# Assign the cell to the warehouse plan.
			Warehouse_Plan[j] = 1
			

	# sys.stdout.write("Generation of warehouse plan Completed.\r")
	
	if sum(Warehouse_Plan) == 0:
		raise ValueError('Sum of warehouse plan = 0. No sites allocated in Generate_WarehousePlan function.')
	
	# print sum(Warehouse_Plan)
	# print "Generate_WarehousePlan"
	
	return Warehouse_Plan


def Generate_WarehousePlan_check_distance(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist):
    # Function creates a Warehouse_Plan based the availability Lookup and a distance constraint.
	
	# sys.stdout.write("Generation of warehouse plan...\r")
	
	# Import the Availability Raster
	available_File      = "Available.tif"
	file_pointer        = rasterIO.opengdalraster(Data_Folder+available_File)  
	Availability_Raster = rasterIO.readrasterband(file_pointer,1) 
	driver, XSize, YSize, proj_wkt, geo_t_params = rasterIO.readrastermeta(file_pointer)
	
    # Handles preventing the code hanging
	check_sum   = 0 # stores the previous Agg_Wahr to indicate if its changed
	check_count = 0 # counts the number of iterations the Agg_Wahr remains unchanged

	Warehouse_Plan = [0]*No_Available # Stores proposed warehouses sites
	Agg_War        = 0				  # Aggregate number of warehouses assigned 
    
    # Upload the lookup table from the generated file.
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Assign a random number of warehouses between 1 and Warehouses_Max:
	Number_of_warehouses = rndm.randint(Warehouses_Min, Warehouses_Max)
	
    # Ensure enough warehouses are assigned
	while Agg_War < Number_of_warehouses:
        # Select a random available site
		j = rndm.randint(0,No_Available-1)

        # Extract the ji location of the site
		ji = tuple(Lookup[j])   
        
        # Check if the site hasn't already been designated and that it is far enough form already assigned sites.
		if Warehouse_Plan[j] == 0:
			Warehouse_Plan[j] = 1
			# Ensure that there is enough space between the proposed site and the already assigned sites.
			if Constraints.Check_Distance(Warehouse_Plan, Lookup, Min_Warh_dist) == True:
                # assign the value of 1 to the chosen site:
				Agg_War += 1
			else:
				Warehouse_Plan[j] = 0
			
        # Prevents the code hanging:
		if check_sum == Agg_War:
            # if the Agg_War was the same on the last iteration the count is increased
			check_count += 1
		else:
            # if Agg_War is different reset count and take the new Agg_War 
			check_count = 0
			check_sum = Agg_War
            
        # If the iteration has gone through with no change return false   
		if check_count > 100000:
			print "Caught hanging in Generate_WarehousePlan_check_distance"           
			return False
	
	if sum(Warehouse_Plan) == 0:
		raise ValueError('Sum of warehouse plan = 0. No sites allocated in Generate_WarehousePlan_check_distance function.')
	
	# print "Generation of warehouse plan completed"
	# print
	return Warehouse_Plan


def Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup):
	# Returns a list of the coordinates of the proposed sites.
	# Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup

	# print sum(Warehouse_Plan)
	# print "Generate_Proposed_Sites"

	# if sum(Warehouse_Plan) == 0:
		# print sum(Warehouse_Plan)
		# print "Warehouse_Plan = ", Warehouse_Plan
		# raise ValueError('Sum of warehouse plan = 0. When called in Generate_Proposed_Sites function. (Initialise)')
	
	Proposed_Sites_List = []
	# for each site in the Warehouse Plan (with the same length as the lookup)
	for j in range(0, len(Lookup)-1):
		if Warehouse_Plan[j] == 1:
			# find the sites yx location
			ji =  Lookup[j]
			ji[0] = int(math.trunc(ji[0]))
			ji[1] = int(math.trunc(ji[1]))
			ji = tuple(ji)
			Proposed_Sites_List.append(ji)
	# if len(Proposed_Sites_List) == 0:
		# raise ValueError('Length of Proposed sites list = 0 Length of Warehouse plan = ', sum(Warehouse_Plan))
		
	return Proposed_Sites_List