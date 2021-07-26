# -*- coding: utf-8 -*-
"""
Calc_fdist_values_network

Module intends to calculate the distances from all available sites within the 
study area and the targets (strategic infrastructure sites) producing a lookup table (fdist_lookup).
Unfortunatly NetworkX greatly increases the running time, therefore instead of calculating a performance
in fdist each iteration of the optimisation, we instead refer the development sites to this
lookup table (fdist_lookup)

csv file created, columns:
Xsite, Ysite, Xtarget, Ytarget, distance
"""

program_name = "Calc_fdist_values_network_500mres"

import os
os.system('cls')  # clears screen
import time

start_time = time.asctime()
print "Program: " , program_name
print "Starts at: " , start_time
print

# print "importing modules..."
import networkx as nx
import rasterIO
import numpy as np
import shapefile as shp
import fiona
import shapely.geometry
import csv
import sys
import math
# print "Modules imported."
# print

data_folder 	= "../Data/Hull_500m_resolution/"
results_folder	= "./Results_500m_resolution"

Road_Network_shapefile   = 'Road_Network.shp'
Available_centroids_shapefile = 'Available_centroids.shp'
Destinations_shapefile   = 'Strategic_infrastructure.shp'

print "Importing road network and target nodes..."
# Road network which forms the path
Road_Network = nx.read_shp(data_folder+Road_Network_shapefile)

# The road nodes point file which we are calculating the shortest path distance from
Available_centroids = nx.read_shp(data_folder+Available_centroids_shapefile)

# The target point file which we are calculating the shortest path distance to
Target_Nodes = nx.read_shp(data_folder+Destinations_shapefile)

# Extracting the dataset for potential sites to calculate fdist from each one
file_pointer = rasterIO.opengdalraster(data_folder+'Available.tif')    
Available    = rasterIO.readrasterband(file_pointer,1)

# Extracting the geotrans which is necessary for calculating the centroids
# of potential development sites
d,X,Y,p,geotrans = rasterIO.readrastermeta(file_pointer)
print "Road network and target nodes imported."
print

"""
"""
def Gen_List_Proposed_Sites(Available_Sites):
    # Create a list of all the sites which we want to calculate the 
    # shortest path fdist_values_Available.csv to. 
    Sites_to_Calc = []
    for x in range(0,X):
        for y in range(0,Y):
            siteyx = (y,x)
            if Available_Sites[siteyx] == 1:
                Sites_to_Calc.append(siteyx)
               
    return Sites_to_Calc

"""
"""	
def Conv_2_Coords(list_of_sites, geo_t_params):
	# Calculates the geographical reference point of a centroid for each 
	# possible development site. Inputs are y,x
	count = 0
	array = []
	site_nodes = []
	for site in list_of_sites:
		y = site[0]
		x = site[1]
		# coord = coord of raster corner + (cell_coord * cell_size) + (cell_size/2)
		x_coord = geo_t_params[0] + (x*geo_t_params[1]) + (geo_t_params[1]/2)
		y_coord = geo_t_params[3] - (y*geo_t_params[1]) + (geo_t_params[5]/2)
		#print y_coord, x_coord
		
		# Have to work in x and y I think
		node_coord=(x_coord, y_coord)
		
		site_nodes.append(node_coord)    
		#site_nodes.extend(node_coord)    
		
		a = [count, x_coord, y_coord, x, y]
		array.append(a)
		#array.extend(a)
		count += 1 
		
	# Write .csv file:
	#np.savetxt('P:/RLO/Python_Codes/Hull_Case_Study/Results/Available_Sites.csv', array, delimiter = ',')
	#print "Conv_2_Coords txt file saved"
	print
	return site_nodes

"""
"""
def write_shp_centroids(site_nodes):
	# This function creates a shapefile .shp pf points representing the centroids of
	# available cells.
	# - See http://pygis.blogspot.co.uk/2012/10/pyshp-attribute-types-and-point-files.html
	
	print "Writing Available_centroids shapefile..."
	import shapefile as shp
	
	#Set up shapefile writer and create empty fields
	w = shp.Writer(shp.POINT)
	w.autoBalance = 1 #ensures gemoetry and attributes match
	w.field('X','F',10,5)
	w.field('Y','F',10,5) #float - needed for coordinates
	
	#loop through the data and write the shapefile
	for node in site_nodes:
		w.point(node[0],node[1]) #write the geometry
		w.record(node[0],node[1]) #write the attributes

	#Save shapefile
	w.save(data_folder+Available_centroids_shapefile)
	
	print "Shapefile saved"
	
"""
"""	
def calc_closest(new_node, node_list):
    best_diff = 10000
    closest_node=[0,0]
    for comp_node in node_list.nodes():
        
        diff = (abs(comp_node[0]-new_node[0])+abs(comp_node[1]-new_node[1]))
        if abs(diff) < best_diff:
            best_diff = diff
            closest_node = comp_node
            
    return closest_node
    
"""
"""  
def Add_Nodes_To_Network(node_list, network):
    # Adds an edge between the node and the node calculated to be closest    
    for node in node_list:
        # Calculate the closest road node
        closest_node = calc_closest(node, network)
        network.add_node(node) #adds node to network
        network.add_edge(node, closest_node) #adds edge between nodes

"""
"""
def Add_Edges(g, node, closest_node):
    # Add node to the network then add an edge
    g.add_node(node)    
    g.add_edge(node, closest_node)
    return g

"""
"""
def Calc_Short_Dist_network(Available_centroids, Target_Nodes, Road_Network):
	print "Beginning Calculate Fitness"
	print
	
	Road_Network = Road_Network.to_undirected()
	print "Road network converted to undirected"
	
	# Add the Target_Nodes to the road network and create an edge between them and the closest road network node
	Add_Nodes_To_Network(Target_Nodes, Road_Network)
	
	print "Add_Nodes_To_Network (1/2 - target nodes): DONE"
	print
	print "Number of Available centroids = ", len(Available_centroids)
	print
	Add_Nodes_To_Network(Available_centroids, Road_Network)
	print "Add_Nodes_To_Network (2/2 - road nodes): DONE"
	print
	
	print "Calculating Shortest Distances for ", len(Available_centroids), " road nodes..."
	# Calculate the shortest distance from each site to a target node    
	Dist_dict = {}
	
	list_points_noPath = []
	
	site_count = 0
	for Dev_Site in Available_centroids:
		Dist_dict[Dev_Site] = {}
		for trgt in Target_Nodes:
			try:
				shrtst_dist = 36000 # 36000 minutes = 10 hours
				# dist = nx.shortest_path_length(Road_Network, Dev_Site, trgt, weight='Dist')
				dist = nx.shortest_path_length(Road_Network, Dev_Site, trgt, weight='FFlowTravT')
				if dist < shrtst_dist:
					shrtst_dist = dist
				
				Dist_dict[Dev_Site][trgt] = shrtst_dist
				
			except nx.NetworkXNoPath:
				print "Ne path pet. Node = ", Dev_Site
				# print "Node = ", Dev_Site, "Trgt = ", trgt
				list_points_noPath.append(Dev_Site)
				# print

		site_count += 1
		if site_count%1000 == 0:
			prog = int(round(site_count*100/len(Available_centroids)))
			sys.stdout.write("Progress: %d %% \r" %prog)
			# print ("iteration number %d of %d" %(site_count, len(Available_centroids)))
	
	# if len(Dist_dict)!= (len(Available_centroids) * len(Target_Nodes)): # len(dict) returns the number of the keys
	print
	print "#####################################"
	print "WARNING: Check lenght of final dictionary"
	print "#####################################"
	
	return Dist_dict, list_points_noPath

"""
"""
 
if __name__ == '__main__': 
    
	print "Generating Sites to Calculate"
	print
	# Create a list of all the sites which we want to calculate the shortest pathfdist_values_Available.csv to:
	Sites_to_Calculate = Gen_List_Proposed_Sites(Available)
	
	if os.path.isfile(os.path.join(data_folder, Available_centroids_shapefile)):
		print "Skip the generation of centroids shapefile because this file already exists."
	else:
		# Print centroids shapefile
		Dev_Nodes = Conv_2_Coords(Sites_to_Calculate, geotrans)
		write_shp_centroids(Dev_Nodes)
	
	Distances_dict, NoPath_Nodes_list = Calc_Short_Dist_network(Available_centroids, Target_Nodes, Road_Network)
	e_time = time.asctime()
	print "Dictionary saved in a variable at: " , e_time
	print
	
	# Print dist_dictionary in a csv file
	print "Writing dictionary in a .csv file..."
	with open('P:/RLO/Python_Codes/Hull_Case_Study/Results_500m_resolution/Dictionary_cells_targets', 'wb') as csv_file:
		writer = csv.writer(csv_file)
		wr_count = 0
		for key1 in Distances_dict:
			wr_count = wr_count + 1
			for key2, value in Distances_dict[key1].items():
				# writer.writerow([key1, key2, value])
				# writer.writerow([int(math.trunc(key1[0])),int(math.trunc(key1[1])), int(math.trunc(key2[0])), int(math.trunc(key2[1])), value])
				writer.writerow([int(round(key1[0])),int(round(key1[1])), int(round(key2[0])), int(round(key2[1])), value])
				if wr_count%1000==0:
					# prog = int(round(wr_count*100/len(Distances_dict)))
					# print ("Progress: %d %%" %prog)
					sys.stdout.write("Written row number %d of %d\r" %(wr_count, len(Distances_dict)*len(Sites_to_Calculate)))
	print
	
	# print txt file with nodes with no path
	np.savetxt(os.path.join(results_folder, "NoPath_Nodes.txt"), NoPath_Nodes_list, delimiter=',', newline='\n') 
	
	print "Ran" 
	
	end_time = time.asctime()
	print "Program terminates at: " , end_time