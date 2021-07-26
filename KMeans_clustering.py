#IMPORT MODULES
import os
os.system('cls')  # clears screen
import fiona
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Calc_fdist_values_network_500mres as DN
import geopandas as gpd
import shapely.geometry
from shapely.geometry import Point
import networkx as nx
import rasterIO
import csv
import math
import sys
import numpy as np
import shapefile as shp

#########################
#INPUT:
Warehouses_Max = 10

# DATA FOLDERS
data_folder    = "../Data/Hull_500m_resolution/"
results_folder = "./Results_500m_resolution/"

Road_Network_shapefile   = 'Road_Network.shp'
Road_Nodes_shapefile     = 'Road_nodes_ID.shp'
available_centroids_file = 'Available_centroids.shp'

# Shapefile containing the strategic infrastructure assets
Infr_shapefile = 'Strategic_infrastructure.shp'

average_speed = 30 # km/h from cell centroid to nearest road node
ranked_cells = 50 # Number of cells ranked in terms of proximity to clusters centroids
#########################

# Create a "fiona object" from the shapefile
points = fiona.open(data_folder+Infr_shapefile)
number_of_assets = len(points)

av_cells = fiona.open(data_folder+available_centroids_file)
number_of_av_cells = len(av_cells)


# Save in a list the coordinates of the points contained in the shapefile
Point_list = []
for p in points:
	x_p = p['geometry']['coordinates'][0]
	y_p = p['geometry']['coordinates'][1]
	Point_list.append([(x_p),(y_p)])

# Transform the list into a Pandas dataframe
point_df = pd.DataFrame(Point_list, columns=['X','Y'])

centroids_dict = {}

"""
"""
def Create_centroids_shp(Results_Folder, centroids, n_shp):
	# This function creates a Point Shapefile starting from coordinates tuples
	# 'centroids' is an array containing couples of coordinates
	
	# Transform the array into a Pandas dataframe
	df = pd.DataFrame(centroids, columns=['X','Y'])

	# print df

	# Create a shapefile folder:
	New_Shp_Folder = Results_Folder+'Clusters_Centroids_Shapefiles'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Shp_Folder): os.makedirs(New_Shp_Folder)
	
	# Create Shapefiles
	gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in df[['X', 'Y']].values.tolist()],crs={'init':'epsg:27700'})
	gdf.to_file(New_Shp_Folder + '/' + str(n_shp) +'_warhouses_centroid'+'.shp')

"""
"""
def Dist_RoadN_ClCentroids(Number_clusters, Results_Folder):
	# This function calculates the distance between road nodes and cluster centroid
	# and saves the result in a .csv file
	
	# Centroids shapefile:
	Destinations_shapefile = Results_Folder+'Clusters_Centroids_Shapefiles/'+str(Number_clusters)+'_warhouses_centroid.shp'
	
	# Road network which forms the path
	Road_Network = nx.read_shp(data_folder+Road_Network_shapefile)

	# The road nodes point file which we are calculating the shortest path distance from
	Road_Nodes = nx.read_shp(data_folder+Road_Nodes_shapefile)

	# The target point file which we are calculating the shortest path distance to
	Target_Nodes = nx.read_shp(Destinations_shapefile)

	# Extracting the dataset for potential sites to calculate fdist from each one
	file_pointer = rasterIO.opengdalraster(data_folder+'Available.tif')    
	Available    = rasterIO.readrasterband(file_pointer,1)

	# Extracting the geotrans which is necessary for calculating the centroids of potential development sites
	d,X,Y,p,geotrans = rasterIO.readrastermeta(file_pointer)
	
	# Create a list of all the sites which we want to calculate the shortest pathfdist_values_Available.csv to:
	Sites_to_Calculate = DN.Gen_List_Proposed_Sites(Available)
	
	if os.path.isfile(os.path.join(data_folder, available_centroids_file)):
		print "Skip the generation of centroids shapefile because this file already exists."
	else:
		# Print centroids shapefile
		Dev_Nodes = DN.Conv_2_Coords(Sites_to_Calculate, geotrans)
		DN.write_shp_centroids(Dev_Nodes)
	
	Distances_dict, NoPath_Nodes_list = DN.Calc_Short_Dist_network(Road_Nodes, Target_Nodes, Road_Network)
	
	# Save the dictionary in a .csv file:
	print "Writing dictionary in a .csv file..."
	with open(Results_Folder+'Dictionary_nodes_ClustersCentroids_'+str(Number_clusters), 'wb') as csv_file:
		writer = csv.writer(csv_file)
		wr_count = 0
		for key1 in Distances_dict:
			for key2, value in Distances_dict[key1].items():
				# writer.writerow([key1, key2, value])
				writer.writerow([int(math.trunc(key1[0])),int(math.trunc(key1[1])), int(math.trunc(key2[0])), int(math.trunc(key2[1])), value])
				wr_count = wr_count + 1
				if wr_count%1000==0:
					# prog = int(round(wr_count*100/len(Distances_dict)))
					# print ("Progress: %d %%" %prog)
					sys.stdout.write("Written row number %d \r" %wr_count)
	print
	
	# print txt file with nodes with no path
	np.savetxt(os.path.join(Results_Folder, "NoPath_Nodes_ClusterCentroids.txt"), NoPath_Nodes_list, delimiter=',', newline='\n') 	

"""
"""
def csv_to_df(points_nodes_csv, speed, nodes_centroids_csv):
	# Saving dictionaries (.csv) into pandas df
	# Also converts the distance into travel time in the preprocess dictionary.
	
	points_nodes_df = pd.read_csv(points_nodes_csv, names=['X_Point','Y_Point','X_Node','Y_Node','Dist_P_N'])

	# Convert distances into travel times>
	for d in range(len(points_nodes_df)):
		points_nodes_df.loc[d,'Dist_P_N'] = points_nodes_df.loc[d,'Dist_P_N']/(float(speed)/(3600.0/1000.0)) # conversion from km/h to m/s
		# prog = int(round(d*100/len(points_nodes_df)))
		# sys.stdout.write(" Progress: %d %% \r" %prog)
		# sys.stdout.write(" Iteration %d of %d\r" % (d, len(points_nodes_df)))
	
	# Saving dictionary nodes_targets into a variable (from .csv)
	nodes_centroids_df = pd.read_csv(nodes_centroids_csv, names=['X_Node','Y_Node','X_Centroid','Y_Centroid','Dist_N_C'])

	return points_nodes_df, nodes_centroids_df	

"""
"""	
def Rank_Cells(Number_clusters, Results_Folder,number_of_assets):
	# This function creates a dataframe combining the cells-nodes dict to the nodes-centroids dict,
	# then it sorts the available cells in terms of proximity to the centroids and saves the result in a .csv file.
	
	nodes_centroids_csv = Results_Folder+'Dictionary_nodes_ClustersCentroids_'+str(Number_clusters)
	preprocess_csv = "./Results_500m_resolution/Dictionary_AVpoints_RDnodes"
	
	# Create a new 'Ranked cells' folder:
	New_Shp_Folder = Results_Folder+'Ranked_Cells/'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Shp_Folder): os.makedirs(New_Shp_Folder)
		
	# Save the dictionaries (.csv) as a pandas data frames
	points_nodes_df, nodes_centroids_df = csv_to_df(preprocess_csv, average_speed, nodes_centroids_csv)
	
	# Join Points-Nodes df with Nodes-Centroids df:
	joined_dfs = points_nodes_df.merge(nodes_centroids_df, how='inner', on=['X_Node','Y_Node'], suffixes=('_PN', '_NT'))
	
	# Sum distances
	joined_dfs['Tot_dist'] = joined_dfs['Dist_P_N'] + joined_dfs['Dist_N_C']
	
	# sort data frame on targets' coordinates and dist value:
	joined_dfs.sort_values(['X_Centroid','Y_Centroid','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# keeps the first row of each centroid (which contains the closest point)
	#joined_dfs.drop_duplicates(subset=['X_Centroid','Y_Centroid'], keep='first', inplace=True)
	'''
	# keeps the first 50 rows of each centroid (which contains the closest 50 points)
	joined_dfs=joined_dfs.groupby(['X_Centroid','Y_Centroid']).head(ranked_cells)
	
	# Save data frame in the form: 'X_Point','Y_Point','X_Centroid','Y_Centroid','Tot_dist'
	## NOTE: distances in SECONDS
	joined_dfs.to_csv(results_folder+'Dictionary_cells_Centorid_'+str(Number_clusters), columns = ('X_Point','Y_Point','X_Centroid','Y_Centroid','Tot_dist'),header=False, index=False)
	'''
	
	# Save data frame in the form: 'X_Point','Y_Point','X_Centroid','Y_Centroid','Tot_dist'
	grouped = joined_dfs.groupby(['X_Centroid','Y_Centroid'])
	c = 1
	for name, group in grouped:
		# group = group.head(ranked_cells)
		## NOTE: distances in SECONDS
		group.head(ranked_cells).to_csv(New_Shp_Folder+'CL'+str(Number_clusters)+'cl'+str(c), columns = ('X_Point','Y_Point','X_Centroid','Y_Centroid','Tot_dist'),header=False, index=False)
		c = c + 1
	
"""
"""	
def Create_rank_shapefile(Results_Folder, n_of_warhouses):
	# This function creates a shapefile containing the centroids of the ranked cells.
	
	file = Results_Folder + 'Dictionary_cells_Centorid_' + str(n_of_warhouses)
	
	# Create a pandas df from the .csv file
	df = pd.read_csv(file, names=['X_Cell','Y_Cell','X_CL_centroid','Y_CL_centroid','Dist'])
	
	# Keep just the first two columns containing the coordinates of the cells
	df = df.loc[:,['X_Cell','Y_Cell']]
	df.columns = ['X', 'Y'] # rename the columns 'X' and 'Y'
	
	# Create a new shapefile folder:
	New_Shp_Folder = Results_Folder+'Ranked_Cells/Shapefiles'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Shp_Folder): os.makedirs(New_Shp_Folder)
	
	# Create the shpaefile:
	gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in df[['X', 'Y']].values.tolist()],crs={'init':'epsg:27700'})
	gdf.to_file(New_Shp_Folder + '/cells_centroids_'+ str(n_of_warhouses) + '_clusters' +'.shp')
	
	
	
for i in range(1, Warehouses_Max+1):
	# Create a KMeans instance with 'i' clusters: model
	model = KMeans(n_clusters=i)

	# Fit model to points
	model.fit(Point_list)

	# Determine the cluster labels
	labels = model.predict(Point_list)

	# Assign the cluster centers: centroids
	centroids = model.cluster_centers_
	
	# Save the coordinates of the centroids in a dictionary
	centroids_dict[i] = centroids
	
	# Save the centroids in a Shapefile
	# If the files already exist, skip this passage.
	if os.path.isfile(results_folder + 'Clusters_Centroids_Shapefiles/' + str(i) + '_warhouses_centroid.shp'):
		# Skip the generation of centroid shapefiles because these files already exist.
		pass
	else:
		Create_centroids_shp(results_folder, centroids, i)
	
	# Calculate the distance between the road nodes and the clusters centroids
	# and save the result in a .csv file
	# If the files already exist, skip this passage.
	if os.path.isfile(results_folder + 'Dictionary_nodes_ClustersCentroids_' + str(i)):
		# Skip the generation of nodes-centroids distance dictionary because these files already exist
		pass
	else:
		Dist_RoadN_ClCentroids(i, results_folder)
	
	# Rank the available cells on the base of their proximity to the centroids
	# And save in a .csv file
	if os.path.isfile(results_folder+'Dictionary_cells_Centorid_' + str(i)):
		# Skip the generation of sorted cells dictionary because these files already exist
		pass
	else:
		Rank_Cells(i, results_folder,number_of_assets)
	'''
	# Create shapefiles of the ranked cells
	if os.path.isfile(results_folder+'Ranked_Cells/Shapefiles'+'/cells_centroids_'+ str(i) + '_clusters' +'.shp'):
		# Skip the generation of sorted cells dictionary because these files already exist
		pass
	else:
		Create_rank_shapefile(results_folder, i)
	'''
	
	
""" MAKE A SCATTER PLOT

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot 
plt.scatter(point_df['X'], point_df['Y'])
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

# Print centroids coordinates
print "Centroids coordinates:"
for c in centroids:
	print c
"""


# CREATE AVAILABLE CELLS RANK
# Calculate the distance of available cells from clusters centroids.
# Rank them on the base of their proximity to the centroids.
