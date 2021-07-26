# -*- coding: utf-8 -*-
"""
Objectives currently optimised include:
    1. Travel times
    2. Costs = travel costs, land costs

"""

"""
# sys.path.append:
# works as it will search the anaconda directory (which is the default python installation)
# for any libraries you try to import, and when it doesn’t find them, it will then search the python27 directory
# as we have added this as a path to look in. The only downside to this is that is a version of a library is in
# the anaconda directory, it will use that, rather than the version you have coped over.
import sys
sys.path.append('C:/Python27/Lib/site-packages')
"""

# program_name = "Evaluate_02"

# import os
# os.system('cls')  # clears screen

import time
start_time = time.asctime()

# print "Program: " , program_name
# print "Starts at: " , start_time
print
# print "Importing modules..."
import fiona
import pandas as pd
import copy
import sys
import Initialise_04 as Init
import Constraints
import math
import numpy as np
# print "Modules imported."
print

# Data_Folder     = "P:/RLO/Python_Codes/Data/Hull/"
# Results_Folder  = "P:/RLO/Python_Codes/Hull_Case_Study/Results/"
# File_centroids  = "Available_centroids.shp"
# Lookup = Init.Generate_Lookup(Data_Folder, Results_Folder, File_centroids)
# Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
# No_Available = len(Lookup)
# Warehouses_Max = 10
# Warehouses_Min = 2
# Warehouse_Plan = Init.Generate_WarehousePlan_check_distance(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
# Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder) # List of coordinates of proposed sites

def Calc_fdist_maxTT(Results_Folder, Proposed_Sites):
	# Function that calculates fdist as the MAX TRavTime (in minutes) from a SINGLE proposed site to the farthest target
	
	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# Create the dataframe from csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets' coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)

	max_dist = Selected_points_trgt_df['Tot_dist'].max() # maximum value of the column

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if len(Proposed_Sites) == 0:
		fdist = 100000 # Meaningless very high number
	else:
		fdist = (max_dist/60.0) # Max TRavTime (in minutes) from a SINGLE proposed site to the farthest assigned target

	return fdist


def Calc_fdist_AV_90_dist(Results_Folder, Proposed_Sites, X_quantile, Min_W, Max_W):
	# Function that calculates fdist as average distance of the XXth percentile of assets
	
	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# Create the dataframe from csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	# sort data frame on targets' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) 

	# Keeps the first row of each target (which contains the closest point)
	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True)

	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # WRONG
	Selected_points_trgt_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	""" WRONG	
	# Create a Dataframe with only the coordinates of strat infr assets and Tot_dist
	A_W_df = Selected_points_trgt_df[['X_Target','Y_Target','Tot_dist']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = A_W_df.groupby(['X_Target', 'Y_Target']).quantile(X_quantile, interpolation='lower')
	"""
	# Create a Dataframe with only the coordinates of warehouses and Tot_dist
	A_W_df = Selected_points_trgt_df[['X_Point','Y_Point','Tot_dist']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = A_W_df.groupby(['X_Point','Y_Point']).quantile(X_quantile, interpolation='lower')
	
	#######################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if Min_W < len(Proposed_Sites) < Max_W:
		q_mean = (Quantile_df.mean())
		fdist = q_mean[0]/60.0 # The mean of the quantiles of each asset. /60 = from seconds to minutes
	else:
		fdist = 100 # Meaningless very high number
	
	return fdist


def Calc_fdist_mean(Results_Folder, Proposed_Sites):
	# FUNCTION THAT CALCULATES fdist as the average TRavTime (in minutes) from a SINGLE proposed site to general/average target

	dist_dict_file = 'Dictionary_cells_targets'
	available_centroids_file = 'Available_centroids.shp'
	
	# print "Creating the dataframe from csv file..."
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	# sort data frame on targets' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True)

	# keeps the first row of each target (which contains the closest point)
	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True)

	agg_dist = Selected_points_trgt_df['Tot_dist'].sum() # sum of all the values of the column
	
	# print "aggregate dist = " , agg_dist
	# print "number of proposed sites for warehouse = " , len(Proposed_Sites)
	

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	########################################################################################
	if len(Proposed_Sites) == 0:
		fdist = 10000 # Meaningless very high number
	else:
		# fdist = agg_dist/len(Proposed_Sites) # average sum (in seconds) of all the TRavTimes from a SINGLE proposed site to ALL the targets
		
		# COMPLETE FORMULA:
		# fdist = ((agg_dist/60)/len(Proposed_Sites))/(len(Selected_points_trgt_df.index)/len(Proposed_Sites)) # average TRavTime (in minutes) from a SINGLE proposed site to general/average target
		
		# SIMPLIFIED FORMULA:
		fdist = (agg_dist/60.0)/(len(Selected_points_trgt_df.index)) # average TRavTime (in minutes) from a SINGLE proposed site to general/average target
		
		# COMPONENTS OF THE FORMULA:
		# (agg_dist/60) = distance in minutes
		# (agg_dist/60)/len(Proposed_Sites) = average sum (in minutes) of all the TRavTimes from a SINGLE proposed site to ALL the targets
		# len(Selected_points_trgt_df.index) = lenght of the dataframe (= n_targets * n_proposed sites)
		# (len(Selected_points_trgt_df.index)/len(Proposed_Sites) = number of targets
		
	########################################################################################
	
	# print "Value of fdist = " , fdist
	# print
	return fdist


def Calc_fdist_GEUD(Results_Folder, Proposed_Sites, Min_W, Max_W, GEUD_power):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	GEUD = 0
	
	for tt in TT_list:
		GEUD = GEUD + tt**GEUD_power
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		fdist = GEUD**(1.0/GEUD_power)
	else:
		fdist = 999999 # Meaningless very high number	
	
	return fdist


def Calc_fdist_GEUD_checkDist(Results_Folder, Proposed_Sites, Min_W, Max_W, GEUD_power, Warehouse_Plan, Min_Warh_dist, Lookup):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of warehouses and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	GEUD = 0
	
	for tt in TT_list:
		GEUD = GEUD + tt**GEUD_power
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		fdist = GEUD**(1.0/GEUD_power)
	else:
		fdist = 999999 # Meaningless very high number
	
	# Check distance between warehouses:
	if Constraints.Check_Distance(Warehouse_Plan, Lookup, Min_Warh_dist) == True:
		# if warehouses are far enough from each other, do nothing
		pass
	else:
		fdist = 999999 # Meaningless very high number
	
	return fdist


def Calc_fdist_DeplTime(Results_Folder, Proposed_Sites, Min_W, Max_W, existing_fleet, Warehouse_Plan, Min_Warh_dist, Lookup):
	# Function that calculates the deployment time in the worst case scenario:
	# i.e. major flood in which the whole region needs barriers deployment.
	
	# Distance function: take into account manpower/fleet: assume a fleet dimension and that a single trip is enough for each SI asset
	# (calculate how many sites and how many defences), sum all the TT, multiply by 2 (i.e. return tirp) and divide them by the fleet dimension
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Save travel times values into a df:
	TT_var = Selected_avail_assets_df['Tot_dist']
	
	# Sum all the travel times:
	TT_sum = TT_var.sum()
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		fdist = 2 * TT_sum / existing_fleet
	else:
		fdist = 999999 # Meaningless very high number
	
	# Check distance between warehouses:
	if Constraints.Check_Distance(Warehouse_Plan, Lookup, Min_Warh_dist) == True:
		# if warehouses are far enough from each other, do nothing
		pass
	else:
		fdist = 999999 # Meaningless very high number
	
	return fdist


def Calc_fdist_squared(Results_Folder, Proposed_Sites, Min_W, Max_W):
	# Function that calculates the travel time "à la Generalised Equivalent Uniform Dose" 
	# from each strategic infrastructure asset to the closest warehouse
	
	# determine the closest warehouse for each available site
	dist_dict_file = 'Dictionary_cells_targets' # File that contains: X_Avail, Y_Avail, X_Asset, Y_Asset, Tot_dist
	
	# Create the dataframe from csv file:
	cells_Assets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Asset','Y_Asset','Tot_dist'])
	
	# Create a dataframe containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_avail_assets_df = cells_Assets_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on on Assets' coordinates and dist values:
	Selected_avail_assets_df.sort_values(['X_Asset','Y_Asset','Tot_dist'], ascending =[True, True, True], inplace=True)
	
	# Keeps the first row of each Asset (which contains the closest warehouse):
	Selected_avail_assets_df.drop_duplicates(subset=['X_Asset','Y_Asset'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the warehouses with ascending Tot_dist:
	# Selected_avail_assets_df.sort_values(['Tot_dist'], ascending =[True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and Tot_dist
	# A_W_df = Selected_avail_assets_df[['X_Asset','Y_Asset','Tot_dist']]
	
	# Save travel times values into a list:
	# TT_var = Selected_avail_assets_df['Tot_dist']
	TT_list = Selected_avail_assets_df["Tot_dist"].tolist()
	
	fdist = 0
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_W < len(Proposed_Sites) < Max_W:
		for tt in TT_list:
			fdist = fdist + tt**2
	else:
		fdist = 999999999999 # Meaningless very high number	
	
	return fdist


def Calc_fcost_constant(Proposed_Sites):
	# Constant fcost function
	# print "Calculate fcost function."
	
	average_cost  = 55.0 # average cost per warehouse = 55 GBP per sq meter per annum
	cell_dim	  = 2500.0 # squared meters
	warehouse_dim = 1000.0 # squared meters
	cost_per_cell =  average_cost * warehouse_dim # annual average cost per warehouse
	
	n_warehouses = len(Proposed_Sites) # number of warehouses:
	

	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if len(Proposed_Sites) == 0:
		fcost = 1000000 # Meaningless very high number
	else:
		fcost = n_warehouses * cost_per_cell
	########################################################################################

	# print "number of warehouses = " , n_warehouses
	# print "annual rent price per warehouse = " , cost_per_cell
	# print "value of fcost = " , fcost
	return fcost


def Calc_fcost(Proposed_Sites, Results_Folder):
	# Cost function that involves different sizes of warehouses
	
	# Dimension in square metres for Small, Medium and Large warehouses:
	S_W = 20*20  # squared metres (20mx20m)	- until 20 assets served
	M_W = 30*30  # squared metres (30mx30m) - until 60 assets served
	L_W = 40*50 # squared metres (40mx50m) 	- around 150 assets served
	
	Max_S_W = 20 # Maximum number of assets that a SMALL warehouse can serve
	Max_M_W = 60 # Maximum number of assets that a MEDIUM warehouse can serve
	
	average_cost  = 1.0 # unitary cost --> evaluating cost as space
	# average_cost  = 55.0 # average cost per warehouse = 55 GBP per sq meter per annum
	
	cost_S_W =  average_cost * S_W # annual average cost per SMALL warehouse
	cost_M_W =  average_cost * M_W # annual average cost per MEDIUM warehouse
	cost_L_W =  average_cost * L_W # annual average cost per LARGE warehouse
	
	# n_warehouses = len(Proposed_Sites) # number of warehouses
	
	# Determine how many assets are served by each warehouse:
	
	dist_dict_file = 'Dictionary_cells_targets'
	
	# Create the DataFrame from .csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	# Count how many assets are assigned to every proposed site:
	df = Selected_points_trgt_df.groupby(['X_Point','Y_Point']).size().reset_index(name="Served_Assets")
	
	# Save the number of served assets in a list:
	list_of_served_assets_numbers = df['Served_Assets'].values
	
	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if len(Proposed_Sites) == 0:
		fcost = 99999999 # Meaningless very high number
	else:
		fcost = 0
		for i in list_of_served_assets_numbers:
			if 0 < i <= Max_S_W:
				fcost = fcost + cost_S_W
			elif Max_S_W < i <= Max_M_W:
				fcost = fcost + cost_M_W
			elif Max_M_W < i:
				fcost = fcost + cost_L_W
	########################################################################################

	# print "number of warehouses = " , n_warehouses
	# print "annual rent price per warehouse = " , cost_per_cell
	# print "value of fcost = " , fcost
	return fcost


def Calc_fcost_dim(Proposed_Sites, Results_Folder, Data_Folder, Min_W, Max_W):
	# Cost function that involves different sizes of warehouses according to the amount of flood barriers needed
	
	# footprint of a 20ft. shipping container:
	# Container_area = 16.0 # Squared metres
	# 1 container = 100m of flood defences
	
	# Dimension in square metres for Small, Medium and Large warehouses:
	NF_W = 70.0  # squared metres - 4 containers - No flood defences, but only pumps and generators
	Small_W = 640.0  # squared metres - 40 containers = 4 km of flood barriers
	Medium_W = 1280.0 # squared metres - 80 containers = 8 km of flood barriers
	Big_W = 1920.0 # squared metres - 120 containers = 12 km of flood barriers
	Huge_W = 2560.0 # squared metres - 160 containers = 16 km of flood barriers
	
	Max_S_W = 4000.0 # Maximum temp flood def that a SMALL warehouse can store
	Max_M_W = 8000.0 # Maximum temp flood def that a MEDIUM warehouse can store
	Max_B_W = 12000.0 # Maximum temp flood def that a BIG warehouse can store
	
	average_cost  = 1.0 # unitary cost --> evaluating cost as space
	# average_cost  = 55.0 # average cost per warehouse = 55 GBP per sq meter per annum
	
	cost_NF_W =  average_cost * NF_W  # annual average cost per smallest warehouse
	cost_S_W =  average_cost * Small_W  # annual average cost per SMALL warehouse
	cost_M_W =  average_cost * Medium_W # annual average cost per MEDIUM warehouse
	cost_B_W =  average_cost * Big_W    # annual average cost per BIG warehouse
	cost_H_W =  average_cost * Huge_W   # annual average cost per HUGE warehouse
	
	# Determine how many assets are served by each warehouse and how much flood def lenght is required for their protection:
	dist_dict_file = 'Dictionary_cells_targets'
	
	# Create the DataFrame from .csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	# Create a df with the coordinates of the targets and the length of flood defences they require
	strat_infr_file = 'Strat_infr_table.csv'
	
	# REMEMBER to check that the table does NOT have headers (if it has headers, delete them!)
	
	strat_infr_table_df = pd.read_csv(Data_Folder+strat_infr_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target'])
	columns = ['ObjectID','Boolean','Typology','Orig_FID','In_floodzone']
	strat_infr_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need

	# Merge the dfs:	
	Pts_trgt_fldefs_df = Selected_points_trgt_df.merge(strat_infr_table_df, on=['X_Target','Y_Target'])

	# Count how many temporary defences are needed every proposed site:
	df = Pts_trgt_fldefs_df.groupby(['X_Point','Y_Point']).sum()

	# Save the amount of temporary defences lenght needed by the served assets in a list:
	list_of_needed_temp_def = df["Temp_flood_def"].tolist()
	
	# if len(list_of_needed_temp_def) == 0:
		# raise ValueError('List_of_needed_temp_def is an empty list. Even if no flood defs needed, the value should be 0, not empty. (Evaluate module, cost function)')
	
	# print list_of_needed_temp_def
	
	########################################################################################
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	########################################################################################
	if Min_W < len(Proposed_Sites) < Max_W:
		fcost = 0
		for i in list_of_needed_temp_def:
			if i == 0:
				fcost = fcost + cost_NF_W
			elif 0 < i <= Max_S_W:
				fcost = fcost + cost_S_W
			elif Max_S_W < i <= Max_M_W:
				fcost = fcost + cost_M_W
			elif Max_M_W < i <= Max_B_W:
				fcost = fcost + cost_B_W
			elif Max_B_W < i :
				fcost = fcost + cost_H_W
	else:
		fcost = 999999 # Meaningless very high number
	########################################################################################

	# print "value of fcost = " , fcost
	return fcost


def Calc_fcost_lin_fdist_depl(Proposed_Sites, Min_W, Max_W, Min_W_dim, Lookup_RurSubUrb,
							Land_price_rural, Land_price_suburban, Land_price_urban, NNN, barriers_lin_cost,
							Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, existing_fleet,
							cells_targets_df, strat_infr_table_df):
	# Linear cost function proportional to the floor space of warehouses multiplied by land value (different areas have different
	# land values according to their location: urban, suburban, rural).
	
	# Note: 1000m of temporary defences require 10 shipping containers --> 64 sq m
	# 100m of temporary def require 1 shipping container --> 6.4 sq m
	# Linear floor space cost per metre of flood barriers: 0.064 sq m
	
	# The cost function also takes into account:
	# - personnel
	# - additional trucks
	# - maintenance cost of warehouses and temporary defences
	# - cost of demountable barriers
	
	# Distance function: take into account manpower/fleet: assume a fleet dimension and that a single trip is enough for each SI asset
	# (calculate how many sites and how many defences), sum all the TT, multiply by 2 (i.e. return tirp) and divide them by the fleet dimension
	
	""" Moeved to main
	# Determine how many assets are served by each warehouse and how much flood def lenght is required for their protection:
	dist_dict_file = 'Dictionary_cells_targets'
	
	# Create the DataFrame from .csv file:
	cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])
	"""
	
	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	""" Moeved to main
	# Create a df with the coordinates of the targets and the length of flood defences they require
	strat_infr_file = 'Strat_infr_table.csv'
	
	# WARNING !!
	# REMEMBER to check that the table does NOT have headers (if it has headers, delete them!)
	strat_infr_table_df = pd.read_csv(Data_Folder+strat_infr_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target'])
	columns = ['ObjectID','Boolean','Typology','Orig_FID','In_floodzone']
	strat_infr_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need
	"""
	
	# Merge the dfs:	
	Pts_trgt_fldefs_df = Selected_points_trgt_df.merge(strat_infr_table_df, on=['X_Target','Y_Target'])
	
	# Land value: transform the list into a Pandas DataFrame
	Lookup_RurSubUrb_df = pd.DataFrame(Lookup_RurSubUrb, columns =['X_Point', 'Y_Point', 'RurSubUrb']) # 1 = rural, 2 = suburban, 3 = urban
	
	# Merge the dfs: add land value
	Pts_trgt_fldefs_lv_df = Pts_trgt_fldefs_df.merge(Lookup_RurSubUrb_df, on=['X_Point', 'Y_Point']) # df: 'X_Point', 'Y_Point', 'Temp_flood_def', 'RurSubUrb'
	
	# Count how many temporary defences are needed every proposed site:
	# fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point']).sum()
	# fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point','RurSubUrb'], as_index=False)[['Temp_flood_def', 'Tot_dist']].sum()
	fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point','RurSubUrb'], as_index=False).agg({'Temp_flood_def':'sum','Tot_dist':'mean'}).rename(columns={'Tot_dist':'Av_dist'}) # mean trip lenght per warehouse in seconds

	# Save the amount of temporary defences length needed by the served assets in a list:
	list_of_needed_temp_def = fl_df_lv['Temp_flood_def'].tolist()
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	if Min_W < len(Proposed_Sites) < Max_W:
		
		# Calculate Wcapex: proportional to warehouse dimension and land price
		# Calculate Wopex: warehouse maintenance cost, proportional to warehouse area
		# Calculate Rcapex: price of demountable barriers
		# Calculate Ropex: personnel cost + additional trucks to existing fleet + emergency resources maintenance cost
		Wcapex = 0
		Wopex = 0
		Rcapex = 0
		Ropex_def_maintenance = 0
		fdist = 0
		for i in range(len(list_of_needed_temp_def)):
		
			Rcapex += fl_df_lv.Temp_flood_def[i] * barriers_lin_cost
			
			Ropex_def_maintenance += fl_df_lv.Temp_flood_def[i] * Def_maint_lin_cost
			
			# N_trips = math.ceil(fl_df_lv.Temp_flood_def[i]/100) --> number of shipping containers needed (1 container = 100m barriers) --> i.e. number of trips
			# fdist = number of trips * average trip length
			N_trips = math.ceil(fl_df_lv.Temp_flood_def[i]/100)
			Av_trip_l = fl_df_lv.Av_dist[i]
			# fdist += (N_trips * Av_trip_l) #  One way
			fdist += (N_trips * Av_trip_l) * 2 #  Return trip
			
			if fl_df_lv.RurSubUrb[i] == 1: # if in rural area
				Wcapex += (fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_rural
				Wopex += NNN * ((fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_rural)
				
			elif fl_df_lv.RurSubUrb[i] == 2: # if in suburban area
				Wcapex += (fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_suburban
				Wopex += NNN * ((fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_suburban)
				
			elif fl_df_lv.RurSubUrb[i] == 3: # if in urban area
				Wcapex += (fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_urban
				Wopex += NNN * ((fl_df_lv.Temp_flood_def[i] * 0.064) * Land_price_urban)
		
		if Wcapex == 0:
			for i in range(len(list_of_needed_temp_def)):
				if fl_df_lv.RurSubUrb[i] == 1: # if in rural area
					Wcapex += (Min_W_dim * Land_price_rural)
					Wopex += NNN * (Min_W_dim * Land_price_rural)
					
				elif fl_df_lv.RurSubUrb[i] == 2: # if in suburban area
					Wcapex += (Min_W_dim * Land_price_suburban)
					Wopex += NNN * (Min_W_dim * Land_price_suburban)
					
				elif fl_df_lv.RurSubUrb[i] == 3: # if in urban area
					Wcapex += (Min_W_dim * Land_price_urban)
					Wopex += NNN * (Min_W_dim * Land_price_urban)
		
		Ropex_personnel = P_pay * P_num * P_h # personnel pay * numebr of people * working hours
		
		Ropex_fleet = T_num * T_cost # Number of trucks * cost per truck
		
		Ropex = Ropex_def_maintenance + Ropex_personnel + Ropex_fleet
		
		fcost = Wcapex + Wopex + Rcapex + Ropex
		
		fdist = (fdist/existing_fleet)/60 # results in minutes
	else:
		fcost = 800000 # Meaningless very high number
		fdist = 2000 # Meaningless very high number
	
	return fcost, fdist


def Calc_fcost_nonlin_fdist_depl(Proposed_Sites, Min_W, Max_W, Min_W_dim, Lookup_RurSubUrb,
							Land_price_factor_suburban, Land_price_factor_urban, NNN, barriers_lin_cost,
							Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, lorries_per_depo,
							cells_targets_df, strat_infr_table_df):
	# Non-linear cost function proportional to the floor space of warehouses and to the land value (different areas have different
	# land values according to their location: urban, suburban, rural).
	# Warehouse cost per sq m formula
	# y = -10.39 * ln(x) + 114.79
	# y = warehouse cost per sq m; x = warehouse dimension
	
	# Note: 1000m of temporary defences require 10 shipping containers --> 64 sq m
	# 100m of temporary def require 1 shipping container --> 6.4 sq m
	# Floor space cost per metre of flood barriers: 0.064 sq m
	
	# The cost function also takes into account:
	# - personnel
	# - additional trucks
	# - maintenance cost of warehouses and temporary defences
	# - cost of demountable barriers
	
	# Distance function: take into account manpower/fleet: assume a fleet dimension proportional to warehouses number and that a single trip is enough for each SI asset
	# (calculate how many sites and how many defences), sum all the TT, multiply by 2 (i.e. return tirp) and divide them by the fleet dimension
	
	# Create a DataFrame containing the coordinates of the proposed sites
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Point','Y_Point'])
	
	Selected_points_trgt_df = cells_targets_df.merge(proposed_sites_df, on=['X_Point','Y_Point'])
	
	Selected_points_trgt_df.sort_values(['X_Target','Y_Target','Tot_dist'], ascending =[True, True, True], inplace=True) # sort data frame on targets coordinates and dist value

	Selected_points_trgt_df.drop_duplicates(subset=['X_Target','Y_Target'], keep='first', inplace=True) # keeps the first row of each target (which contains the closest point)
	
	# Merge the dfs:	
	Pts_trgt_fldefs_df = Selected_points_trgt_df.merge(strat_infr_table_df, on=['X_Target','Y_Target'])
	
	# Land value: transform the list into a Pandas DataFrame
	Lookup_RurSubUrb_df = pd.DataFrame(Lookup_RurSubUrb, columns =['X_Point', 'Y_Point', 'RurSubUrb']) # 1 = rural, 2 = suburban, 3 = urban
	
	# Merge the dfs: add land value
	Pts_trgt_fldefs_lv_df = Pts_trgt_fldefs_df.merge(Lookup_RurSubUrb_df, on=['X_Point', 'Y_Point']) # df: 'X_Point', 'Y_Point', 'Temp_flood_def', 'RurSubUrb'
	
	# Count how many temporary defences are needed every proposed site:
	# fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point']).sum()
	# fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point','RurSubUrb'], as_index=False)[['Temp_flood_def', 'Tot_dist']].sum()
	fl_df_lv = Pts_trgt_fldefs_lv_df.groupby(['X_Point','Y_Point','RurSubUrb'], as_index=False).agg({'Temp_flood_def':'sum','Tot_dist':'mean'}).rename(columns={'Tot_dist':'Av_dist'}) # mean trip lenght per warehouse in seconds

	# Save the amount of temporary defences length needed by the served assets in a list:
	list_of_needed_temp_def = fl_df_lv['Temp_flood_def'].tolist()
	
	# If a Warehouse plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fcost
	if Min_W < len(Proposed_Sites) < Max_W:
		
		# Calculate Wcapex: proportional to warehouse dimension and land price
		# Calculate Wopex: warehouse maintenance cost, proportional to warehouse area
		# Calculate Rcapex: price of demountable barriers
		# Calculate Ropex: personnel cost + additional trucks to existing fleet + emergency resources maintenance cost
		Wcapex = 0
		Wopex = 0
		Rcapex = 0
		Ropex_def_maintenance = 0
		fdist = 0
		for i in range(len(list_of_needed_temp_def)):
		
			Rcapex += fl_df_lv.Temp_flood_def[i] * barriers_lin_cost
			
			Ropex_def_maintenance += fl_df_lv.Temp_flood_def[i] * Def_maint_lin_cost
			
			# N_trips = math.ceil(fl_df_lv.Temp_flood_def[i]/100) --> number of shipping containers needed (1 container = 100m barriers) --> i.e. number of trips
			# fdist = number of trips * average trip length
			N_trips = math.ceil(fl_df_lv.Temp_flood_def[i]/100)
			Av_trip_l = fl_df_lv.Av_dist[i]
			# fdist += (N_trips * Av_trip_l) #  One way
			fdist += (N_trips * Av_trip_l) * 2 #  Return trip
			
			if fl_df_lv.RurSubUrb[i] == 1: # if in rural area
				Warehouse_cost_per_sq_m = -10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79 #  Added "+ 1" to avoid log(0)
				Warehouse_dimension = fl_df_lv.Temp_flood_def[i] * 0.064
				Wcapex +=  Warehouse_cost_per_sq_m * Warehouse_dimension
				Wopex += NNN * Warehouse_cost_per_sq_m * Warehouse_dimension
				
			elif fl_df_lv.RurSubUrb[i] == 2: # if in suburban area
				Warehouse_cost_per_sq_m = (-10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79) * Land_price_factor_suburban #  Added "+ 1" to avoid log(0)
				Warehouse_dimension = fl_df_lv.Temp_flood_def[i] * 0.064
				Wcapex += Warehouse_cost_per_sq_m * Warehouse_dimension
				Wopex += NNN * Warehouse_cost_per_sq_m * Warehouse_dimension
				
			elif fl_df_lv.RurSubUrb[i] == 3: # if in urban area
				Warehouse_cost_per_sq_m = (-10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79) * Land_price_factor_urban #  Added "+ 1" to avoid log(0)
				Warehouse_dimension = fl_df_lv.Temp_flood_def[i] * 0.064
				Wcapex += Warehouse_cost_per_sq_m * Warehouse_dimension
				Wopex += NNN * Warehouse_cost_per_sq_m * Warehouse_dimension
		
		if Wcapex == 0: # Use minimum warehouse dimension (i.e. no barriers stored, only pumps and generators)
			for i in range(len(list_of_needed_temp_def)):
				if fl_df_lv.RurSubUrb[i] == 1: # if in rural area
					Warehouse_cost_per_sq_m = -10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79 #  Added "+ 1" to avoid log(0)
					Warehouse_dimension = Min_W_dim
					Wcapex += Warehouse_cost_per_sq_m * Warehouse_dimension
					Wopex += NNN * (Warehouse_cost_per_sq_m * Warehouse_dimension)
					
				elif fl_df_lv.RurSubUrb[i] == 2: # if in suburban area
					Warehouse_cost_per_sq_m = (-10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79) * Land_price_factor_suburban #  Added "+ 1" to avoid log(0)
					Warehouse_dimension = Min_W_dim
					Wcapex += Warehouse_cost_per_sq_m * Warehouse_dimension
					Wopex += NNN * (Warehouse_cost_per_sq_m * Warehouse_dimension)
					
				elif fl_df_lv.RurSubUrb[i] == 3: # if in urban area
					Warehouse_cost_per_sq_m = (-10.39 * math.log(fl_df_lv.Temp_flood_def[i] * 0.064 + 1) + 114.79) * Land_price_factor_urban #  Added "+ 1" to avoid log(0)
					Warehouse_dimension = Min_W_dim
					Wcapex += Warehouse_cost_per_sq_m * Warehouse_dimension
					Wopex += NNN * (Warehouse_cost_per_sq_m * Warehouse_dimension)
		
		Ropex_personnel = P_pay * P_num * P_h # personnel pay * numebr of people * working hours
		
		Ropex_fleet = T_num * T_cost # Number of trucks * cost per truck
		
		Ropex = Ropex_def_maintenance + Ropex_personnel + Ropex_fleet
		
		fcost = Wcapex + Wopex + Rcapex + Ropex
		
		fleet = len(list_of_needed_temp_def) * lorries_per_depo # n of warehouses * lorries_per_depo
		fdist = (fdist/fleet)/60 # results in minutes
	else:
		fcost = 800000 # Meaningless very high number
		fdist = 6000 # Meaningless very high number
	
	return fcost, fdist


# if __name__ == '__main__':
	
	# fdist = Calc_fdist(Results_Folder, Proposed_Sites)
	# fcost = Calc_fcost(Proposed_Sites)
	
# print	
# print "Program" , program_name, " started at: " , start_time
# end_time = time.asctime()
# print "Program" , program_name, " terminates at: " , end_time