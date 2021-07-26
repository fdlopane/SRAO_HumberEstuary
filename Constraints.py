# -*- coding: utf-8 -*-
"""
Module Containing a series of constraint handling for the genetic algorithm
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

import numpy as np
import os


def Check_Distance(Warehouse_Plan, Lookup, Min_distance):
	# Function that checks if a proposed site is far enough from other already assigned sites.
	
	war_list = []
	dist_list = []
	
	for w in range(len(Warehouse_Plan)):
		if Warehouse_Plan[w] == 1:
			war_list.append(Lookup[w])
	
	if len(war_list) > 1:
		for i in war_list:
			for j in war_list:
				d = ( (i[0] - j[0])**2 + (i[1] - j[1])**2 )**(0.5)
				if d > 0:
					dist_list.append(d)

		if min(dist_list) < Min_distance:
			# print "too close: ", min(dist_list)
			return False
		else:
			# print "distance check passed"
			return True
	else:
		# in the Warehouse_Plan there is a number of warehouses < 2
		return True


def OLD_Check_Distance(Warehouse_Plan, ji, Availability_Raster, XSize, YSize, No_Available, Lookup):
	# Function that checks if a proposed site is far enough from other already assigned sites.
	# Returns True if there are not other occupied cells in a radius as big as "distance" variable
	# Returns False if there are other occupied cells in a 1km radius

	cell_size = 50    # metres
	Min_distance  = 30000.0 # allowed distance between cells (metres) 30km=60 min at 30km/h

	# Initialise the occupied_raster with all zeros. Raster dimensions equal to the Availability_Raster
	occupied_raster = np.zeros(Availability_Raster.shape, dtype = np.int)

	# Generate a lookup_raster in which there are ones in the already occupied cells
	for i in range(No_Available):
		tu_l  = tuple(Lookup[i])
		x_L = int(tu_l[1])
		y_L = int(tu_l[0])
		if Warehouse_Plan[i] == 1:
			occupied_raster[y_L,x_L] = 1

	x_site = int(ji[1])
	y_site = int(ji[0])
	p = 0

	for x in range(max(0, x_site-(Min_distance/cell_size)), min(XSize, x_site+(Min_distance/cell_size)+1)):
		for y in range(max(0, y_site-(Min_distance/cell_size)), min(YSize, y_site+(Min_distance/cell_size)+1)):
			if occupied_raster[y,x] == 1:
				p += 1
	if p == 0:
		return True
	else:
		return False


def Remove(orig_tuple, element_to_remove):
	# Function which is called once a child in the offspring is found to
	# exceed the maximum number of households or be lower than the minimum
	lst = list(orig_tuple)
	lst.remove(element_to_remove)
	# return the offspring array with the element removed
	return tuple(lst)


def Check_TotWarehouse_Constraint(Warehouses_Max, Warehouses_Min, Results_Folder):
	# Decorator function acts as a constraint which interrupts the selection process to ensure
	# that the children selected achieve the required number of warehouses

	def decCheckBounds(func):
		def wrapCheckBounds(*args, **kargs):

			# Extract the offspring solutions
			offsprings = func(*args, **kargs)
			strt_len = len(offsprings)
			# Extract each of the children from the offspring
			for child in offsprings:
				# import intialise module to create a dwelling plan else
				# the total dwellings only counts densities
				num_warehouses = sum(child)

				if num_warehouses < Warehouses_Min or num_warehouses > Warehouses_Max:
					# if a warehouse plan doesn't fall between the min and max it is removed from the offspring
					offsprings.remove(child)
					# print('removed child')

			end_len = len(offsprings)

			# Calculate the number of solutions retained after the constraint
			per_retained = float(100 * end_len / strt_len)

			# Load the previous list of retaintion rates and add new retention
			Retained_list = np.loadtxt(Results_Folder+"Warehouse_Constraint.txt", delimiter=",")
			Updated_Retained_list = np.append(Retained_list, per_retained)
			# Save the updated list
			np.savetxt(Results_Folder+"Warehouse_Constraint.txt", Updated_Retained_list, delimiter=',', newline='\n',fmt="%i")

			# print '% of solutions retained after Total Dwellings Constraint', per_retained
			return offsprings
		return wrapCheckBounds
	return decCheckBounds
	
	
def Check_Constraint_Select(Warehouses_Max, Warehouses_Min, Results_Folder, Min_distance):
	# Decorator function acts as a constraint which interrupts the selection process to ensure
	# that the children selected don't have too close warehouses
	
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
	
	def decCheckConstr(func):
		def wrapCheckConstr(*args, **kargs):
			# Extract the offspring solutions
			offsprings = func(*args, **kargs)
			strt_len = len(offsprings)
			# Extract each of the children from the offspring
			for child in offsprings:
				num_warehouses = sum(child)
				
				# CHECK DISTANCE ##
				war_list = []
				dist_list = []
				
				for w in range(len(child)):
					if child[w] == 1:
						war_list.append(Lookup[w])
				
				# print
				# print "n. of warehouses of this child: " , len(war_list)
				# print war_list
				
				if len(war_list) > 1:
					for i in war_list:
						for j in war_list:
							d = ( (i[0] - j[0])**2 + (i[1] - j[1])**2 )**(0.5)
							if d > 0:
								dist_list.append(d)

				
					# print "min d = ", min(dist_list)
					if min(dist_list) < Min_distance:
						# if a warehouse plan doesn't have far enough warehouses it is removed from the offspring
						offsprings.remove(child)
						# print('removed child in selection because of too close warehouses')

				# CHECK NUM. OF WAREHOUSES ##
				if num_warehouses < Warehouses_Min or num_warehouses > Warehouses_Max:
					# if a warehouse plan doesn't fall between the min and max it is removed from the offspring
					offsprings.remove(child)
					# print('removed child in selection because too many (or not enough) warehouses')
					
			end_len = len(offsprings)

			# Calculate the number of solutions retained after the constraint
			per_retained = float(100 * end_len / strt_len)

			# Load the previous list of retaintion rates and add new retention
			Retained_list = np.loadtxt(Results_Folder+"Warehouse_Constraint.txt", delimiter=",")
			Updated_Retained_list = np.append(Retained_list, per_retained)
			# Save the updated list
			np.savetxt(Results_Folder+"Warehouse_Constraint.txt", Updated_Retained_list, delimiter=',', newline='\n',fmt="%i")

			# print '% of solutions retained after Total Dwellings Constraint', per_retained
			return offsprings
		return wrapCheckConstr
	return decCheckConstr


def Check_Constraint_Mate_Mutate(Warehouses_Max, Warehouses_Min, Results_Folder, Min_distance):
	# Decorator function acts as a constraint which makes sure 
	# that the mated or mutated children don't have too many or close warehouses
	
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
	
	def decCheckConstr(func):
		def wrapCheckConstr(*args, **kargs):
			# Extract the offspring solutions
			offsprings = func(*args, **kargs)
			strt_len = len(offsprings)
			# Extract each of the children from the offspring
			# count_warehouses = 0
			for child in offsprings:
				num_warehouses = sum(child)
				war_list = []
				dist_list = []
				
				# CHECK MAX NUM. OF WAREHOUSES ##
				for w in range(len(child)):
					if child[w] == 1:
						count_warehouses =+ 1
						if count_warehouses > Warehouses_Max:
							child[w] = 0
							# print('warehouse removed warehouse in mating/mutating because there are too many')
						else:
							war_list.append([Lookup[w], w])
				
				
				# CHECK DISTANCE ##
				
				if len(war_list) > 1:
					for i in war_list:
						for j in war_list:
							d =  ( (int(i[0][0]) - int(j[0][0]))**2 + (int(i[0][1]) - int(j[0][1]))**2 )**(0.5)
							if d > 0:
								dist_list.append([i,j,d])
								# print "i = ",i
								# print "int(i[0][0]) = " , int(i[0][0])
								# print "int(j[0][0]) = " , int(j[0][0])
								# print "int(i[0][1]) = " , int(i[0][1])
								# print "int(j[0][1]) = " , int(j[0][1])
							
				for wa in dist_list:
					if wa[2] < Min_distance:
						child[wa[1][1]] = 0 # removing a warehouse from the child
						# print('removed warehouse in mating/mutating because too close to another')
								

				
				"""
				elif min(dist_list) < Min_distance:
					# print "length child = ", len(child)
					# print "j = ", j
					# print "[j][0][0] = ",   [j][0][0]
					# print "[j][0][0][0] = ",[j][0][0][0]
					# print "[j][0][0][1] = ",[j][0][0][1]
					# print "[j][0][1] = ",   [j][0][1]
					# print "[j][0][1][0] = ",[j][0][1][0]
					
					# child[[j][0][1][0]] = 0
					
					## MODIFICATION:
					## with war_list.remove(j) I remove an element from a list but I don't do nothing to the child
					# war_list.remove(j)
					offsprings.remove(child)
					
					# print('removed warehouse because too close to another')
				"""
			
			end_len = len(offsprings)

			# Calculate the number of solutions retained after the constraint
			per_retained = float(100 * end_len / strt_len)
			
			""" NOT SAVING RETAIN LIST, uncomment to save it
			# Load the previous list of retaintion rates and add new retention
			Retained_list = np.loadtxt(Results_Folder+"Warehouse_Constraint.txt", delimiter=",")
			Updated_Retained_list = np.append(Retained_list, per_retained)
			
			# Save the updated list
			np.savetxt(Results_Folder+"Warehouse_Constraint.txt", Updated_Retained_list, delimiter=',', newline='\n',fmt="%i")
			"""
			# print '% of solutions retained after Total Dwellings Constraint', per_retained
			return offsprings
		return wrapCheckConstr
	return decCheckConstr