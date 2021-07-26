# RESOURCE ALLOCATION OPTIMISATION (RAO)

# Started writing: 11/07/2017

program_name = "RAO"

""" IMPORT MODULES
"""
import os
os.system('cls')  # clears screen

import time
import os.path

start_time = time.asctime()
running_time_start = time.clock()

print "Program: RAO"
print "Starts at: " , start_time
print
print "Importing modules..."

# DEAP modules to facilitate the genetic algorithm
from deap import algorithms
from deap import base 
from deap import creator # creates the initial individuals
from deap import tools # defines operators

# Modules to handle rasters into arrays
import rasterIO

# Module to handle arrays
import numpy as np

# Module to handle math operators
import math
import random as rndm
from copy import copy

# Module to handle .csv files
import pandas as pd

# Module to handle shape files
import fiona

# Modules for the spatial optimisation framework
import Initialise_04 as Init       # initialisation module
import Evaluate_02 as Eval         # Module to calculate and return fitnesses for BAU scenarios
import FL_Evaluate_02 as Eval_Disr # Module to calculate and return fitnesses for disruption scenario
import Constraints as Constraint   # Contraints module
import Outputs as Output           # Outputs module for BAU scenarios
import FL_Outputs as Output_Disr   # Outputs module for disruption scenarios

import gc
gc.disable()

print "All modules imported."
print

""" DIRECTORIES
"""
# on P: drive:
Data_Folder     = "../Data/Hull_500m_resolution/"
Code_Folder     = "../Hull_Case_Study/"
Results_Folder  = "./Results_500m_resolution/"
External_Results_Folder = "./Results_500m_resolution/"

""" CHOICE OF SCENARIO
"""
# Comment out all the scenarios that are not to be run
# scenario = "BAU_discrete"              # Business as usual scenario (BAU) with discrete dimensions of warehouses
# scenario = "Disruption"                 # Scenario with disrupted road network
# scenario = "BAU_linear_av"              # BAU scenario with linear cost function - land prices: average
# scenario = "BAU_linear_urban_rural"     # BAU scenario with linear cost function - land prices: urban and rural
# scenario = "BAU_linear_urban_sub_rural" # BAU scenario with linear cost function - land prices: urban, suburban and rural
# scenario = "BAU_Energy_nonlin"          # BAU scenario with non-linear cost function. Priority to energy infr. Land prices: urban, suburban and rural
# scenario = "BAU_RankCI_nonlin"          # BAU scenario with non-linear cost function. Priority to ranked CI (e.g. top3, top5 etc...) Land prices: urban, suburban and rural
scenario = "BAU_RankCI_NoPoliceFire_nonlin" # BAU scenario with non-linear cost function. Priority to ranked CI (e.g. top3, top5 etc...) without police and fire stations Land prices: urban, suburban and rural

if scenario == "Disruption":
	Modules = ['Initialisation', Init.__name__, 'Evaluation', Eval.__name__, 
           'Constraints', Constraint.__name__, 'Output', Output.__name__]
else:
	Modules = ['Initialisation', Init.__name__, 'Evaluation', Eval.__name__, 
           'Constraints', Constraint.__name__, 'Output', Output.__name__]

""" PROBLEM FORMULATION - General Parameters
"""
# Variables for the search
Spat_Res       = 500		# Defines the spatial resolution (length of the side of a single cell - in meters)
Warehouses_Max = 10			# Maximum number of warehouses
Warehouses_Min = 3			# Minimum number of warehouses
Min_Warh_dist  = 10000.0    # Minimum distance between 2 warehouses - in metres

X_quantile = 0.9 # Quantile for travel time opt function. (i.e. how much time XX % of assets are reached from the warehouses)

GEUD_power = 2 # Power of the Generalised Equivalent uniform dose

# Land price
land_price_av = 55.0			      # Average cost per warehouse = 55 GBP per sq meter per annum
suburb_moltiplicator = 1.5
urb_moltiplicator = 2.
land_price_suburb = suburb_moltiplicator * land_price_av # 1.5 times of the average land price
land_price_urb = urb_moltiplicator * land_price_av    # 2x of the average land price

# Time function parameters
existing_fleet = 10. # Number of trucks already available
lorries_per_depo = 1. # Number of lorries per warehouse

# Critical infrastructure to prioritise (e.g. only protect top3 CI)
CI_rank = 10

# Cost function parameters
Min_W_dim = 70.0  # Minimmum warehouse dimension (in sq m) - 4 shipping containers - No flood defences, but only pumps and generators
NNN = 0.2 # The NNN fees are property taxes, property insurance and CAM (Common Area Maintenance) - calculated as an additional % of the warehouse rental price
# barriers_lin_cost = 20000./100 # Linear cost (per metre) of demountable barriers - 20k GBP per 100m; 1.2m height
barriers_lin_cost = 0. # Linear cost (per metre) of additional demountable barriers
Def_maint_lin_cost = 0. # Linear cost of barriers maintenance (0 because considered included in barriers_lin_cost)
P_pay = 0. # Additional personnel hourly pay in GBP
P_num = 0. # Additional personnel required (number of people)
P_h = 0. # Number of working hours per person
T_num = 0. # Number of additional trucks required
T_cost = 0. # Cost in GBP per additional truck

Problem_Parameters = ['Spatial Resolution (m^2)', Spat_Res, 'Maximum warehouses', Warehouses_Max, 'Minimum warehouses', Warehouses_Min, 'GEUD power', GEUD_power,
					  'Average cost per warehouse', land_price_av, 'Price multiplicator for suburban areas', suburb_moltiplicator,
					  'Price multiplicator for urban areas', urb_moltiplicator,
					  'Existing lorries fleet (only for fixed fleet scenarios)', existing_fleet,
					  'Lorries per warehouse (for non/fixed fleet scenarios)', lorries_per_depo,
					  'Strategic infrastructure rank selection (for CI priority scenarios). Top:', CI_rank,
					  'Minimmum warehouse dimension (in sq m)', Min_W_dim, 'NNN fees', NNN]

# Generate availability raster:
if os.path.isfile(os.path.join(Data_Folder, "Available.tif")):
	time_aval = time.asctime()
	print "Skip the generation of Availability Raster because this file already exists in this directory."
	print
else:
	time_aval = time.asctime()
	print "Availability raster, starts at: " , time_aval
	print
	Init.Generate_Availability(Data_Folder)


#LOOKUP
# To handle the constraints the algorithm uses a lookup for proposed allocation sites.
# The lookup list contains the locations of sites actually available for building warehouses.
# The function called creates a lookup based on our preferences, saves it and returns the list. 

if os.path.isfile(os.path.join(Results_Folder, "lookup.txt")):
	time_lookup_s = time.asctime()
	print "Generation of Lookup starts at: " , time_lookup_s
	print "Skip the generation of Lookup because this file already exists in this directory."
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
	time_lookup_e = time.asctime()
	print "Lookup uploaded at: " , time_lookup_e
else:
	File_centroids = "Available_centroids.shp"
	Lookup = Init.Generate_Lookup(Data_Folder, Results_Folder, File_centroids)

if os.path.isfile(os.path.join(Results_Folder, "lookup_RurSubUrb.txt")):
	Lookup_RurSubUrb = (np.loadtxt(os.path.join(Results_Folder, "lookup_RurSubUrb.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
else:
	File_centroids_RurSubUrb = "Available_centroids_RurSubUrb.shp"
	Lookup_RurSubUrb = Init.Generate_Lookup_RurSubUrb(Data_Folder, Results_Folder, File_centroids_RurSubUrb)


# So we know how long to make the chromosome
No_Available = len(Lookup) # number of sites with space for development
print "Number of available cells: " , No_Available 


# OUTPUT VARIABLES
# For results 
Sols, Gens = [],[] # Saves all solutions found, saves each generation created                       
# Keep a record of the retaintion after constraints
start = [] # initial array
# Resave the files to contain the arrays
np.savetxt(Results_Folder+'Warehouse_Constraint.txt', start,  delimiter=',', newline='\n')


""" TYPES - creating fitness class, negative weight implies minimisation 
"""
# FITNESS - Defining the number of fitness 
# objectives to minimise or maximise
# Creating types (Fitness, Individual), DEAP documentation: http://deap.readthedocs.io/en/master/tutorials/basic/part1.html
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # -1.0 for each objective to minimise

# INDIVIDUAL
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMin) #typecode b = integer


""" INITIALISATION - Initially populating the types
"""
toolbox = base.Toolbox()

def Generate_W_Plan(Ind, Warehouses_Max, Warehouses_Min):
	# this function takes as an argument "Ind" which is the creator.individual that takes as an argument the warehouse plan
	# and returns the Individual in which is saved a potential solution
	# Warehouse_Plan is a list of 0s and 1s, where 1=assigned warehouse. To know the coordinates make reference to Lookup list.
	
	# Warehouse_Plan = Init.Generate_WarehousePlan_check_distance(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
	
	Warehouse_Plan = Init.Generate_WarehousePlan_Cluster_Ranking(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder, Min_Warh_dist)
	# Warehouse_Plan = Init.Generate_WarehousePlan(No_Available, Warehouses_Max, Warehouses_Min, Data_Folder, Results_Folder)

	return Ind(Warehouse_Plan) 

toolbox.register("individual", Generate_W_Plan, creator.Individual, Warehouses_Max, Warehouses_Min) # creates an individual = single potential solution
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # creates a population of individuals = set of potential solutions


"""  FUNCTIONS - Evaluate functions and constraint handling
"""
# Create dataframes to be used in evaluation functions
# (do it here, so the csv reading is not repeated at each iteration --> pass the df as an argument of eval function)

dist_dict_file = 'Dictionary_cells_targets'
cells_targets_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Point','Y_Point','X_Target','Y_Target','Tot_dist'])

# Creation of strategic infrastructure dataframes

# WARNING !!
# REMEMBER to check that the table does NOT have headers (if it has headers, delete them!)

# Strategic infrastructure
strat_infr_file = 'Strat_infr_table.csv'
strat_infr_table_df = pd.read_csv(Data_Folder+strat_infr_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target'])
columns = ['ObjectID','Boolean','Typology','Orig_FID','In_floodzone']
strat_infr_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need

# Power prod/distr Strategic infrastructure
strat_infr_power_file = 'Strat_infr_power_table.csv'
strat_infr_power_table_df = pd.read_csv(Data_Folder+strat_infr_power_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target'])
columns = ['ObjectID','Boolean','Typology','Orig_FID','In_floodzone']
strat_infr_power_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need

# Ranked Strategic infrastructure
strat_infr_rank_file = 'Strat_infr_table_rank.csv'
strat_infr_rank_table_df = pd.read_csv(Data_Folder+strat_infr_rank_file, names=['ObjectID','Boolean','Typology','Orig_FID','In_floodzone','Temp_flood_def','X_Target','Y_Target','Served_buildings','Rank'])
columns = ['ObjectID','Boolean','Orig_FID','In_floodzone']
strat_infr_rank_table_df.drop(columns, inplace=True, axis=1) # Drop the columns that I don't need
# Filter the strategic infrastructure table only selecting the assets to protect (e.g. top3, top5 etc...)
strat_infr_rank_table_df = strat_infr_rank_table_df.loc[strat_infr_rank_table_df['Rank'] <= CI_rank]
strat_infr_rankNoPoliceFire_table_df = strat_infr_rank_table_df.loc[strat_infr_rank_table_df['Rank'] <= CI_rank]
strat_infr_rankNoPoliceFire_table_df = strat_infr_rankNoPoliceFire_table_df[~strat_infr_rankNoPoliceFire_table_df['Typology'].str.contains('station')] # select all the raws that NOT contain the word 'station'


def Evaluate(Warehouse_Plan):
	# Generate the evaluation of functions to minimise/maximise
	
	if scenario == "BAU_discrete":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		# Dist_Fit = Eval.Calc_fdist(Results_Folder, Proposed_Sites)   
		# Dist_Fit = Eval.Calc_fdist_AV_90_dist(Results_Folder, Proposed_Sites, X_quantile, Warehouses_Min, Warehouses_Max)   
		# Dist_Fit = Eval.Calc_fdist_GEUD(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power)   
		Dist_Fit = Eval.Calc_fdist_GEUD_checkDist(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power, Warehouse_Plan, Min_Warh_dist, Lookup)   
		# Dist_Fit = Eval.Calc_fdist_squared(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max)
		
		# Cost_Fit = Eval.Calc_fcost(Proposed_Sites, Results_Folder)
		Cost_Fit = Eval.Calc_fcost_dim(Proposed_Sites, Results_Folder, Data_Folder, Warehouses_Min, Warehouses_Max)
		
	elif scenario == "Disruption":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites

		# Dist_Fit = Eval_Disr.Calc_fdist(Results_Folder, Proposed_Sites)   
		# Dist_Fit = Eval_Disr.Calc_fdist_AV_90_dist(Results_Folder, Proposed_Sites, X_quantile, Warehouses_Min, Warehouses_Max)   
		# Dist_Fit = Eval_Disr.Calc_fdist_GEUD(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power)   
		Dist_Fit = Eval_Disr.Calc_fdist_GEUD_checkDist(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max, GEUD_power, Warehouse_Plan, Min_Warh_dist, Lookup)   
		# Dist_Fit = Eval_Disr.Calc_fdist_squared(Results_Folder, Proposed_Sites, Warehouses_Min, Warehouses_Max)
		
		# Cost_Fit = Eval_Disr.Calc_fcost(Proposed_Sites, Results_Folder)
		Cost_Fit = Eval_Disr.Calc_fcost_dim(Proposed_Sites, Results_Folder, Data_Folder, Warehouses_Min, Warehouses_Max)
	
	elif scenario == "BAU_linear_av":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_lin_fdist_depl(Proposed_Sites, Warehouses_Min, Warehouses_Max,Min_W_dim, Lookup_RurSubUrb,
															land_price_av, land_price_av, land_price_av, NNN, barriers_lin_cost,
															Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, existing_fleet,
															cells_targets_df, strat_infr_table_df) # 3 times land_price_av because this scenario is av price
		
	elif scenario == "BAU_linear_urban_rural":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_lin_fdist_depl(Proposed_Sites,	Warehouses_Min, Warehouses_Max, Min_W_dim, Lookup_RurSubUrb,
															land_price_av, land_price_av, land_price_urb, NNN, barriers_lin_cost,
															Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, existing_fleet,
															cells_targets_df, strat_infr_table_df) # 3 times land_price_av because this scenario is av price
		
	elif scenario == "BAU_linear_urban_sub_rural":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_lin_fdist_depl(Proposed_Sites,	Warehouses_Min, Warehouses_Max, Min_W_dim, Lookup_RurSubUrb,
															land_price_av, land_price_suburb, land_price_urb, NNN, barriers_lin_cost,
															Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, existing_fleet,
															cells_targets_df, strat_infr_table_df) # 3 times land_price_av because this scenario is av price
	
	elif scenario == "BAU_Energy_nonlin":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_nonlin_fdist_depl(Proposed_Sites, Warehouses_Min, Warehouses_Max, Min_W_dim, Lookup_RurSubUrb,
							suburb_moltiplicator, urb_moltiplicator, NNN, barriers_lin_cost,
							Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, lorries_per_depo,
							cells_targets_df, strat_infr_power_table_df)
	
	elif scenario == "BAU_RankCI_nonlin":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_nonlin_fdist_depl(Proposed_Sites, Warehouses_Min, Warehouses_Max, Min_W_dim, Lookup_RurSubUrb,
							suburb_moltiplicator, urb_moltiplicator, NNN, barriers_lin_cost,
							Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, lorries_per_depo,
							cells_targets_df, strat_infr_rank_table_df)
	
	elif scenario == "BAU_RankCI_NoPoliceFire_nonlin":
		Proposed_Sites = Init.Generate_Proposed_Sites(Warehouse_Plan, Results_Folder, Lookup) # List of coordinates of proposed sites
		
		Cost_Fit, Dist_Fit = Eval.Calc_fcost_nonlin_fdist_depl(Proposed_Sites, Warehouses_Min, Warehouses_Max, Min_W_dim, Lookup_RurSubUrb,
							suburb_moltiplicator, urb_moltiplicator, NNN, barriers_lin_cost,
							Def_maint_lin_cost, P_pay, P_num, P_h, T_num, T_cost, lorries_per_depo,
							cells_targets_df, strat_infr_rankNoPoliceFire_table_df)
	
	
	else: print("***(Evaluate)WARNING: CHECK SCENARIO SELECTION")
	
	return Dist_Fit, Cost_Fit

Fitnesses = ['fdist', 'fcost']


def Track_Offspring():
    # Decorator function to save the solutions within the generators
    def decCheckBounds(func):
        def wrapCheckBounds(*args, **kargs):
            offsprings = func(*args, **kargs)
            # Append this generations offspring
            Gens.append(offsprings)
            for child in offsprings:
                # attach each individual solution to solution list. Allows the
				# demonstration of which solutions the Algorithm has investigated.
                Sols.append(child)
            return offsprings
        return wrapCheckBounds
    return decCheckBounds  


""" OPERATORS - Registers Operators and Constraint handlers for the GA
"""

## Evaluator
# Evaluation module - so takes the development plan
toolbox.register("evaluate", Evaluate)

## EVOLUTIONARY OPERATORS

# CROSSOVER
# Takes two points along the array and swaps the warehouses Between them.
# (String name for output text document)
Crossover = "tools.cxTwoPoint"
toolbox.register("mate", tools.cxTwoPoint)

# MUTATION
# mutShuffleIndexes moves the elements of the array around
Mutation = "tools.mutShuffleIndexes, indpb=0.1" # indpb - Independent probability for each attribute to be exchanged to another position.
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)

# SELECTION
Selection = "tools.selNSGA2"
toolbox.register("select", tools.selNSGA2)

Operators = ['Selection', Selection, 'Crossover', Crossover, 'Mutation', Mutation]

# CONSTRAINT HANDLING
# Using a decorator function in order to enforce a constraint of the operation.
# This handles the constraint on the total number of warehouses.
# So the module interrupts the selection phase and investigates the solutions selected.
# If they fail to exceed the minimum warehouse number or exceed the max warehouse number
# its deleted from the gene pool.   
# Moreover to this, each generation is saved to the Gen_list and each generated
# Solution is saved to a sol_list (for display purposes).

# Constraint to ensure the number of warehouses falls within the targets
# toolbox.decorate("select", Constraint.Check_TotWarehouse_Constraint(Warehouses_Max, Warehouses_Min, Results_Folder))
toolbox.decorate("select", Constraint.Check_Constraint_Select(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist), Track_Offspring())

toolbox.decorate("mate", Constraint.Check_Constraint_Mate_Mutate(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist))
toolbox.decorate("mutate", Constraint.Check_Constraint_Mate_Mutate(Warehouses_Max, Warehouses_Min, Results_Folder, Min_Warh_dist))

# toolbox.decorate("select", Track_Offspring())

## GA PARAMETERS

MU      = 1000	# Number of individuals to select for the next generation
NGEN    = 50    # Number of generations
LAMBDA  = 1000  # Number of children to produce at each generation

CXPB    = 0.6   # Probability of mating two individuals
MUTPB   = 0.3   # Probability of mutating an individual

GA_Parameters = ['Generations', NGEN, 'No of individuals to select', MU, 
                 'No of children to produce', LAMBDA, 'Crossover Probability',
                 CXPB, 'Mutation Probability', MUTPB]


def Genetic_Algorithm():    
    # Genetic Algorithm    
    print "Beginning GA operation"
    
    # Create initialised population
    print "Initialising"
    pop = toolbox.population(n=MU)
    
    # hof records a pareto front during the genetic algorithm
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", tools.mean)
    #stats.register("std", tools.std)
    stats.register("min", min)
    #stats.register("max", max)
    
    # Genetic algorithm with inputs
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats= stats, halloffame=hof)
                                                     
    return hof  


if __name__ == "__main__":
	# Returns the saved PO solution stored during the GA
	hof = Genetic_Algorithm()
	
	Complete_Solutions = copy(Sols)
	for PO in hof:
		Complete_Solutions.append(PO)
	
	if scenario == "BAU_discrete" or scenario == "BAU_linear_av" or scenario == "BAU_linear_urban_rural" or scenario == "BAU_linear_urban_sub_rural" or scenario == "BAU_Energy_nonlin" or scenario == "BAU_RankCI_nonlin" or scenario == "BAU_RankCI_NoPoliceFire_nonlin":
		
		# Update the results folder to the new directory specifically for this run
		Results_Folder = Output.New_Results_Folder(Results_Folder)    
		
		# Format the solutions so they are compatible with the output functions
		# Gives each a number as well as added the fitness values to from:
		# [ Sol_Num, Sites, Fitnesses]
		frmt_Complete_Solutions = Output.Format_Solutions(Complete_Solutions)
		
		# Extract the minimum and maximum performances for each objective
		# To allow for solutions to be normalised
		MinMax_list = Output.Normalise_MinMax(frmt_Complete_Solutions)
		
		# Normalise the formatted Solution list using the Min and Maxs for 
		# each objective function    
		Normalised_Solutions = Output.Normalise_Solutions(MinMax_list, frmt_Complete_Solutions)
			
		## OLD PLACE OF Output.Output_Run_Details
		
		# Extract all the Pareto fronts using the normalised solutions
		Output.Extract_ParetoFront_and_Plot(Normalised_Solutions, True, External_Results_Folder, Results_Folder, Data_Folder)
		
		# Extract all the Pareto fronts using the solutions retaining their true values.
		Output.Extract_ParetoFront_and_Plot(frmt_Complete_Solutions, False, External_Results_Folder, Results_Folder, Data_Folder)
		
		# Output a file detailing all the run parameters
		running_time_end_s = time.clock() # running time in seconds
		running_time_end_minutes = running_time_end_s/60 # running time in minutes
		run_time = str(int(running_time_end_minutes))
		Output.Output_Run_Details(External_Results_Folder, Results_Folder, Modules, Operators, Problem_Parameters, GA_Parameters, Fitnesses, run_time)

		# Create Shapefiles
		Output.Create_sol_shapefile(Results_Folder, External_Results_Folder)
		
		# Create Distance .csv files for statistics
		Output.Save_dist_W_T(External_Results_Folder, Results_Folder)
		
		# GENERATIONS OUTPUTS
		
		# Create a new array to hold the formatted generations
		frmt_Gens = []    
		for Gen in Gens:
			# For each generation, format it and append it to the frmt_Gens list
			frmt_Gens.append(Output.Format_Solutions(Gen))
		# 
		Output.Extract_Generation_Pareto_Fronts(frmt_Gens,MinMax_list, Results_Folder, Data_Folder, External_Results_Folder)
		
	elif scenario == "Disruption":
		# Update the results folder to the new directory specifically for this run
		Results_Folder = Output_Disr.New_Results_Folder(Results_Folder)    
		
		# Format the solutions so they are compatible with the output functions
		# Gives each a number as well as added the fitness values to from:
		# [ Sol_Num, Sites, Fitnesses]
		frmt_Complete_Solutions = Output_Disr.Format_Solutions(Complete_Solutions)
		
		# Extract the minimum and maximum performances for each objective
		# To allow for solutions to be normalised
		MinMax_list = Output_Disr.Normalise_MinMax(frmt_Complete_Solutions)
		
		# Normalise the formatted Solution list using the Min and Maxs for 
		# each objective function    
		Normalised_Solutions = Output_Disr.Normalise_Solutions(MinMax_list, frmt_Complete_Solutions)
			
		## OLD PLACE OF Output.Output_Run_Details
		
		# Extract all the Pareto fronts using the normalised solutions
		Output_Disr.Extract_ParetoFront_and_Plot(Normalised_Solutions, True, External_Results_Folder, Results_Folder, Data_Folder)
		
		# Extract all the Pareto fronts using the solutions retaining their true values.
		Output_Disr.Extract_ParetoFront_and_Plot(frmt_Complete_Solutions, False, External_Results_Folder, Results_Folder, Data_Folder)
		
		# Output a file detailing all the run parameters
		running_time_end_s = time.clock() # running time in seconds
		running_time_end_minutes = running_time_end_s/60 # running time in minutes
		run_time = str(int(running_time_end_minutes))
		Output_Disr.Output_Run_Details(External_Results_Folder, Results_Folder, Modules, Operators, Problem_Parameters, GA_Parameters, Fitnesses, run_time)

		# Create Shapefiles
		Output_Disr.Create_sol_shapefile(Results_Folder, External_Results_Folder)
		
		# Create Distance .csv files for statistics
		Output_Disr.Save_dist_W_T(External_Results_Folder, Results_Folder)
		
		# GENERATIONS OUTPUTS
		
		# Create a new array to hold the formatted generations
		frmt_Gens = []    
		for Gen in Gens:
			# For each generation, format it and append it to the frmt_Gens list
			frmt_Gens.append(Output_Disr.Format_Solutions(Gen))
		# 
		Output_Disr.Extract_Generation_Pareto_Fronts(frmt_Gens,MinMax_list, Results_Folder, Data_Folder, External_Results_Folder)
	
	else: print("***(Outputs)WARNING: CHECK SCENARIO SELECTION")
	
	end_time = time.asctime()
	
	print "END. end time = ", end_time
	print "Running time = ", int(running_time_end_minutes), " minutes"