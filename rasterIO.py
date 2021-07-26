''' Library of Geospatial raster functions to convert OSGEO.GDAL formats to/from Numpy masked arrays.

rasterIO
========

This library contains wrapper functions for GDAL Python I/O bindings, converting data to Numerical Python 
multi-dimensional array's in memory for processing. Subsequent, generated array's can be written to disk
in the standard Geospatial GeoTiff format.

Notes
-----
	Error checking - rasterIO contains minimal user-level error checking.

Supported Formats
-----------------
	Input: rasterIO supports reading any GDAL supported raster format

	Output: rasterIO generates GeoTiff files by default (this can be modified in the code).
		GeoTiffs are created with embedded binary header files containing geo information

Supported Datatypes
-------------------
	Raster IO supports Float32 and Int16 data types.
	The default datatype is Float32. Boolean datasets use Int16 datatypes.
	
NoDataValue
-----------
	If the input data has no recognisable NoDataValue (readable by GDAL) then the input NoDataValue
	is assumed to be 9999. This can be changed by manually specifying an input NoDataVal when calling readrasterbands().
	In accordance with GDAL the output data NoDataValue is 9999 or 9999.0 or can be manually set by when writrasterbands().
	When using unsigned integer data types the default output NoDataValue will be 0.

How to use documentation
------------------------
Documentation for module functions is provided as Python docstrings, accessible from an interactive Python terminal.
Within docstrings examples from an interactive Python console are identified using '>>>'. 
Further information is given to developers within the source code using '#' comment strings.
To view this text and a list of available functions call the Python in-built help command, specifying module name.


	>>> import rasterIO
	>>> help(rasterIO)
	...this text...
	
For help on a specific function call the Python in-built help command, specifying module.function.

	>>> import rasterIO
	>>> help(rasterIO.wkt2epsg)
		
		Help on function wkt2epsg in module rasterIO:

	wkt2epsg(wkt)
    		Accepts well known text of Projection/Coordinate Reference System and generates
    		EPSG code
	(END) 

How to access functions
-----------------------
To access functions, import the module to Python and call the desired function, assigning the output to a named variable.
Note that the primary input datatype (default) for all functions is either a Numpy array or a Numpy masked array. 
Within this module the term "raster" is used to signify a Numpy/Numpy masked array of raster values.
Use the rasterIO module to convert Numpy arrays to/from Geospatial raster data formats.

	>>> import rasterIO
	>>> band_number = 1
	>>> rasterdata = rasterIO.readrasterband(gdal_file_pointer, band_number)

Optional function arguments are shown in document strings in brackets [argument].
	
Dependencies
------------
Python 2.5 or greater
Numerical python (Numpy) 1.2.1 or greater (1.4.1 recommended).
	- Note that due to bugs in Numpy.ma module, Numpy 1.4.1 or greater is required to support masked arrays of integer values. 
		* See comments in reasrasterband() for more information.

License & Authors
-----------------
Copyright: Tom Holderness
Released under the Simplified BSD License (see LICENSE.txt).
'''
__version__ = "1.1.1"
#!/usr/bin/env python
# raster.py - module of raster handling functions using GDAL and NUMPY
# T.Holderness 19/05/2010
#
# ChangeLog
# 02/08/2010 - TH - Added NoDataVal handling for both read and write.
# 04/08/2010 - TH - Added PCRS support using WKT from source image metadata
# 			New function wkt2epsg
# 04/08/2010 - TH - Moved UHII function to avhrr.py
# 18/08/2010 - TH - Moved all AVHRR specific functions to avhrr.py
# Functions moved: "ndvi" and "lst"
# 23/08/2010 - TH - Moved all statistical functions to rasterStats.py.v1
# 23/08/2010 - TH - Moved all processing functions to rasterProcs.py.v1
# 23/08/2010 - TH - Moved this file (remaining functions) to rasterIO.py.v1
# rasterIO.py.v1
# 06/09/2010 - TH - readrasterband - Added masking to NaN values.
# 05/11/2010 - TH - opengdalraster - Added exception, raising IOError if opening broken raster.
# 10/11/2010 - TH - Added exceptions, raising errors where appropriate.
# 10/11/2010 - TH - Marked this as version 1.0.1 - working.
# 29/12/2010 - TH - Development version - multi-band raster images: 1.0.2
# 29/12/2010 - TH - Created new function "writerasterbands" for writing rasters with multiple layers.
# 29/12/2010 - TH - Changed function variable name "myraster" in "writerasterband()" to "rasterarray".
# 29/12/2010 - TH - Large change in write functions, created two new functions for better low level control of fileO.
# 29/12/2010 - TH - Created new function "newgdalraster()" - for creating new rasters on disk.
# 29/12/2010 - TH - Created new function "newrasterband()" - for creating for writing raster band to new file.
# 29/12/2010 - TH - Marked this version 1.0.2
# 12/01/2011 - TH - Updated changelog and comments
# 12/01/2011 - TH - Pushed tom beanstalk from eeepc.
# 12/01/2011 - TH - Fixed comment error after submit - pushed to beanstalk.
# 12/01/2011 - TH - Fixed comment error after submit (2) - pushed to beanstalk.
# 12/01/2011 - TH - Added catch for LOCAL_CS in "wkt2epsg".
# 13/01/2011 - TH - Experimental branch
# 17/01/2011 - TH - Changed all data handling to 8-bit-unsigned-int for Landsat data.
# 18/01/2011 - TH - Added data type dictionary, references GDAL GDT_'type' integers against Numpy equivalents.
# 18/01/2011 - TH - Added data type dictionary for struct.
# 18/01/2011 - TH - Changed input data value to 9999 (see 19/01/2011)
# 18/01/2011 - TH - Added dictionary data type look ups to read and writes.
# 19/01/2011 - TH - Added optional user specified input and ouput NoDataValues.
# 19/01/2011 - TH - Added 7-data types to dictionaries and changed names of dictionaries.
# 19/01/2011 - TH - Added test for Numpy 1.3.0 or greater to support Masked Array fill_values. 
# 19/01/2011 - TH - Changed test to Numpy 1.4.1 - min version required for integer masking.
# 19/01/2011 - TH - Cleaned up writerasterbands() comments.
# 27/01/2011 - TH - Marked this version ni experimental on 478 as 1.1.0
# 27/01/2011 - TH - Tweaked doc strings, including note about Numpy 1.4.1 and Masked integer arrays.
# 27/01/2011 - TH - Changed doc strings, input default no data value is now 9999 and user can specify input/output NoDataVal.
# 27/01/2011 - TH - Changed masked_values, masked_equal, masked_invalid function calls to include 'copy=False'. Thanks to Ian Martin for spotting this.
# 03/02/2011 - TH - Experiemental branch
# 03/02/2011 - TH - Changed masking input and output handling.
# 03/02/2011 - TH - Added catch in epsg2wkt to catch empty strings, now returns 0 which forces a gdal error.
# 03/02/2011 - TH - Tested input/output with new masking and numpy array flag.
# 03/02/2011 - TH - Cleaned up legacy code in newrasterband.
# 03/02/2011 - TH - Marked this version as 1.1.1 in experimental.
# 03/20/2011 - TH - Added optional NoDataVal argumen to writerasterband() function.
# 03/02/2011 - TH - Fixed bug in epsg2wkt exception handling tests. 
# 03/02/2011 - TH - Tested using AVHRR_Himalaya.py. Will now merge to master.
# 03/02/2011 - TH - Merged experimental to master with no conflicts. Will now rename to rasterIO.py (droppping '.v1').
# 03/02/2011 - TH - Added check for uint8 type when writing out NoDataValue (9999 = 15 @ uint8!). Updated docs.

import os, sys, struct
import numpy as np
import numpy.ma as ma
import osgeo.osr as osr
import osgeo.gdal as gdal
from osgeo.gdalconst import *

# Data type dictionaries - references from GDT's to other Python types.
# GDT -> Numpy
gdt2npy	=	{
			1:'uint8',
			2:'uint16',
			3:'int16',
			4:'uint32',
			5:'int32',
			6:'float32',
			7:'float64'
		
		}	
# Numpy -> GDT
npy2gdt = 	{	
			'uint8':1,
			'uint16':2,
			'int16':3,
			'uint32':4,
			'int32':5,	
			'float32':6,
			'float64':7
			
		}
		
# GDT -> Struct
gdt2struct =	{	
			1:'B',
			2:'H',
			3:'h',
			4:'I',
			5:'i',
			6:'f',
			7:'d'
		}


# function to open GDAL raster dataset
def opengdalraster(fname):
	'''Accepts gdal compatible file on disk and returns gdal pointer.'''
	dataset = gdal.Open( fname, GA_ReadOnly)
	if dataset != None:
		return dataset
	else: 
		raise IOError
		
# function to read raster image metadata
def readrastermeta(dataset):
	'''Accepts GDAL raster dataset and returns, gdal_driver, XSize, YSize, projection info(well known text), geotranslation data.'''
		# get GDAL driver
	driver_short = dataset.GetDriver().ShortName
	driver_long = dataset.GetDriver().LongName
		# get projection	
	proj_wkt = dataset.GetProjection()
		# get geotransforamtion parameters
	geotransform = dataset.GetGeoTransform()
		# geotransform[0] = top left x
		# geotransform[1] = w-e pixel resolution
		# geotransform[2] = rotation, 0 if image is "north up"
		# geotransform[3] = top left y
		# geotransform[4] = rotation, 0 if image is "north up"
		# geotransform[5] = n-s picel resolution
	XSize = dataset.RasterXSize
	YSize = dataset.RasterYSize
	
	return driver_short, XSize, YSize, proj_wkt, geotransform

# function to read a band from a dat# apply NoDataValue masking.aset
def readrasterband(dataset, aband, NoDataVal=None, masked=True):
	'''Accepts GDAL raster dataset and band number, returns Numpy 2D-array.'''
	if dataset.RasterCount >= aband:		
		# Get one band
		band = dataset.GetRasterBand(aband)
		# test for user specified input NoDataValue
		if NoDataVal is None:
			# test for band specified NoDataValue
			if band.GetNoDataValue() != None:
				NoDataVal = band.GetNoDataValue()
			else:
				# else set NoDataValue to be 9999.
				NoDataVal = 9999
		# set NoDataVal for the band (not strictly needed, but good practice if we call the band later).		
		band.SetNoDataValue(NoDataVal)
		# create blank array (full of 0's) to hold extracted data [note Y,X format], get data type from dictionary.
		datarray = np.zeros( ( band.YSize,band.XSize ), gdt2npy[band.DataType] )
			# create loop based on YAxis (i.e. num rows)
		for i in range(band.YSize):
			# read lines of band	
			scanline = band.ReadRaster( 0, i, band.XSize, 1, band.XSize, 1, band.DataType)
			# unpack from binary representation
			tuple_of_vals = struct.unpack(gdt2struct[band.DataType] * band.XSize, scanline)
			# tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
			# add tuple to image array line by line	
			datarray[i,:] = tuple_of_vals
		
		# check if masked=True
		if masked is True:
			# check if data type is int or float using dictionary for numeric test.
			if npy2gdt[datarray.dtype.name] <= 5:
				# data is integer use masked_equal
				# apply NoDataValue masking.
				dataraster = ma.masked_equal(datarray, NoDataVal, copy=False)
				# apply invalid data masking
				dataraster = ma.masked_invalid(dataraster, copy=False)
				return dataraster				
			else:
				# data is float use masked_values
				dataraster = ma.masked_values(datarray, NoDataVal, copy=False)
				# finaly apply mask for NaN values
				dataraster = ma.masked_invalid(dataraster, copy=False)
				# return array (raster)
				return dataraster
		else:
			# user wants numpy array, no masking.
			return datarray
	else:
		raise TypeError	

# function to create new (empty) raster file on disk.
def newgdalraster(outfile, format, XSize, YSize, geotrans, epsg, num_bands, gdal_dtype ):
	'''Accepts file_path, format, X, Y, geotransformation, epsg, number_of_bands, gdal_datatype and returns gdal pointer to new file.

	This is a lower level function that allows users to control data output stream directly, use for specialist cases such as varying band data types or memory limited read-write situations.
	Note that users should not forget to close file once data output is complete (dataset = None).'''
	# get driver and driver properties	
	driver = gdal.GetDriverByName( format )
	metadata  = driver.GetMetadata()
	# check that specified driver has gdal create method and go create	
	if metadata.has_key(gdal.DCAP_CREATE) and metadata[gdal.DCAP_CREATE] =='YES':	
		# Create file
		dst_ds = driver.Create( outfile, XSize, YSize, num_bands, gdal_dtype )
		# define "srs" as a home for coordinate system parameters
		srs = osr.SpatialReference()
		# import the standard EPSG ProjCRS
		srs.ImportFromEPSG( epsg )
		# apply the geotransformation parameters
		#print geotrans
		dst_ds.SetGeoTransform( geotrans )
		# export these features to embedded well Known Text in the GeoTiff
		dst_ds.SetProjection( srs.ExportToWkt() )
		return dst_ds
	# catch error if no write method for format specified
	else:
		#print 'Error, GDAL %s driver does not support Create() method.' % outformat
		raise TypeError

def newrasterband(dst_ds, rasterarray, band_num, NoDataVal=None):
	'''Accepts a GDAL dataset pointer, rasterarray, band number, [NoDataValue], and creates new band in file.'''
	# check for user output NoDataVal and set accordingly
	if NoDataVal is None:
		NoDataVal = 9999
	if npy2gdt[rasterarray[0].dtype.name] == 1 and NoDataVal > 255:
		NoDataVal = 0	
	dst_ds.GetRasterBand(band_num).SetNoDataValue(NoDataVal)
	# check for mask and apply NoDataValue to values in numpy array
	if ma.isMaskedArray(rasterarray) is True:
		# create a numpy view on the masked array	
		output = np.array(rasterarray, copy=False)
		# check if maskedarray has valid mask and apply to numpy array using binary indexing.
		if rasterarray.mask is not ma.nomask:
			output[rasterarray.mask] = NoDataVal
		# write out numpy array with masking
		dst_ds.GetRasterBand(band_num).WriteArray ( output )
	else:
	# input array is numpy already, write array to band in file
		dst_ds.GetRasterBand(band_num).WriteArray ( rasterarray )

# create function to write GeoTiff raster from NumPy n-dimensional array
def writerasterbands(outfile, format, XSize, YSize, geotrans, epsg, NoDataVal=None, *rasterarrays ):
	''' Accepts raster(s) in Numpy 2D-array, outputfile string, format and geotranslation metadata and writes to file on disk'''
	# get number of bands
	num_bands = len(rasterarrays)	
	# create new raster
	dst_ds = newgdalraster(outfile, format, XSize, YSize, geotrans, epsg, num_bands, npy2gdt[rasterarrays[0].dtype.name])
	# add raster data from raster arrays
	band_num = 1 # band counter
	for band in rasterarrays:
		newrasterband(dst_ds, band, band_num, NoDataVal)
		band_num += 1
	# close output and flush cache to disk
	dst_ds= None

# legacy function to write GeoTiff raster from NumPy n-dimensional array - use writerasterbands instead
def writerasterband(rasterarray, outfile, format, aXSize, aYSize, geotrans, epsg, NoDataVal=None):
	''' Legacy function for backwards compatability with older scripts. Use writerasterbands instead.

	Accepts raster in Numpy 2D-array, outputfile string, format and geotranslation metadata and writes to file on disk'''
	writerasterbands(outfile, format, aXSize, aYSize, geotrans, epsg, NoDataVal, rasterarray)
		
# function to get Authority (e.g. EPSG) code from well known text
def wkt2epsg(wkt):
	'''Accepts well known text of Projection/Coordinate Reference System and generates EPSG code'''
	if wkt is not None:
		if wkt == '':
			return 0
		else:
			srs = osr.SpatialReference(wkt)
			if (srs.IsProjected()):
				return int(srs.GetAuthorityCode("PROJCS"))
			elif (srs.IsLocal()):
				return 0
			else:
			 	return int(srs.GetAuthorityCode("GEOGCS"))
	else:
		raise TypeError	 
