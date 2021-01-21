# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:05:02 2015
Last update: 17 Apr 2017

@author: Jessica Scheick (@JessicaS11)
LICENSE AND USE:

SYNTAX:
From terminal (with gdal_opencv environment)
python icebergdelineation.py [full path of directory of directories of LS scenes to be processed] [full path of directory for saving results (cannot be the same as the processing directory)]

ex: "python icebergdelineation.py /Users/jessica/DiskoBayIcebergs/Development2_restart2017/code_test/ /Users/jessica/DiskoBayIcebergs/Development2_restart2017/results"
    
-required changes for different fjords (most have ## comments before them):
    --MergeImages -- 'utm_zone' to project images into, where needed ('22N' for Disko Bay)
    --ProcessScene(Land mask) -- vector ROI and output raster file[name]s for that region
    --ProcessScene(cloud mask) -- 'run_type' needs a descriptive string for saving files
    This string will be used to create raster 'final_name' and shapefile 'new_shape_name' later in fn
    --aoi for entire area (vs portion visible in each scene)


-required files
    --vector file of the region of interest, in the directory with the Landsat scene directories
    --classifier model
-created files
    --if two scenes are merged, the merged band for that scene is saved in the LS scene dir
    --raster of the vector file of the region of interest. Note this file is replaced
    for each scene, so the mask there will be for the last scene run
    --a raster for each scene date, thresholded and with masks applied (Nodata value is 0)
    --a shapefile for each scene date, containing polygons of icebergs

DESCRIPTION:
main program to run iceberg delineation algorithm, including:
-combine images to cover entire study area (includes reprojection, if needed)
-create and apply a land mask (from shape file)
-create and apply a cloud mask (several options)
-threshold to find icebergs
-remove icebergs that are touching the scene edges or land boundaries,
compile information about the icebergs, and add to a shapefile attribute table


NOTES:
Cloud masks. The current cloud mask is machine learning based (a logistic regression
classifier, which must be regenerated if any of the versions of the libraries
change). The other options are no cloud mask or threshold based cloud mask 
(requires [un]commenting code). The threshold option will be excluded from further development.

-calculation of the extent mask (since each LS band covers slightly different extents)
is reliant on either being integer input with 0=no data or a non-integer input with -9999
being no data (as in reflectance or bright temp values)

"""

import sys
import os
import csv
from osgeo import gdal, ogr, gdal_array, osr
import numpy as np
from scipy import ndimage
from raster import rasterfile as rasterfile

'''
def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    raw_input("Press key to exit.")
    sys.exit(-1)

sys.excepthook = show_exception_and_exit
'''
######## FUNCTIONS ########
#don't need unless I want to set any class variable (so maybe a path at some point?)
#def __init__ (self, variables):

#determine band numbers and data type based on sensor
def getBandNum(spacecraft, band_str):
    
    if spacecraft == "LANDSAT_7":
        dtype = 'Byte'
        if band_str == "pan":
            band = 8
        elif band_str == "vred":
            band = 3
        elif band_str == "nir":
            band = 4
        elif band_str == "swir1":
            band = 5
        elif band_str == "swir2":
            band = 7
        elif band_str == "thermal":
            band = 6
        elif band_str == "vred_refl":
            band = '3_refl'
        elif band_str == "nir_refl":
            band = '4_refl'
        elif band_str == "swir1_refl":
            band = '5_refl'
        elif band_str == "swir2_refl":
            band = '7_refl'
        elif band_str == "pan_refl":
            band = '8_refl' 
        elif band_str == "bright_temp":
            band = '6_bt'        
    elif spacecraft == "LANDSAT_8":
        dtype = 'UInt16'
        if band_str == "pan":
            band = 8
        elif band_str == "vred":
            band = 4
        elif band_str == "nir":
            band = 5
        elif band_str == "swir1":
            band = 6
        elif band_str == "swir2":
            band = 7
        elif band_str == "thermal":
            band = 11
        elif band_str == "vred_refl":
            band = '4_refl'
        elif band_str == "nir_refl":
            band = '5_refl'
        elif band_str == "swir1_refl":
            band = '6_refl'
        elif band_str == "swir2_refl":
            band = '7_refl'
        elif band_str == "pan_refl":
            band = '8_refl'
        elif band_str == "bright_temp":
            band = '11_bt'
    
    if 'refl' in band_str or 'temp' in band_str:
        refl_flag = True
        dtype = 'float32'
    else:
        refl_flag = False
    
    return dtype, band, refl_flag
    
#updates the existing data extent array with the data limits of another array. Note the inputs are two arrays;
#these must be the same size and spatial coverage. Also, the no data value is assumed to be 0 with 
#no input values <0 (so all positive values in inputs) for integer data or no data of -9999 for float
#inputs, which are assumed to have data everywhere they don't =-9999 (including 0)
def updateExtentArray(extent_array, *args): #args are the arrays needing to be added to the extent mask
    
#    print range(0, len(args))
    extent_array_32 = extent_array.astype(np.int32)
    for arraynum in range(0,len(args)):
#        print args[arraynum]
#        print args[arraynum].dtype
#        print np.min(args[arraynum])
        
        add_array_mask = np.copy(args[arraynum])        
#        if np.min(add_array_mask)==-9999:
##            print 'got into loop'
#            add_array_mask[add_array_mask!=-9999] = 1 #this step is so that when reflectance and bright temp bands come in the pixels with values are turned back into integers for creating the mask.
            #It works reliant on the fact that -9999 is set as the no data value during conversion
        nodata = np.min(add_array_mask)
#        print 'nodata is ' + str(nodata) + ' of type '
#        print nodata.dtype
        add_array_mask[add_array_mask!=nodata] = 1
        test_equiv_bool = np.isclose(add_array_mask, nodata)
#        print test_equiv_bool
        add_array_mask[np.invert(test_equiv_bool)] = 1
        add_array_mask = add_array_mask.astype(np.int32)
        add_array_mask[test_equiv_bool] = -9999    #formerly 0, not nodata minimum value variable
        
#        print add_array_mask
#        print extent_array_32.dtype
#        print add_array_mask.dtype
           
        extent_array_32 = extent_array_32 + add_array_mask
#        print extent_array_32.dtype
#        print extent_array_32
    
    #note <=0 doesn't work below if you have any input values with negative numbers and an unspecified (or non -9999) no data value
    extent_array[extent_array_32<=0] = 0

    extent_array[extent_array!=0] = 1 
#    print extent_array

    test_equiv_bool = None
    extent_array_32 = None
    add_array_mask = None

    return extent_array

#generates a classified image using the thermal, red, NIR, and SWIR2 bands (thermal
#is rescaled, all are reflectances/bright temps). The classifier has been trained externally and is imported. Then
#holes in the clouds are filled in (where they were classified as ice; water and
#ice under cloud stays) and a cloud mask resampled to match the pan resolution/extent returned    
def GetCloudMask(path, folder, merge_flag, spacecraft, dtype, ds_pan, *args):
#cloud_mask_array = GetCloudMask(path, fold, merge_flag, spacecraft, dtype)
#    from sklearn.preprocessing import MinMaxScaler    
#    from sklearn.linear_model import LogisticRegression
    import pickle
    from scipy import ndimage as ndi 
    print 'Using trained classifier mask with bands thermal and red, NIR, and SWIR2 reflectance'

    #opening needed bands (including merged for dates with two scenes)
    if merge_flag == True:
        fold2 = args[0]
        ds_vred, vred_array = getRasterArray(path, folder, 'vred_refl', merge_flag, spacecraft, fold2)
        ds_nir, nir_array = getRasterArray(path, folder, 'nir_refl', merge_flag, spacecraft, fold2)
        ds_swir2, swir2_array = getRasterArray(path, folder, 'swir2_refl', merge_flag, spacecraft, fold2)
        ds_therm, therm_array = getRasterArray(path, folder, 'bright_temp', merge_flag, spacecraft, fold2)
        
    else:
        ds_vred, vred_array = getRasterArray(path, folder, 'vred_refl', merge_flag, spacecraft)
        ds_nir, nir_array = getRasterArray(path, folder, 'nir_refl', merge_flag, spacecraft)
        ds_swir2, swir2_array = getRasterArray(path, folder, 'swir2_refl', merge_flag, spacecraft)
        ds_therm, therm_array = getRasterArray(path, folder, 'bright_temp', merge_flag, spacecraft)  
    
    #compute extent array for bands used. It is important to do this before the rescaling of the brightness temp
    #or there are negative values and I think no -9999 no data value so the updateExtentArray function would need to be different
    cloud_ext_array = np.ones_like(vred_array)
    cloud_ext_array = updateExtentArray(cloud_ext_array, vred_array, nir_array, swir2_array, therm_array)

    #rescale brightness temp values (open here, apply below)
    print 'NOTICE: If update version of sklearn, rerun brightness temp scaling and model!!!'
    scaler_fn = path+'/brighttemp_scaling.sav'
    scaler = pickle.load(open(scaler_fn, 'rb'))
    
    #opening and applying model
    model_fn = path+'/fitted_regression_model.sav'
    mdl = pickle.load(open(model_fn, 'rb'))
    predicted_array = np.zeros(vred_array.shape)
    print 'Predicting values using statistical model...'
    col = 0
    for col in range(0, len(vred_array[0])-1):   
        therm_array[:,col] = scaler.transform(therm_array[:,col].reshape(-1,1)).reshape(1,-1)
        xvals = np.column_stack((therm_array[:,col], vred_array[:,col], nir_array[:,col], swir2_array[:,col]))      
    
#        if col%1000 == 0:
#            print 'on col number ' + str(col)
        
        predicted_array[:,col] = mdl.predict(xvals)
#        probs = mdl.predict_proba(xvals)
        
#       raster.rasterfile(bblue_file, predicted_file, predicted_array, gdal.GDT_Byte)
    
    #turning model results into a cloud mask
    #print 'we have a cloud mask prediction array'
    cld_msk_a = (predicted_array==5)
    fill_cld_msk = ndi.morphology.binary_fill_holes(cld_msk_a, structure=np.ones((10,10)))
    #morphological opening to smooth cloud mask (and fill in any remaining small holes)
    #note default structure is a cross shape, so direct neighbors only (no diagonals)
    fill_cld_msk = ndi.morphology.binary_opening(fill_cld_msk)
    add_cld_msk = (fill_cld_msk*5 + predicted_array).astype('int')

    zeros = np.zeros_like(predicted_array, dtype='uint8')
    ones = np.ones_like(predicted_array, dtype ='uint8')
    cloud_mask_array = np.where(np.logical_or(add_cld_msk==6, add_cld_msk==10), zeros, ones) #opposite of saved masks 2/16/18

    #make cloud mask with 0=no data, 1=include, 255=cloud
    cloud_mask_array[cloud_mask_array==0] = 255    

    #apply cloud mask extent    
    cloud_mask_array = cloud_mask_array * cloud_ext_array

#    cloud_ext_fn = save_path + '/' + folder[dtst:dtend] + '_cloudmask_extent.TIF'
#    rasterfile(ds_vred.GetFileList()[0], cloud_ext_fn, cloud_ext_array, gdal.GDT_Byte)
    
    #write cloud mask array to file, resample with gdalwarp to match pan band extent and resolution, and reopen
    cloud_mask_fn = save_path + '/' + folder[dtst:dtend] + '_cloudmask.TIF'
    rasterfile(ds_vred.GetFileList()[0], cloud_mask_fn, cloud_mask_array, gdal.GDT_Byte)
    cloud_mask_fnout = save_path + '/' + folder[dtst:dtend] + '_cloudmask_resampled.TIF'

    pan_geo = ds_pan.GetGeoTransform()
    ulx = pan_geo[0] #where ul = upper left and lr = lower right
    uly = pan_geo[3]
    lrx = ulx + pan_geo[1]*ds_pan.RasterXSize
    lry = uly + pan_geo[5]*ds_pan.RasterYSize
    
    xmin = min(ulx,lrx)
    ymin = min(uly,lry)
    xmax = max(ulx,lrx)
    ymax = max(uly,lry)
    
    cmd = "gdalwarp -te " + str(xmin) +" "+ str(ymin) +" "+ str(xmax) +" "+ str(ymax) + " -tr 15 15 -overwrite " + cloud_mask_fn + ' ' + cloud_mask_fnout
    os.system(cmd)
    
    cloud_ds = gdal.Open(cloud_mask_fnout)
    cloud_mask_array = cloud_ds.GetRasterBand(1).ReadAsArray()
    #consider adding full path to gdalwarp in cmd, above (and for other, similar cases)
    #cloud_mask_array = gdal.ReprojectImage() looks like an inline alternative, but there are no docs for it; similar for gdal.AutoCreateWarpedVRT
    
    #clean up
    cmd = 'rm ' + cloud_mask_fn
    os.system(cmd)
    cmd = None
    cloud_ds = None
    ds_vred = None
    ds_nir = None
    ds_swir2 = None
    ds_therm = None
    vred_array = None
    nir_array = None
    swir2_array = None
    therm_array = None
    cld_msk_a = None
    fill_cld_msk = None
    add_cld_msk = None
   
    #return np.ones_like(cloud_mask_array)-cloud_mask_array
    return cloud_mask_array
    
#ThresCloudMask will create a binary cloud mask based on thresholds applied to
#nir/swir1 and swir1 reflectance
#Note: code within this section is incomplete for broad application (see 2nd print statement, below)
def ThreshCloudMask(path, folder, merge_flag, spacecraft, dtype, ds_pan, *args):
#cloud_mask_array = GetCloudMask(path, fold, merge_flag, spacecraft, dtype)
    print 'Using thresholding cloud mask with bands nir and swir1 (DN and reflectance)'
    
#   #opening needed bands and calculating reflectance files for swir1
#    dtype, band, refl_flag = getBandNum(spacecraft, 'swir1')
#    cmd = "python /Users/jessica/Scripts/python_geotif_fns/Landsat_TOARefl.py " + path+'/'+folder+'/' + " " + folder+"_MTL.txt 'ETM+ Thuillier' false " + str(band)
#    os.system(cmd)
#    cmd = None
    
    if merge_flag == True:
        fold2 = args[0]
#        ds_vred, vred_array = getRasterArray(path, folder, 'vred', merge_flag, spacecraft, fold2)
        ds_nir, nir_array = getRasterArray(path, folder, 'nir', merge_flag, spacecraft, fold2)
        ds_swir1, swir1_array = getRasterArray(path, folder, 'swir1', merge_flag, spacecraft, fold2)
        ds_swir1_refl, swir1_refl_array = getRasterArray(path, folder, 'swir1_refl', merge_flag, spacecraft, fold2)
               
    else:
        ds_nir, nir_array = getRasterArray(path, folder, 'nir', merge_flag, spacecraft)
        ds_swir1, swir1_array = getRasterArray(path, folder, 'swir1', merge_flag, spacecraft)
        ds_swir1_refl, swir1_refl_array = getRasterArray(path, folder, 'swir1_refl', merge_flag, spacecraft)
    
    is_zero = (swir1_array==0)
    swir1_array[is_zero] = 1 #because in the denominator below
    
    #NIR/SWIR1 threshold   
    ratio = np.true_divide(nir_array, swir1_array)
    ratio[is_zero] = 0
#    ratio = ndimage.zoom(ratio, 2, order=0)
#    ratio = np.delete(ratio, 0, 0) #removing extra row and column resulting from zoom. probably not the best method, as definitely offsets pixels
#    ratio = np.delete(ratio, 0, 1)

#    print 'band_min = ', np.amin(ratio)
#    print 'band_max = ', np.amax(ratio)
#    print 'band_mean = ', np.mean(ratio)
     #cloud is ratio < 1; non-cloud is ratio >=1  
#    cloud_mask_ind1 = ratio[ratio < 1]
    
    #VRED/SWIR1 threshold -->
    '''a vred/swir1 threshold is the method copied from the 2014 original code. See notes in older versions
    of this script about renormalization not seeming to have much, if any, impact on results.
    After additional investigation (and trying to reproduce a figure in my Sept 2015 Space Grant report),
    it seems that by then I was not using this ratio but a threshold based on the swir1-reflectance, which
    I have in my notes as part of a decision tree (Nov 2014) and later in a writeout of the algorithm
    steps (Oct 2015). This is the code that now remains below.
    '''

    #SWIR1 reflectance threshold
    ratio2 = swir1_refl_array
    
#    ratio2 = ndimage.zoom(ratio2, 2, order=0)
#    ratio2 = np.delete(ratio2, 0, 0)
#    ratio2 = np.delete(ratio2, 0, 1)

#    print 'band_min = ', np.amin(ratio2)
#    print 'band_max = ', np.amax(ratio2)
#    print 'band_mean = ', np.mean(ratio2)
    #cloud is reflectance > 0.2; non-cloud is reflectance <= 0.2
#    cloud_mask_ind2 = ratio2[ratio2 < 40]

    #combine two thresholds into one mask index
    zeros = np.zeros_like(ratio, dtype='uint8')
    ones = np.ones_like(ratio, dtype ='uint8')
    cloud_mask_array = np.where(np.logical_or(ratio < 1, ratio2 > 0.2), zeros, ones)
    #cld_msk_a = np.where(np.logical_or(ratio < 1, ratio2 > 0.2), zeros, ones)
##    cloud_mask_array = np.where(np.logical_or(ratio < 1, ratio2 < 40), zeros, ones)
    #I am not applying this fill-holes function to this because I did not do it for the method comparison runs and
    #the threshold mask is technically a different mask than the machine learning one, so it doesn't necessarily need to have all the same processing
    #also, it would need to be applied on the inverse of cld_msk_a in order to fill in clouds rather than non-clouds
    #cloud_mask_array = ndi.morphology.binary_refill_holes(cld_msk_a, structure=np.ones((10,10)))
    #morphological opening to smooth cloud mask (and fill in small holes)
    cloud_mask_array = ndimage.morphology.binary_opening(cloud_mask_array).astype(cloud_mask_array.dtype)
#
#    #make cloud mask with 0=no data, 1=include, 255=cloud
    cloud_mask_array[cloud_mask_array==0] = 255   
    #generate and apply cloudmask extent array
    swir1_array[is_zero] = 0 #turn them back into zeros so they work for the extent array!
    cloud_ext_array = np.ones_like(nir_array)
    cloud_ext_array = updateExtentArray(cloud_ext_array, nir_array, swir1_array) #don't need swir1_refl because same extent as swir1
    cloud_mask_array = cloud_mask_array * cloud_ext_array
     
    cloud_ext_fn = save_path + '/' + folder[dtst:dtend] + '_threshold_cloudmask_extent.TIF'
#    rasterfile(ds_nir.GetFileList()[0], cloud_ext_fn, cloud_ext_array, gdal.GDT_Byte)    

    #write cloud mask array to file, resample with gdalwarp to match pan band extent and resolution, and reopen
    cloud_mask_fn = save_path + '/' + folder[dtst:dtend] + '_thresh_cloudmask.TIF'
    rasterfile(ds_nir.GetFileList()[0], cloud_mask_fn, cloud_mask_array, gdal.GDT_Byte)
    cloud_mask_fnout = save_path + '/' + folder[dtst:dtend] + '_thresh_cloudmask_resampled.TIF'

    pan_geo = ds_pan.GetGeoTransform()
    ulx = pan_geo[0] #where ul = upper left and lr = lower right
    uly = pan_geo[3]
    lrx = ulx + pan_geo[1]*ds_pan.RasterXSize
    lry = uly + pan_geo[5]*ds_pan.RasterYSize
    
    xmin = min(ulx,lrx)
    ymin = min(uly,lry)
    xmax = max(ulx,lrx)
    ymax = max(uly,lry)
    
    cmd = "gdalwarp -te " + str(xmin) +" "+ str(ymin) +" "+ str(xmax) +" "+ str(ymax) + " -tr 15 15 -overwrite " + cloud_mask_fn + ' ' + cloud_mask_fnout
    os.system(cmd)
    
    cloud_ds = gdal.Open(cloud_mask_fnout)
    cloud_mask_array = cloud_ds.GetRasterBand(1).ReadAsArray()
    #consider adding full path to gdalwarp in cmd, above (and for other, similar cases)
    #cloud_mask_array = gdal.ReprojectImage() looks like an inline alternative, but there are no docs for it; similar for gdal.AutoCreateWarpedVRT

    #clean up
    cmd = 'rm ' + cloud_mask_fn
    os.system(cmd)
    cmd = None
#    ds_vred = None
#    vred_array = None
    ds_nir = None
    nir_array = None
    ds_swir1 = None
    swir1_array = None
    ds_swir1_refl = None
    swir1_refl_array = None
    
    return cloud_mask_array
    
#BergInfo will take the shapefile of iceberg clusters and clean it up (eg remove holes
#in polygons) and calculate useful information. When land/scene boundaries are specified,
#this also removes polygons that border these boundaries
#any fields in the incoming layer will be lost unless appropriate code is uncommented
#CODE IMPROVEMENT: look into skimage.measure.grid_points_in_poly to test for grid points within a polygon
def BergInfo(orig_shpfile, new_shpfile, *args):
    #open shapefile and get layer/srs info
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(orig_shpfile, 0 )
    inLayer = ds.GetLayer()
    srs = osr.SpatialReference()
    srs.ImportFromWkt( inLayer.GetSpatialRef().ExportToWkt() )
    
    #get geometries of land and scene extents
    if args:
        land_ext_ds = driver.Open(args[1], 1)
        land_extLayer = land_ext_ds.GetLayer()
        feature = land_extLayer[0]
        land_geom = feature.GetGeometryRef()
    
    
        scene_ext_ds = driver.Open(args[0], 1)
        scene_extLayer = scene_ext_ds.GetLayer()

        idx = None
        max_area = None
        
        for x, feat in enumerate(scene_extLayer):
            area = feat.GetGeometryRef().GetArea()
            if max_area is None or area > max_area:
                max_area = area
                idx = x           
        max_area=None
        
        feature2 = scene_extLayer[idx]
        scene_geom = feature2.GetGeometryRef().Buffer(-8,30)
        simp_scene_geom = scene_geom.Simplify(2)

#        cloud_border_ds = driver.Open(args[2], 1)
#        cloud_border_extLayer = cloud_border_ds.GetLayer()
#
#        idx = None
#        max_area = None
#        
#        for x, feat in enumerate(scene_extLayer):
#            area = feat.GetGeometryRef().GetArea()
#            if max_area is None or area > max_area:
#                max_area = area
#                idx = x           
#        max_area=None
#        
#        feature3 = scene_extLayer[idx]
#        scene_geom = feature2.GetGeometryRef().Buffer(-8,30)
#        simp_scene_geom = scene_geom.Simplify(2)


    # Remove output shapefile if it already exists
    if os.path.exists(new_shpfile):
        driver.DeleteDataSource(new_shpfile)
    
    #create new output shapefile
    out_ds = driver.CreateDataSource(new_shpfile)
    out_lyr_name = os.path.splitext( os.path.split( new_shpfile )[1] )[0]
    #print 'out layer name is ' + out_lyr_name
    outLayer = out_ds.CreateLayer( out_lyr_name, geom_type=ogr.wkbMultiPolygon, srs = srs)

    #add fields to shapefile
    field_names = ['x','y','area','units']
    field_types = ['real','real','int','string']
    field_length = [20,20,20,30]
    field_precision = [8,8,0,0]
    
    for i in range(0, len(field_names)):
        if field_types[i] == 'string':
            f_type = ogr.OFTString
        elif field_types[i] == 'real':
            f_type = ogr.OFTReal
        
        fldDef = ogr.FieldDefn(field_names[i], f_type)
        fldDef.SetWidth(field_length[i])
        fldDef.SetPrecision(field_precision[i])
        
        #adds attribute fields to layer
        outLayer.CreateField(fldDef)

    # Get the output Layer's Feature Definition
    outLayerDefn = outLayer.GetLayerDefn()
    
    total_ice_area = 0

    '''
    #Add features to the output layer (try with ogr2ogr to remove bordering polygons)
    for inFeature in inLayer:
        geometry = inFeature.GetGeometryRef()
       
        # Create output Feature
        outFeature = ogr.Feature(outLayerDefn)

        # Add field values from input Layer
        #for i in range(0, outLayerDefn.GetFieldCount()):
        #    fieldDefn = outLayerDefn.GetFieldDefn(i)
        #    outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(),
        #        inFeature.GetField(i))

        # Set geometry (and remove inner rings)
        geomWkt = geometry.ExportToWkt()
        newGeomWkt, trash = geomWkt.split(')', 1)
        newGeomWkt = newGeomWkt + '))'
        newGeom = ogr.CreateGeometryFromWkt(newGeomWkt)
        outFeature.SetGeometry(newGeom)
        
        #calculate centroid and add to attribute table
        centroid = newGeom.Centroid()
        centroid_text = centroid.ExportToWkt()
        trash, cent_x, cent_y = centroid_text.split(' ', 2)
        outFeature.SetField(field_names[0], cent_x[1:])
        outFeature.SetField(field_names[1], cent_y[:-1])
        
        #calculate area and add to attribute table
        area = newGeom.GetArea()
        outFeature.SetField(field_names[2], area)
        total_ice_area += area
        
        #area units
        outFeature.SetField(field_names[3], 'm^2')
        
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
        
        #remove features touching the scene or ROI border
        #I was trying to implement this method on 12/18/2017 based on a stackexchange response I got to an old post on a faster way to do this.
        #so far, I don't know how to input multiple directories (since the needed input files aren't in the same one), plus I'm getting an unrecognized token error when I try to run any part of it in terminal
#        cmd = "ogr2ogr -f 'ESRI Shapefile' -dialect sqlite -sql \
#        'SELECT a.* FROM bergs a, AOI_buffered100 b WHERE ST_Intersects(a.geometry, b.geometry)' \
#        -overwrite " + out_lyr_name + " " + sub_directory
        #in-progress terminal command: (single quotes in sql statement are because layer name begins with a number)
        #ogr2ogr -f 'ESRI Shapefile' -dialect sqlite -sql "SELECT a.* FROM '2015146_bergs_threshcloud', AOI_nofjords_buffered100 b WHERE ST_Intersects(a.geometry, b.geometry)" bergroitest.shp /Users/jessica/DiskoBayIcebergs/Processing/test /Volumes/TITANIC/DiskoBayImagery/Landsat/test

#        print cmd
#        os.system(cmd)
#        cmd = None
        
        
        
    '''           
    #Add features to the output layer (orig)
    print 'number of features is: ' + str(len(inLayer))
    featurecount=0
    for inFeature in inLayer:
        
        if featurecount%1000 == 0:
            print 'on feature number ' + str(featurecount)
        featurecount = featurecount + 1
        
        geometry = inFeature.GetGeometryRef()
        if 'simp_scene_geom' in locals() and simp_scene_geom.Overlaps(geometry) == True or land_geom.Overlaps(geometry) == True: 
            #I seem to get correct results with .Overlaps (but not contains/within) and I'm not sure if the definitions of "within" are at fault (see method_comparison readme)
            pass
        else:
            # Create output Feature
            outFeature = ogr.Feature(outLayerDefn)
    
            # Add field values from input Layer
            #for i in range(0, outLayerDefn.GetFieldCount()):
            #    fieldDefn = outLayerDefn.GetFieldDefn(i)
            #    outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(),
            #        inFeature.GetField(i))
    
            # Set geometry (and remove inner rings)
            geomWkt = geometry.ExportToWkt()
            newGeomWkt, trash = geomWkt.split(')', 1)
            newGeomWkt = newGeomWkt + '))'
            newGeom = ogr.CreateGeometryFromWkt(newGeomWkt)
            outFeature.SetGeometry(newGeom)
            
            #calculate centroid and add to attribute table
            centroid = newGeom.Centroid()
            centroid_text = centroid.ExportToWkt()
            trash, cent_x, cent_y = centroid_text.split(' ', 2)
            outFeature.SetField(field_names[0], cent_x[1:])
            outFeature.SetField(field_names[1], cent_y[:-1])
            
            #calculate area and add to attribute table
            area = newGeom.GetArea()
            outFeature.SetField(field_names[2], area)
            total_ice_area += area
            
            #area units
            outFeature.SetField(field_names[3], 'm^2')
            
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy()
    
        
    ds.Destroy()
    out_ds.Destroy()
    return total_ice_area
    
    
# Creates raster mask from vector_fn of ds.
# The rasterized mask is named raster_fn
def GetLandMask(ds, raster_fn, vector_fn, dtype):

    # Open vector file and get Layer
    vec_ds = ogr.Open(vector_fn)
    vec_layer = vec_ds.GetLayer()

    # Get GeoTransform of Geotiff to be masked
    geo = ds.GetGeoTransform()

    # Create the mask geotiff
    #dtype_str = 'gdal.GDT_' + dtype
    fm_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    mask_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, ds.RasterXSize, ds.RasterYSize, 1, fm_dtype)
    mask_ds.SetGeoTransform((geo[0], geo[1], geo[2], geo[3], geo[4], geo[5]))
    mask_ds.SetProjection(ds.GetProjection())
    band = mask_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Rasterize (vector file is rasterized into the projection of the original dataset, ds)
    gdal.RasterizeLayer(mask_ds, [1], vec_layer, burn_values=[1])

    vec_ds = None
    return mask_ds
    
#MergeImages checks the projections of the input images and if they don't match
#reprojects them. Then it combines (merges) the two images. new_name is the full path.
def MergeImages(img1, img2, new_name, dtype):
    ds = gdal.Open(img1)
    proj1 = ds.GetProjection()
    nodata1 = ds.GetRasterBand(1).GetNoDataValue()
    if nodata1 == None:
        nodata1 = 0
    ds = None

    ds = gdal.Open(img2)
    proj2 = ds.GetProjection()
    nodata2 = ds.GetRasterBand(1).GetNoDataValue()
    if nodata2 == None:
        nodata2 = 0
    ds = None

    isReproj1 = False
    isReproj2 = False
    
    if nodata1 != nodata2:
        nodata2=nodata1
    
    if proj1 != proj2:
        #this line will need to be changed for different fjords!
        utm_zone = '22N'
        if utm_zone in proj1:
            isReproj2 = True
            print "Projections dont match. Reprojecting..."
            cmd = "gdalwarp -t_srs " + proj1 + " -srcnodata " + str(nodata2) + " -dstnodata " + str(nodata1) + " " + img2 + " " + img2[:-4] + "_reproj.TIF"
            print cmd
            os.system(cmd)
            img2 = img2[:-4] + "_reproj.TIF"
        elif utm_zone in proj2:
            isReproj1 = True
            print "Projections dont match. Reprojecting..."
            cmd = "gdalwarp -t_srs " + proj2 + " -srcnodata " + str(nodata1) + " -dstnodata " + str(nodata1) + " " + img1 + " " + img1[:-4] + "_reproj.TIF"
            print cmd
            os.system(cmd)
            img1 = img1[:-4] + "_reproj.TIF"     
        else:
            print "Check projections of " + img1 + " and " + img2 + " Exiting."
            exit() 
            
    print "Merging images " + img1[-30:-4] + " and " + img2[-30:-4]
#    cmd = "gdal_merge.py -o " + new_name + " -n 0 -ot " + dtype + " " + img1 + " " + img2
    cmd = "gdal_merge.py -o " + new_name + " -n " + str(nodata1) + " -a_nodata '-9999' " + img1 + " " + img2
    os.system(cmd)
    
    return new_name

#this function is used by removeBorderClouds to determine if a cluster of "ice" pixels should be kept as ice or turned to cloud
def test_feature(feat):
   cl = np.count_nonzero(feat==2)
   wat = np.count_nonzero(feat==1)
   if wat==0 or np.float(cl)/wat >= 0.4:
       return 11
#           print 'cloud'
   else:
       return 12
#           print 'not cloud'

#this function takes in an array showing anticipated iceberg polygons (still raster clusters)
#and an array classed for ice, water, and cloud. It dilates the predicted ice and then checks
#to see if the dilated version has above a certain ratio of cloud to water pixels in it. If so, it
#flags the cluster for removal. Otherwise, it keeps them as ice. It ultimately returns an updated
#classed array with cloud-bordering clusters turned into cloud.
#the idea to use dilation and then check for cloud within the dilated version comes from David Shean
def removeBorderClouds(ice_array, classed_array):
    #note pan array is not yet binary, so make a binary ice mask
    binary_ice = np.copy(ice_array)
    binary_ice[binary_ice>0] = 1
    dilated_ice = np.copy(binary_ice)
    dilated_ice = ndimage.binary_dilation(dilated_ice, structure=ndimage.generate_binary_structure(2,1)).astype(dilated_ice.dtype)
    labelled_ice, num_features = ndimage.measurements.label(dilated_ice)
    print num_features
    labels = np.arange(1, num_features+1)   
    
    dilated_ice = None
#    print 'about to iterate over features'

    #iterate over all features, using labeled_comprehension, to determine if each feature should be marked cloud or ice    
    feature_values = ndimage.labeled_comprehension(classed_array, labelled_ice, labels, test_feature, int, -1)
#    print 'done iterating over features' 
#    print len(feature_values)
#    print np.min(feature_values)
#    print np.max(feature_values)

    #fill in binary array using values returned from labeled_comprehension            
    #I only have a vague understanding of how this works, but it does, so...
    #it's from: https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    #and is way faster than my previous attempt using np.nditer (code removed)      
    from_values = labels # pair values from 1 to number of features
    to_values = feature_values # values for each label
#    d = dict(zip(from_values, to_values))
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values,labelled_ice,sorter = sort_idx)
    ice_or_cloud = to_values[sort_idx][idx]
#    print('Check method_searchsort :', np.all(out == array/2)) # Just checking
#    print('Time method_searchsort :', timeit.timeit(method_searchsort, number = 1))
    sort_idx = None
    idx = None
    
#    print 'done updating feature_values array'
#    rasterfile(ds_pan.GetFileList()[0], classed_name[:-4]+'_featurevalues_faster.tif', ice_or_cloud, gdal.GDT_Byte)
#    rasterfile(ds_pan.GetFileList()[0], classed_name[:-4]+'_binary_ice.tif', binary_ice, gdal.GDT_Byte)
    
    #"remove" effects of dilation and turn cloud border features into cloud in classed array
#    print 'doing rest of calcs'
    ice_or_cloud = ice_or_cloud*binary_ice
    classed_array[ice_or_cloud==11] = 2
    classed_array[ice_or_cloud==12] = 3    

    binary_ice = None
    ice_or_cloud = None
            
    return classed_array
#    rasterfile(ds_pan.GetFileList()[0], classed_name[:-4]+'_afterbordercloudrem.tif', classed_array, gdal.GDT_Byte)

    
#getRasterArray is the first thing called for any file processing. It will return
#the dataset object as well as the numpy array of raster data. The merge flag value
#indicates whether or not there are multiple scenes that need to be merged; if so,
#this will merge them, save the raster, and return the merged ds and array
#The refl_flag indicates whether reflectance/brightness temperature is desired for that band
#if set to true, it will calculate these where needed. Thus reflectance/BT doesn't need to
#be calculated within functions that need reflectance/BT info because it is done here
#(in most case comments refer only to "reflectance" but mean both)
#if ever I want to try and calculate reflectance for multiple bands at once (capability already there for TOAreflectance.py),
#see a version of icebergdelineation.py prior to 27 Feb 2018 (inc depricated version used for first time I processed data)
def getRasterArray(path, folder, band_str, merge_flag, spacecraft, *args):
        
    #determine band numbers and data type based on sensor
    #note the input band string MUST be the non-reflectance version for this code to work
    #whether or not reflectance is calculated/opened is determined by the refl flag
    dtype, band, refl_flag = getBandNum(spacecraft, band_str)
    
    #don't need reflectance
    if refl_flag == False:
    
        #don't need mergeing
        if merge_flag == False:
            print "Opening Band " + band_str + '('+repr(band)+')'
            ds = gdal.Open(path + '/' + folder + '/' + folder + '_B' + str(band) + '.TIF')
        
        #need merged files
        elif merge_flag == True:
            f2 = args[0]       
            ds_filename = path + '/' + folder[dtst:dtend] + '_B' + str(band) + '_merged.TIF'
            ds_file2 = path[:-21] + '/' + folder[dtst:dtend] + '_B' + str(band) + '_merged.TIF' #CHANGE THIS TO 21 for running on titanic!
#            print 'ds_filename is ' + ds_filename
#            print 'ds_file2 is ' + ds_file2
            
#check to see if merged file exists (ds_filename and ds_file2 are two possible locations). If it does, open it. Otherwise, create it
            if os.path.isfile(ds_filename) == True:
                print 'file ' + ds_filename + ' exists; not re-merging'
                ds = gdal.Open(ds_filename)
                
            elif os.path.isfile(ds_file2) == True:
                print 'file ' + ds_file2 + ' exists; not re-merging'
                ds = gdal.Open(ds_file2)
                
            #if it doesn't exist, merge the non-reflectance files and open the result
            else:
                fn1 = path + '/' + folder + '/' + folder + '_B' + str(band) + '.TIF'
                fn2 = path + '/' + f2 + '/' + f2 + '_B' + str(band) + '.TIF'       
#                print 'fn1 is ' + fn1
#                print 'fn2 is ' + fn2
                ds = gdal.Open(MergeImages(fn1, fn2, ds_filename, dtype))
                
    
    #need reflectance or BT
    elif refl_flag == True:
        
        #don't need merging
        if merge_flag == False:
            #check for reflectance file
            ds_refl_filename = path+'/'+folder+'/'+ folder+"_B"+band+".tif"
#            print 'ds_refl_filename is ' + ds_refl_filename
            if os.path.isfile(ds_refl_filename) == True:
                print 'reflectance file exists for band ' + repr(band)
                pass
            else:
                #compute reflectance
                computeReflBT(spacecraft, path, folder, ds_refl_filename, band)                
                
            ds = gdal.Open(ds_refl_filename)    
        
        #need merged reflectance/BT files            
        elif merge_flag == True:
            f2 = args[0] 
            ds_refl_merge_fn = path+'/'+ folder[dtst:dtend]+"_B"+band+"_merged.tif"
            ds_refl_merge_fn2 = path[:-21] +'/'+ folder[dtst:dtend]+"_B"+band+"_merged.tif" #CHANGE THIS TO 21 for running on titanic!, 20 on HD
#            print 'ds_refl_merge_fn is ' + ds_refl_merge_fn
#            print 'ds_refl_merge_fn2 is ' + ds_refl_merge_fn2
            
            #check two possible locations for merged reflectance files
            if os.path.isfile(ds_refl_merge_fn) == True:
                print 'file ' + ds_refl_merge_fn + ' exists; not re-merging'
                ds = gdal.Open(ds_refl_merge_fn)
                
            elif os.path.isfile(ds_refl_merge_fn2) == True:
                print 'file ' + ds_refl_merge_fn2 + ' exists; not re-merging'
                ds = gdal.Open(ds_refl_merge_fn2)
           
            #the merged reflectance files don't exist
            else:
                #check to see if either or the reflectance files exist. If so, use them, otherwise, make them
                refl_1_fn = path+'/'+folder+'/'+ folder+"_B"+band+".tif"
                refl_2_fn = path+'/'+f2+'/'+ f2+"_B"+band+".tif"
#                print 'refl_1_fn is ' + refl_1_fn
#                print 'refl_2_fn is ' + refl_2_fn
                
                if os.path.isfile(refl_1_fn) == True:
                    print 'reflectance file exists for band ' + repr(band)
                else:
                    #compute reflectance
                    computeReflBT(spacecraft, path, folder, refl_1_fn, band)             

                #check for scene 2 files
                if os.path.isfile(refl_2_fn) == True:
                    print 'reflectance file exists for band ' + repr(band)
                else:
                    #compute reflectance
                    computeReflBT(spacecraft, path, f2, refl_2_fn, band)
                ds = gdal.Open(MergeImages(refl_1_fn, refl_2_fn, ds_refl_merge_fn, dtype))     

            
    ds_array = ds.GetRasterBand(1).ReadAsArray()
    
    return ds, ds_array

def computeReflBT(spacecraft, path, folder, fn, band):
    #alternative to in: band[0] == '6'
    #if BT
    if spacecraft == "LANDSAT_7" and '6' in band:
        cmd = "python /Users/jessica/Scripts/python_geotif_fns/BrightTempCalc.py  " + path+'/'+folder+'/' + " " + folder+"_MTL.txt false " + repr(band[0])
        os.system(cmd)
        
    elif spacecraft == "LANDSAT_8" and '11' in band:
        print 'bright temp'
        print repr(band[0:2])
        cmd = "python /Users/jessica/Scripts/python_geotif_fns/BrightTempCalc.py  " + path+'/'+folder+'/' + " " + folder+"_MTL.txt false " + repr(band[0:2])
        print cmd
        os.system(cmd)
    
    #otherwise (e.g. reflectance)    
    else:
        cmd = "python /Users/jessica/Scripts/python_geotif_fns/Landsat_TOARefl.py " + path+'/'+folder+'/' + " " + folder+"_MTL.txt 'ETM+ Thuillier' false " + repr(band[0])
        os.system(cmd)
        cmd = None

#readMetadata is from Steve Kochaver's Landsat_TOARefl script. It creates a dictionary
#of Landsat metadata
def readMetadata(metadataFile):
    f = metadataFile
    
    #Create an empty dictionary with which to populate all the metadata fields.
    metadata = {}

    #Each item in the txt document is seperated by a space and each key is
    #equated with '='. This loop strips and seperates then fills the dictonary.
    for line in f:
        if not line.strip() == "END":
            val = line.strip().split('=')
            metadata [val[0].strip()] = val[1].strip().strip('"')      
        else:
            break

    return metadata
    

#ProcessFolder collects the list of scene_directories to be processed and puts them in a dictionary by date
#thus, we can now process each scene date
#it will work for either pre-collection or collection landsat scenes but assumes you have all of one or the other in any given run
#CODE IMPROVEMENT: make it possible to do a combination of pre and collection scenes, and in general be more flexible for different inputs
def ProcessFolder (path, save_path):
    global dtst
    global dtend
    
    folders = [f for f in os.listdir(path) if os.path.isdir(path + '/' + f)]
    #print 'folders are ' + ', '.join(folders)
    proc_dict = dict()
    
    #check for collection vs pre-collection. If put this into the below loop, could potentially add functionality for using both types at once (note would need to convert to all DOY or date)
    if len(folders[0]) < 22:
        dtst = 9
        dtend = 16
    else:
        dtst = 17
        dtend = 25
        
    print dtst
    print dtend
    print folders[0][dtst:dtend]
        
    for folder in folders:        
        if folder[dtst:dtend] in proc_dict:
            proc_dict[folder[dtst:dtend]].append(folder)
        else:
            proc_dict[folder[dtst:dtend]] = [folder]
        
    print proc_dict

    for date in list(proc_dict.keys()):
        if len(proc_dict[date]) == 1:
            merge_flag = False
            scene = proc_dict[date]
            print '\n Processing scene: ' + ''.join(scene)
            ProcessScene(path, ''.join(scene), save_path, merge_flag)
        elif len(proc_dict[date]) == 2:
            merge_flag = True
            scene1 = proc_dict[date][0]
            scene2 = proc_dict[date][1]
            print '\n Processing scene: ' + scene1 + ' with second scene ' + scene2
            ProcessScene(path, scene1, save_path, merge_flag, scene2)
        else:
            print 'WARNING: date with >2 scenes in this folder!'
        
              
#ProcessScene is the main workhorse of the code. It takes the scene dir (or dirs,
#if there are two) and performs the analyses on them and exports the results
#CODE IMPROVEMENT: use classes! By doing this, you could put getting all the needed info into one step and then maybe have less within each function
def ProcessScene(path, fold, save_path, merge_flag, *args):
    #determine spacecraft by reading in metadata (needed to determine relevant bands later)
    metadataPath = path + '/' + fold + '/' + fold + '_MTL.txt' 
    metadataFile = open(metadataPath)
    metadata = readMetadata(metadataFile)
    metadataFile.close()
    spacecraft = metadata["SPACECRAFT_ID"]
    
    if spacecraft == "LANDSAT_7":
        dtype = np.dtype('uint8') #'Byte' could also use 'b'
    elif spacecraft == "LANDSAT_8":
        dtype = np.dtype('uint16') #'UInt16'
    
    print 'getting Panchromatic band'
    if merge_flag == True:
        fold2 = args[0]
        ds_pan, pan_array = getRasterArray(path, fold, 'pan', merge_flag, spacecraft, fold2)
        ds_pan_refl, pan_refl_array = getRasterArray(path, fold, 'pan_refl', merge_flag, spacecraft, fold2)
    else:
        ds_pan, pan_array = getRasterArray(path, fold, 'pan', merge_flag, spacecraft)
        ds_pan_refl, pan_refl_array = getRasterArray(path, fold, 'pan_refl', merge_flag, spacecraft)
    
      
    ####GENERATE SCENE EXTENT####
    #the scene extent was originally based on the panchromatic band extent. However, it was found that other bands had slightly different extents
    #thus, the scene extent needs to be determined based on where all bands used have data (the extent mask is initiated here and updated when new bands are used)
    #the extent array is exported prior to iceberg delineation
    extent_array = np.ones_like(pan_array) 
    extent_array = updateExtentArray(extent_array, pan_array)
    
#    scene_ext_name = save_path + '/' + fold[dtst:dtend] + '_scene_extent-initial.tif'
#    rasterfile(ds_pan.GetFileList()[0], scene_ext_name, extent_array, gdal.GDT_Byte)

    ####CREATE CLASSED ARRAY####
    classed_name = save_path + '/' + fold[dtst:dtend] + '_classified.TIF'
    classed_array = np.ones_like(pan_array) #this must happen before the pan array is thresholded becuase it sets up where there is data for a scene

    ####GENERATE LAND MASK####
    print 'rasterizing land mask from input shapefile (must update land mask shapefile path and filename for each location)'
#    landmaskpath = '/Users/jessica/IcebergDelineation/Landmask_shapefiles/AOI_DBbergdelin_buffered100.shp' #path+'/AOI_DBbergdelin_buffered100.shp'
    landmaskpath = '/Users/jessica/IcebergDelineation/Landmask_shapefiles/AOI_KSbergdelin_poly.shp'
    ##change required here for different fjords!!! (output and input filenames, below)
    #orig land mask
    #land_mask_ds = GetLandMask(ds_pan, path+'/DiskoBay_LandMask_Raster.TIF', path+'/AOI_nofjords_buffered100.shp', dtype)
    #updated land mask (to have max coverage be 100% instead of like 95%)
    land_mask_ds = GetLandMask(ds_pan, path+'/Kangerlussuup_LandMask_Raster.TIF', landmaskpath, dtype)        
#    land_mask_ds = gdal.Open(path+'/DiskoBay_LandMask_Raster.TIF')
###!!!    #only need byte here!!
    
    ####GENERATE CLOUD MASK####
    print 'generating cloud mask'
    run_type = "machlearn"  #"threshcloud"
    #for run_type threshcloud, a different function needs to be called: ThreshCloudMask
    if merge_flag == True:
        cloud_mask_array = GetCloudMask(path, fold, merge_flag, spacecraft, dtype, ds_pan, fold2)
    else:
        cloud_mask_array = GetCloudMask(path, fold, merge_flag, spacecraft, dtype, ds_pan)
#    if merge_flag == True:
#        cloud_mask_array = ThreshCloudMask(path, fold, merge_flag, spacecraft, dtype, ds_pan, fold2)
#    else:
#        cloud_mask_array = ThreshCloudMask(path, fold, merge_flag, spacecraft, dtype, ds_pan)
#    rasterfile(ds_pan.GetFileList()[0], save_path+'/'+fold[dtst:dtend]+'_'+run_type+'_mask.tif', cloud_mask_array, gdal.GDT_Byte)    


    ####UPDATE SCENE EXTENT AND APPLY (AND EXPORT)####
    #the scene extent requires updating to the cloud mask input band extent
    #here it is updated, exported, and applied to the pan band before iceberg delineation
    cloud_ext_array = np.ones_like(cloud_mask_array)
    cloud_ext_array[cloud_mask_array==0]=0
    extent_array = updateExtentArray(extent_array, cloud_ext_array)
    cloud_ext_array = None

    pan_array = pan_array * extent_array  #this and the prior block should effectively also be applying the cloud mask, so I could probably remove that step above, but I'm going to leave it for now unless I'm doing further code testing
    pan_refl_array = pan_refl_array * extent_array
    classed_array = classed_array*extent_array
    cloud_mask_array = cloud_mask_array*extent_array
        
    scene_ext_name = save_path + '/' + fold[dtst:dtend] + '_scene_extent.tif'
    rasterfile(ds_pan.GetFileList()[0], scene_ext_name, extent_array, gdal.GDT_Byte)
    extent_array = None
  
    scene_ext_shape = path + '/' + fold[dtst:dtend] + '_scene_extent.shp' #scene_ext_name[:-4] + '.shp'
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(scene_ext_shape):
        driver.DeleteDataSource(scene_ext_shape)    
    cmd = 'gdal_polygonize.py -mask ' + scene_ext_name + ' ' + scene_ext_name + ' -b 1 -f "ESRI Shapefile" ' + scene_ext_shape
    os.system(cmd)


    ####APPLY LAND MASK####
    print 'applying land mask to Panchromatic band'
    land_mask_array = land_mask_ds.GetRasterBand(1).ReadAsArray()
    pan_array = pan_array*land_mask_array
    pan_refl_array = pan_refl_array*land_mask_array
    cloud_mask_array = cloud_mask_array * land_mask_array
    classed_array = classed_array*land_mask_array
    ####CODE IMPROVEMENT: could write a function that applies masks to all the input arrays
    
    pixel_size = 225  #code improvement: get this from the metadata file
    #aoi = np.count_nonzero(land_mask_array) * pixel_size  #this number is meaningless because it depends on the row/path and image extents
##need to change for each fjord/aoi!
    #this area is that of the vector polygon delineating the region of interest, rounded down to the nearest number of pixels. area was calculated using vector tools (attribute table calculator doesn't work) in QGIS
#    aoi = 12898981350  #rounded down to the nearest pixel 12898981350; the original value from the vector polygon is 12898981410m2
    #aoi = 60494255 * pixel_size  #code improvement: don't hard code this in (or have a lookup table for by region)
    aoi = 363756314  #KS
    scene_aoi = np.count_nonzero(pan_array) * pixel_size
        
    ####APPLY CLOUD MASK####
#    cloud_aoi = (cloud_mask_array==255).sum() * pixel_size
#    print np.count_nonzero(land_mask_array)
#    print np.count_nonzero(pan_array)
#    print (cloud_mask_array==255).sum()
#    print np.count_nonzero(cloud_mask_array==255)
#    print np.count_nonzero(cloud_mask_array==255) * pixel_size
    #ice = 3 (applied after thresholding), water = 1, cloud = 2, no data = 0
    classed_array[cloud_mask_array==255] = 2
#    classed_array = classed_array*land_mask_array #must apply the land mask after the cloud mask because otherwise clouds over land are included

#    cloud_mask_array[cloud_mask_array==255] = 0
    pan_array[cloud_mask_array==255] = 0 #= pan_array*cloud_mask_array
    pan_refl_array[cloud_mask_array==255] = 0 #= pan_refl_array*cloud_mask_array
#    cloud_aoi = np.count_nonzero(land_mask_array*(np.ones_like(cloud_mask_array)-cloud_mask_array)) * pixel_size

    land_mask_ds = None
    land_mask_array = None
    cloud_mask_array = None     

    ####ICEBERG DELINEATION####
    print 'threshold image and find clusters'  
    threshold = 0.19
#    threshold = [0.2 if spacecraft=='LANDSAT_7' else 0.18]
    #threshold = [12000 if dtype==np.dtype('uint16') else 47] #8000/33 or 17990/70
    #only threshold the pan_refl_array, then update the pan_array after removing cloud border objects below
    pan_refl_array[pan_refl_array<threshold] = 0
#    pan_array[pan_refl_array<threshold] = 0
    
    #ice = 3 (applied after thresholding), water = 1, cloud = 2, no data = 0
    classed_array[pan_refl_array>0] = 3
    #this (below) returns a raster with the extent based on this date. I think the best approach to combining these is going to be to use gdalwarp (or maybe
    #write something specific if that will be faster to run) to create rasters that all have the same extent, then just add/layer the arrays. Xarray or
    #geopandas might be useful tools here
    rasterfile(ds_pan.GetFileList()[0], classed_name[:-4]+'_preborderrem.tif', classed_array, gdal.GDT_Byte)
#    geo = ds_pan.GetGeoTransform()
#    output_ds = gdal.GetDriverByName('GTiff').Create(classed_name, ds_pan.RasterXSize, ds_pan.RasterYSize, 1, gdal.GDT_Byte)
#    output_ds.SetGeoTransform((geo[0], geo[1], geo[2], geo[3], geo[4], geo[5]))
#    output_ds.SetProjection(ds_pan.GetProjection())
#    output_ds.GetRasterBand(1).WriteArray(classed_array)
#    classed_array = None
    

    ####REMOVE "ICEBERGS" THAT BORDER CLOUDS####
    print 'removing false positive "icebergs" clusters that border clouds'
    #these next few (~8) lines were so I could run/test the removal of border polygons without having to run the rest of the algorithm
#    classed_name = '/Users/jessica/DiskoBayIcebergs/Development2_restart2017/cloud_border_FPs/2015281_classified.TIF'
#    ds_classed = gdal.Open(classed_name)
#    classed_array = ds_classed.GetRasterBand(1).ReadAsArray()
#    
#    threshold = 0.19
#    pan_array[pan_refl_array<threshold] = 0
#    pan_array[classed_array==0] = 0
#    pan_array[classed_array==2] = 0

    classed_array = removeBorderClouds(pan_refl_array, classed_array)
    pan_array[classed_array!=3] = 0
    rasterfile(ds_pan.GetFileList()[0], classed_name, classed_array, gdal.GDT_Byte)

    ds_pan_refl = None
    pan_refl_array = None    

    
    ####EXPORT RESULTS AND MAKE SHAPEFILE OF ICEBERGS####
    print 'export masked panchromatic scene to a geotif and make shapefile of clusters'
    ##the filename here will need to be changed!!!
    final_name = save_path + '/' + fold[dtst:dtend] + '_processed_' + run_type + '.TIF'    
    fm_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    rasterfile(ds_pan.GetFileList()[0], final_name, pan_array, fm_dtype)
   
    #create a binary version of the thresholded file to use to make shapefile of clusters
    bin_name = save_path + '/' + fold[dtst:dtend] + '_binary.TIF'
    pan_array[pan_array>0] = 1
    geo = ds_pan.GetGeoTransform()
    bin_ds = gdal.GetDriverByName('GTiff').Create(bin_name, ds_pan.RasterXSize, ds_pan.RasterYSize, 1, gdal.GDT_Byte)
    bin_ds.SetGeoTransform((geo[0], geo[1], geo[2], geo[3], geo[4], geo[5]))
    bin_ds.SetProjection(ds_pan.GetProjection())
    bin_ds.GetRasterBand(1).WriteArray(pan_array)
    bin_ds = None
    shape_fname = save_path + '/' + fold[dtst:dtend] + '.shp'
    cmd = 'gdal_polygonize.py -mask ' + bin_name + ' ' + bin_name + ' -b 1 -f "ESRI Shapefile" ' + shape_fname
    os.system(cmd)
    os.system('rm ' + bin_name)
    
    print 'add fields to cluster shapefile and remove polygons bordering scene or land'
    #do some calculations and processing on the clusters
    ##the filename here will need to be changed!!
    new_shape_fname = shape_fname[:-4] + '_bergs_' + run_type + '.shp'
#    ice_aoi = BergInfo(shape_fname, new_shape_fname) #the optional arguments, below, remove polygons that intersect a scene or land border; this process is quite slow (minimum 10min/scene, usually 20-30)
    ice_aoi = BergInfo(shape_fname, new_shape_fname, scene_ext_shape, landmaskpath)
#    os.system('rm -r ' + shape_fname[:-3] + '*')

    #calculate and add to text file information on areal coverage of scene within bay and of cloud cover
    cloud_aoi = (classed_array==2).sum() * pixel_size

    scene_perc = np.float(scene_aoi)/aoi
    cloud_perc = np.float(cloud_aoi)/aoi
    ice_perc = np.float(ice_aoi)/aoi
    textfile = 'regional_area_info.txt'
    
    if not os.path.isfile(save_path + '/' + textfile):
        with open(save_path + '/' + textfile, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([fold[dtst:dtend], aoi, scene_aoi, scene_perc, cloud_aoi, cloud_perc, ice_aoi, ice_perc])
    else:
        with open(save_path + '/' + textfile, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([fold[dtst:dtend], aoi, scene_aoi, scene_perc, cloud_aoi, cloud_perc, ice_aoi, ice_perc])
            
   
    ds_pan = None
    pan_array = None
    bin_ds = None 

    
######## MAIN ########

#only input required is full path of dir to be processed. all other variables [for now] will be set internally
if len(sys.argv) != 3:
    print "Usage is:   python icebergdelineation.py [full path of directory of directories of LS scenes to be processed] [full path of directory for saving results (cannot be the same as the processing directory)"
    exit()
    
path = sys.argv[1]

if path[-1] == '/':
    path = path[:-1]

save_path = sys.argv[2]
if save_path[-1] == '/':
    save_path = save_path[:-1]
cmd = 'mkdir ' + save_path
os.system(cmd)

ProcessFolder(path, save_path)

