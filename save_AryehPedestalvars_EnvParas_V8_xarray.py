#!/usr/bin/env python3
import numpy as np
import datetime
import pytz
import pygrib
from timezonefinder import TimezoneFinder
import glob
from scipy import ndimage
from collections import Iterable
from skimage import measure
from skimage import filters
import pyhdf
from pyhdf.SD import SD, SDC 
from pyhdf.HDF import *
from pyhdf.VS import *
import pprint
import csv
import xarray as xr
from metpy.calc import vapor_pressure,dewpoint
from metpy.units import units
import os

from lib.utils import gunzip, slide_max
from lib.define_pedestal import define_cluster, def_cutoff

def group_consecutives(vals, step=1):
    #find out the continous numbers
    #"""Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result



def Month_jdate(month,yr):
####################################################################################################
#this function determin the starting and ending Jdate for the month 
    #leap year 
    msdate0=[1,32,61,92,122,153,183,214,245,275,306,336] 
    #perpetual year 
    msdate1=[1,32,60,91,121,152,182,213,244,274,305,335] 
    if (yr % 4) == 0:  
        if (yr % 100) == 0: 
            if (yr % 400) == 0:
                jdate0=msdate0[month]
                if month<11:
                    jdate1=msdate0[month+1]-1
                else:
                    jdate1=366
            else:
                jdate0=msdate1[month]
                if month<11:
                    jdate1=msdate1[month+1]-1
                else:
                    jdate1=365
        else:
            jdate0=msdate0[month]
            if month<11:
                jdate1=msdate0[month+1]-1
            else:
                jdate1=366
    else:
        jdate0=msdate1[month]
        if month<11:
            jdate1=msdate1[month+1]-1
        else:
            jdate1=365
    return(jdate0,jdate1)

def find_nearest(array, value):
####################################################################################################
# find_nearest:
#     Find the nearest index of value to the array
# Input: 
#     array - a array
#     value - the value where the value is located in array
# Return:
#     idx - index within the array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def add_ENV_csv(fileloc,jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,cape,cin,lcl,SensFlux,
                LantFlux,MSLP,SST, Omega, lowVWS, midVWS, upVWS, ttlVWS,lowVWS_dir, midVWS_dir,
                upVWS_dir, ttlVWS_dir,lowSH, midSH, upSH, ttlSH,lowRH, midRH, upRH, ttlRH,
                mIWV,mIWVRatio,AOD,IWP,SWP,SWPC,COmeanAMSRR,PedmeanAMSRR,PedmaxAMSRR,
                COmeanCldRR,PedmeanCldRR,PedmaxCldRR):
    #input:
    # fileloc - path_to_file
    # gnu     - granule number
    # stid    - storm number in Juliet's dataset
    # UTC_time- UTC time
    # Time    - day or night: daytime=1, nighttime=0
    # lon/lat - cloud object mean latitude and longitude 
    # CAPE/CIN/LCL      - ERA5 surface CAPE/CIN/LCL
    # SensFlux/LantFlux - ERA5 surface sensible/latent heat flux
    # MSLP              - ERA5 surface mean sea level pressure
    # SST               - ERA5 surface sea surface temperature
    # Omega             - ERA5 all levels omega at 500 hPa
    # low/mid/up/ttlVWS - ECMWF-AUX low/mid/up/total level VWS
    # low/mid/up/ttlVWS-dir - ECMWF-AUX low/mid/up/total level VWS direction
    # low/mid/up/ttlSH  - ECMWF-AUX low/mid/up/total level SH
    # low/mid/up/ttlRH  - ECMWF-AUX low/mid/up/total level RH
    # IWV/IWV Ratio     - integrated water vapor of whole column, and ratio between IWV/IWV_Saturation
    # AOD               - MERRA2 AOD
    # IWP, SWP, SWPC    - integrated ice water path, snow water path, snow water path confidence level 
    # COmeanAMSRR       - mean rain rate over the whole CO
    # PedmeanAMSRR, PedmaxAMSRR - mean/max rain rate over pedestal region from AMSR
    # COmeanCldRR, COmaxCldRR   - mean/max rain rate over the whole CO from CloudSat
    # PedmeanCldRR,PedmaxCldRR  - mean/max rain rate over pedestal region from CloudSat 
    with open(fileloc, 'a+',  newline='') as f:
        what = csv.writer(f)
        list_of_stuff = [jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,cape,cin,lcl,
                         SensFlux,LantFlux,MSLP,SST, Omega, lowVWS, midVWS, upVWS, ttlVWS,
                         lowVWS_dir, midVWS_dir, upVWS_dir, ttlVWS_dir,lowSH, midSH, upSH,
                         ttlSH,lowRH, midRH, upRH, ttlRH,mIWV,mIWVRatio,AOD, IWP, SWP,SWPC,
                         COmeanAMSRR,PedmeanAMSRR,PedmaxAMSRR,COmeanCldRR,PedmeanCldRR,
                         PedmaxCldRR]
        what.writerow(list_of_stuff)
        
def add_SFC_csv(fileloc,jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,ped_soil1,ped_soil2,ped_soil3,
                ped_soil4,ped_soiltype,ped_low_veg_cover,ped_high_veg_cover,ped_veg_type,
                soil1, soil2, soil3,soil4,soiltype,low_veg_cover, high_veg_cover, veg_type,
                tmk2m, dew2m,skinT):
    #input:
    # fileloc - path_to_file
    # gnu     - granule number
    # stid    - storm number in Juliet's dataset
    # UTC_time- UTC time
    # Time    - day or night: daytime=1, nighttime=0
    # lon/lat - cloud object mean latitude and longitude 
    # meanflag- average land-sea mask among the CO
    # flag    - bin the surface mask (1-land,2-ocean,3-coast,4-coastal?,5-river)
    # Pedestal region soil, vegetation types
    # ped_soil1-ped_soil4 - sum of the volumetric soil layer in m3/m3
    # ped_soiltype - soil number in each bins
    # CO over soil1, soil2, soil3, soil4
    # soil and vegation type
    # SkinT   - skin temperature 
    # tmk2m, dew2m- temperature, dewpoint at 2m
    with open(fileloc, 'a+',  newline='') as f:
        what = csv.writer(f)
        list_of_stuff = [jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,ped_soil1,ped_soil2,ped_soil3,
                         ped_soil4,ped_soiltype,ped_low_veg_cover,ped_high_veg_cover,ped_veg_type,soil1,
                         soil2, soil3,soil4,soiltype,low_veg_cover, high_veg_cover, veg_type,tmk2m, dew2m,skinT]
        what.writerow(list_of_stuff)

def add_CUTOFF_csv(fileloc,jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,cutoff,freeze,Auxfrz,minbase,
                   maxtop,meanbase,meantop,pedw,anvw,AuxCTT,AuxminCTT,meanIR,minIR,meanCTT,minCTT,
                  PedmeanHght,PedmaxHght,meanHght,maxHght,Pedmeantopo,Pedmaxtopo,nPed,Ped_wid,nCore,Core_wid):
    #input:
    # fileloc - path_to_file
    # gnu     - granule number
    # stid    - storm number in Juliet's dataset
    # UTC_time- UTC time
    # Time    - day or night
    # meanflag- average land-sea mask among the CO
    # flag    - bin the surface mask (1-land,2-ocean,3-coast,4-coastal?,5-river)
    # cutoff  - cutoff height [m] 
    # freeze  - freezing level from Juliet and AUX over pedstal region
    # base    - cloud base [m] from CloudSat 
    # top     - cloud top [m] from CloudSat
    # pedw    - pedestal width [idx]
    # anvw    - anvil width [idx]
    # flag    - surface flag
    # lon/lat - cloud object mean latitude and longitude 
    # meanIR/minIR   - CloudSat 2B-TB94 minimum cloud top temperature in IR
    # meanCTT/minCTT   - CloudSat 2B-TB94 minimum cloud top temperature
    # nCore/Core_wid   - number of cores and each core width in indices 
    # nPed/Ped_wid     - number of pedestals and each pedestal width in indices 
    # Pedmeantopo, Pedmaxtopo - mean and max Tropopause height over the pedestal region
    # PedmaxHght, PedmeanHght - maximum and mean topography height (DEM_height) over the pedestal region 
    # maxHght, meanHght- maximum and mean topography height (DEM_height) over the whole CO

    with open(fileloc, 'a+',  newline='') as f:
        what = csv.writer(f)
        list_of_interest = [jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,cutoff,freeze,Auxfrz,
                            minbase,maxtop,meanbase,meantop,pedw,anvw,AuxCTT,AuxminCTT,
                            meanIR,minIR,meanCTT,minCTT,PedmeanHght,PedmaxHght,meanHght,
                            maxHght,Pedmeantopo,Pedmaxtopo,nPed,Ped_wid,nCore,Core_wid]
        what.writerow(list_of_interest)

def add_SOUNDING_csv(fileloc,jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,
                     Tmk_arr,Td_arr,SH_arr,U_arr,V_arr):
    #input:
    # fileloc - path_to_file
    # gnu     - granule number
    # stid    - storm number in Juliet's dataset
    # UTC_time- UTC time
    # Time    - day or night
    # meanflag- average land-sea mask among the CO
    # flag    - bin the surface mask (1-land,2-ocean,3-coast,4-coastal?,5-river)
    # Tmk_arr - temperature [K] 
    # Td_arr  - dewpoint [K]
    # SH_arr  - specific humidity[g/kg]
    # U_arr   - U [m/s]
    # V_arr   - V [m/s]
    # flag    - surface flag
    # lon/lat - cloud object mean latitude and longitude 
    with open(fileloc, 'a+',  newline='') as f:
        what = csv.writer(f)
        list_of_interest = [jdate,gnu,stid,UTCtime,Time,lon,lat,meanflag,flag,
                            Tmk_arr,Td_arr,SH_arr,U_arr,V_arr]
        what.writerow(list_of_interest)
#END OF FUNCTIONS

# DEFINE VARIABLE NAMES
# Determine variables that does not change 
# vertical bin size, in km:
dz = 0.23972353
# check 8 levels above the given level to cover 2 km range
num_levels_to_check = 8

# $$$-index of lowest possible tropopause height at 7.5 km
max_idx = 73
# $$$-index of default tropopause height at 18 km
default_idx = 29

ihght_diff = 29 #indices difference with full height and Juliet's height
trop_gap   = 16 #$$$-Gap between tropopause and cloud top height (16 indices--3.84km; ~4km)
ground_gap = 6  #$$$-Gap between DEM_height and cloud base height to be 6 indices -- 1.4 km

#Determine variables that does not change
Months = ['January','February','March','April','May','June','July','August','September','October','November','December']
Acry   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

cldgrid   = 1.079 #km 
cldzgrid  = 0.24  #km

lvls      = np.array([200,300,500,700,850,925,1000])
#coarse, medium, medium fine, fine, very fine, organic, tropical organic
sat_soil  = np.array([np.nan,0.403,0.439,0.430,0.520,0.614,0.766,0.439]) #Saturation volumetric soil moisture m3/m3
nlv    = len(lvls)
ERAtime= np.arange(24)
ntime= 24

#Define constant variables 
Rv = 461.5250 #J/(kg K)
Rd = 287.0597 #J/(kg K)
epsilo = Rv/Rd-1

###READ IN LON/LAT of ERA5 and MERRRA2 ahead of the loop.
####----READIN GENERAL MERRA2 DATA----####
ncfile1 = xr.load_dataset('/avalanche/bpan/MERRA2/MERRA2_300.inst3_2d_gas_Nx.20060815.nc4.nc4')
merLAT  = ncfile1.variables['lat']
merLON  = ncfile1.variables['lon']
merTIME = ncfile1.variables['time']

merLAT = np.array(merLAT,dtype=float)
merLON = np.array(merLON,dtype=float)
merTIME= np.array(merTIME,dtype=int)

####----READIN GENERAL ERA5 DATA----####
ERAI = '/avalanche/bpan/ERA5/2006/August/ecmwf.grb2p40.20060815'
grbs       = pygrib.open(ERAI)
grb  = grbs.select()[0]
lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), \
float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']) )
lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), \
float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']) )
####READin change in files####

###---READIN ERA5 land-ocean mask
era5mask = '/avalanche/bpan/ERA5/ecmwf.grb2p40.20060815-surfacemask-single'
grbs     = pygrib.open(era5mask)
lsmask   = grbs.select(name='Land-sea mask')
ERAMask  = lsmask[0].values[:,:]
###END of mask readin

#---LOOP THROUGH ALL YEARS 
for yr in range(2006,2007):
    #FILE DIRECTORY
    Path_to_nc = '/avalanche/bpan/CloudSat/CloudSat_v2/'+str(yr)
    Path_to_cld = '/avalanche/bpan/CloudSat/2B-CLDCLASS/'+str(yr)+'/'
    Path_to_aux  = '/avalanche/bpan/CloudSat/ECMWF-AUX/'+str(yr)+'/'

    #Loop through months
    for imonth in range(7,10):
        #Find the month and day of the time
        tmpm = Months[imonth]
        tmpa = Acry[imonth]
        Path_to_juliet = Path_to_nc+'/global_convective_vars_'+str(yr)
        sday,eday = Month_jdate(imonth,yr)
        cldbase= np.empty([ntime,721,1440])
        CAPE   = np.empty([ntime,721,1440])
        LCL    = np.empty([ntime,721,1440])
        CIN    = np.empty([ntime,721,1440])
        CINnan = np.empty([ntime,721,1440])
        MSLP   = np.empty([ntime,721,1440])
        SST    = np.empty([ntime,721,1440])
        SSTnan = np.empty([ntime,721,1440])
        Omega  = np.empty([ntime,721,1440])
        SFLX   = np.empty([ntime,721,1440])
        LFLX   = np.empty([ntime,721,1440])
        SOIL1  = np.empty([ntime,721,1440])
        SOIL2  = np.empty([ntime,721,1440])
        SOIL3  = np.empty([ntime,721,1440])
        SOIL4  = np.empty([ntime,721,1440])
        SOILTYPE= np.empty([ntime,721,1440])
        LOWVEG  = np.empty([ntime,721,1440])
        HIGHVEG = np.empty([ntime,721,1440])
        LOWTYPE = np.empty([ntime,721,1440])
        HIGHTYPE= np.empty([ntime,721,1440])
        TMK2m   = np.empty([ntime,721,1440])
        DEW2m   = np.empty([ntime,721,1440])
        SH   = np.empty([ntime,nlv,721,1440])
        Tempk= np.empty([ntime,nlv,721,1440])
        Uwind= np.empty([ntime,nlv,721,1440])
        Vwind= np.empty([ntime,nlv,721,1440])
        soil_percent1 = np.empty([ntime,721,1440])
        soil_percent2 = np.empty([ntime,721,1440])
        soil_percent3 = np.empty([ntime,721,1440])
        soil_percent4 = np.empty([ntime,721,1440])

        ####READin change in files####
        #determin the starting point of the monthly date
        ### SET UP THE TEXT FILE TO STORE THE VARIABLES
        txtfile_header = ['JDate','Granuel','STID','UTC Time','Day or Night','Longitude','Latitude','MeanFlag','Surface_Flag',
                        'Cutoff[m]','Freeze[m]','AUX_Freeze[m]','min Cloud Base[m]','max Cloud Top[m]','Cloud Base[m]',
                        'Cloud Top[m]',' Pedetal Width[km]', ' Anvil Width[km]','AUX_CTT[K]','min_AUX_CTT[K]',
                        'mean_IR_CTT[K]','min_IR_CTT[K]','mean_CloudSat_CTT[K]','min_CloudSat_CTT[K]','Ped_mean_DEMHght[m]',
                        'Ped_max_DEMHght[m]','mean_DEMHght[m]','max_DEMHght[m]','Ped_mean_Tropopause[km]','Ped_max_Tropopause[km]',
                        'Num_Pedestal','Pedestal_Width_idx','Num_Core','Core_Width_idx']
        txtfile = Path_to_nc+'/'+str(yr)+tmpm+'_PedestalMorphology_V8.csv'
        with open(txtfile, 'w', encoding='UTF8', newline='') as f:
                txtwriter = csv.writer(f)
                txtwriter.writerow(txtfile_header)

        envfile_header = ['JDate','Granuel','STID','UTCTime','Day or Night','Longitude','Latitude','MeanFlag','Surface_Flag',
                        'ERA5_CAPE[J/kg]','ERA5_CIN[J/kg]','ERA5_LCL[m]','ERA5_SensibleFlux[J/m2]','ERA5_LatentFlux[J/m2]',
                        'ERA5_MSLP[hPa]','ERA5_SST[K]','ERA5_Omega[Pa/s]','ERA5_lowVWS[m/s]','ERA5_midVWS[m/s]','ERA5_upVWS[m/s]',
                        'ERA5_ttlVWS[m/s]','ERA5_lowVWS_dir[deg]','ERA5_midVWS_dir[deg]','ERA5_upVWS_dir[deg]',
                        'ERA5_ttlVWS_dir[deg]','ERA5_lowSH[g/kg]','ERA5_midSH[g/kg]','ERA5_upSH[g/kg]',
                        'ERA5_ttlSH[g/kg]','ERA5_lowRH[%]','ERA5_midRH[%]','ERA5_upRH[%]','ERA5_ttlRH[%]',
                        'AUX_IWV[g/m2]','AUX_IWVRatio[%]','MERRA2_AOD','2C-ICE-IWP[g/m2]','2C-SNOW-SWP[g/m2]',
                        '2C-SNOW-SWP-CONF','AMSR_RainRate[mm/hr]','Ped_AMSR_mean_RainRate[mm/hr]','Ped_AMSR_max_RainRate[mm/hr]',
                        'CloudSat_RainRate[mm/hr]','Ped_CloudSat_mean_RainRate[mm/hr]','Ped_CloudSat_max_RainRate[mm/hr]'
                        ]
        envfile = Path_to_nc+'/'+str(yr)+tmpm+'_Environmental_Parameters_V8.csv'
        with open(envfile, 'w', encoding='UTF8', newline='') as f:
                envwriter = csv.writer(f)
                envwriter.writerow(envfile_header)
                
        sfcfile_header = ['JDate','Granuel','STID','UTCTime','Day or Night','Longitude','Latitude','MeanFlag','Surface_Flag', 
                        'PED_ERA5_volumetric_soil_water_layer_1[m3/m3]','PED_ERA5_volumetric_soil_water_layer_2[m3/m3]',
                        'PED_ERA5_volumetric_soil_water_layer_3[m3/m3]','PED_ERA5_volumetric_soil_water_layer_4[m3/m3]',
                        'PED_ERA5_soil_type[type]','PED_ERA5_low_vegetation_cover','PED_ERA5_high_vegetation_cover',
                        'PED_ERA5_Type_vegetation[type]',
                        'ERA5_volumetric_soil_water_layer_1[m3/m3]','ERA5_volumetric_soil_water_layer_2[m3/m3]',
                        'ERA5_volumetric_soil_water_layer_3[m3/m3]','ERA5_volumetric_soil_water_layer_4[m3/m3]',
                        'ERA5_soil_type[type]','ERA5_low_vegetation_cover','ERA5_high_vegetation_cover',
                        'ERA5_Type_vegetation[type]','ERA_2m_TMK[K]','ERA_2m_DEW[K]','AUX_SkinTmk[K]']
        sfcfile = Path_to_nc+'/'+str(yr)+tmpm+'_Surface_Parameters_V8.csv'
        with open(sfcfile, 'w', encoding='UTF8', newline='') as f:
                sfcwriter = csv.writer(f)
                sfcwriter.writerow(sfcfile_header)

        sfcfile_header = ['JDate','Granuel','STID','UTCTime','Day or Night','Longitude','Latitude','MeanFlag','Surface_Flag', 
                        'Vertical_Tmk[K]','Vertical_dewpoint[K]','Vertical_SH[g/kg]','Vertical_U[m/s]','Vertical_V[m/s]']
        soundingfile = Path_to_nc+'/'+str(yr)+tmpm+'_Sounding_V8.csv'
        with open(soundingfile, 'w', encoding='UTF8', newline='') as f:
                sfcwriter = csv.writer(f)
                sfcwriter.writerow(sfcfile_header)
                
        ###READ IN LON/LAT of ERA5 and MERRRA2 ahead of the loop.
        ####----READIN GENERAL MERRA2 DATA----####
        ncfile1 = xr.load_dataset('/avalanche/bpan/MERRA2/MERRA2_300.inst3_2d_gas_Nx.20060815.nc4.nc4')
        merLAT  = ncfile1.variables['lat']
        merLON  = ncfile1.variables['lon']
        merTIME = ncfile1.variables['time']

        merLAT = np.array(merLAT,dtype=float)
        merLON = np.array(merLON,dtype=float)
        merTIME= np.array(merTIME,dtype=int)

        #Loop through dates in the month 
        for jdate in range(sday,eday+1):#sday,eday+1):
            #Path to input
            Path_to_file = Path_to_cld+str(yr)+str(jdate).zfill(3) #CLDCLASS
            #Convert date to certain format
            what = datetime.datetime.strptime(str(yr)+str(jdate), '%Y%j').date()
            ####----READIN MERRA2 DATA----####
            ncfile1 = xr.open_dataset('/avalanche/bpan/MERRA2/MERRA2_300.inst3_2d_gas_Nx.'+str(yr)+str(what.month).zfill(2)+str(what.day).zfill(2)+'.nc4.nc4')
            print('/avalanche/bpan/MERRA2/MERRA2_300.inst3_2d_gas_Nx.'+str(yr)+str(what.month).zfill(2)+str(what.day).zfill(2)+'.nc4.nc4')
            AODANA  = ncfile1.variables['AODANA']

            ####----READIN ERA5 DATA-----####
            #Read in ERA5 multi-level data
            ERAI = '/avalanche/bpan/ERA5/'+str(yr)+'/'+tmpm+'/ecmwf.grb2p40.'+str(yr)+str(what.month).zfill(2)+str(what.day).zfill(2)
            print(ERAI)
            grbs = xr.load_dataset(ERAI,engine='cfgrib')
            vertv= grbs.variables['w']

            #ERA-single level data readin
            ERAI = '/avalanche/bpan/ERA5/'+str(yr)+'/'+tmpm+'/ecmwf.grb2p40.'+str(yr)+str(what.month).zfill(2)+str(what.day).zfill(2)+'-single'
            print(ERAI)
            era_ncfile = xr.open_dataset(ERAI)#,engine='cfgrib')
            xlon  = era_ncfile.variables['longitude']
            ylat  = era_ncfile.variables['latitude']
            DEW2m  = era_ncfile.variables['d2m']
            TMK2m  = era_ncfile.variables['t2m']        
            SFLX   = -era_ncfile.variables['sshf']/3600 #W/m2
            LFLX   = -era_ncfile.variables['slhf']/3600 #W/m2
            LCL    =  era_ncfile.variables['cbh']
            CAPE   =  era_ncfile.variables['cape']
            CIN    =  era_ncfile.variables['cin']
            MSLP   =  era_ncfile.variables['msl']
            SST    =  era_ncfile.variables['sst']
            SOIL1  =  era_ncfile.variables['swvl1']
            SOIL2  =  era_ncfile.variables['swvl2']
            SOIL3  =  era_ncfile.variables['swvl3']
            SOIL4  =  era_ncfile.variables['swvl4']
            SOILTYPE= era_ncfile.variables['slt']
            LOWVEG  = era_ncfile.variables['cvl']
            HIGHVEG = era_ncfile.variables['cvh']       
            LOWTYPE = era_ncfile.variables['tvl']
            HIGHTYPE= era_ncfile.variables['tvh']

            SOIL1nan = np.where(SOIL1<0,np.nan,SOIL1)
            SOIL2nan = np.where(SOIL2<0,np.nan,SOIL2)
            SOIL3nan = np.where(SOIL3<0,np.nan,SOIL3)
            SOIL4nan = np.where(SOIL4<0,np.nan,SOIL4)
            intSOILD = np.array(np.round(SOILTYPE),dtype=int)

            #m3/m3
            soil_percent1= SOIL1nan/sat_soil[intSOILD]
            soil_percent2= SOIL2nan/sat_soil[intSOILD]
            soil_percent3= SOIL3nan/sat_soil[intSOILD]
            soil_percent4= SOIL4nan/sat_soil[intSOILD]

            SSTnan = np.where(SST>1000,np.nan,SST)
            LCLnan = np.where(LCL>9000,np.nan,LCL)

            #Convert everything to ndarray for round up
            CAPE = np.array(CAPE,dtype=float)
            CIN  = np.array(CIN,dtype=float)
            DEW2m= np.array(DEW2m,dtype=float)
            TMK2m= np.array(TMK2m,dtype=float)
            SFLX = np.array(SFLX,dtype=float)
            LFLX = np.array(LFLX,dtype=float)
            MSLP = np.array(MSLP,dtype=float)/100 #Convert from Pa to hPa
            Omega = np.array(vertv,dtype=float)
            AODANA= np.array(AODANA,dtype=float)

            granule = []  #Store temporary GRANULE name
            #If there is file presented on certain Jdate
            print(Path_to_juliet+str(jdate).zfill(3))
            if len(glob.glob(Path_to_juliet+str(jdate).zfill(3)+'*',recursive=True))>0:
                #loop through all files on the same date
                #FIND ALL GRANULE NUMBER AT THIS DAY
                for file in glob.glob(Path_to_juliet+str(jdate).zfill(3)+'*',recursive=True):
                    #Store file number
                    print(file)
                    granule.append(file[-11:-6])
                granule=np.array(granule,dtype=int)
                granule.sort() #Rank from low to high
                ####----2B-CLDCLASS----####
                #LOOP THROUGH ALL GRANULE IN THIS DAY
                for ig in range(len(granule)):
                    tmpgrn  = granule[ig]
                    cldfile = glob.glob(Path_to_file+'*'+str(granule[ig]).zfill(5)+'_CS_2B-CLDCLASS_GRANULE_P1_R05_*',recursive=True)
                    auxfile = glob.glob(Path_to_aux+str(yr)+str(jdate).zfill(3)+'*'+str(granule[ig]).zfill(5)+'_CS_ECMWF-AUX_GRANULE_P_R05_*',recursive=True)
                    if len(cldfile)>0:
                        #----ECMWF-AUX READIN-----#
                        f = HDF(auxfile[0], SDC.READ)
                        vs = f.vstart()
                        #Get Vdata info
                        vdata_skin  = vs.attach('Skin_temperature')
                        SkinT = vdata_skin[:]

                        # Get SDS info                       
                        file = SD(auxfile[0], SDC.READ)
                        # get CPR_Cloud_mask
                        sds_obj = file.select('Specific_humidity')
                        SH = sds_obj.get()

                        sds_obj = file.select('U_velocity')
                        Uspd = sds_obj.get()

                        sds_obj = file.select('V_velocity')
                        Vspd = sds_obj.get()

                        sds_obj = file.select('Temperature')
                        Tmk  = sds_obj.get()

                        sds_obj = file.select('Pressure')
                        Prs  = sds_obj.get()
                        #----END OF ECMWF-AUX FILE----#

                        #FILTER OUT -999 VALUES
                        SH   = np.where(SH<-900,np.nan,SH)
                        Uspd = np.where(Uspd<-900,np.nan,Uspd)
                        Vspd = np.where(Vspd<-900,np.nan,Vspd)
                        Prs  = np.where(Prs<-900,np.nan,Prs)
                        Tmk  = np.where(Tmk<-900,np.nan,Tmk)

                        #Calculate saturation vapor pressure
                        #Added 1-30-2023
                        #According to ERA-5 document:
                        #https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf
                        SatPrs_i = 611.21*np.exp(22.587*((Tmk-273.16)/(Tmk+0.7))) #with regard to ice 
                        SatPrs_l = 611.21*np.exp(17.502*((Tmk-273.16)/(Tmk-32.19))) #with regard to liquid 

                        alpha = ((Tmk-273.16+23)/23)**2
                        alpha = np.where(Tmk>273.16,1,alpha)
                        alpha = np.where(Tmk<250.16,0,alpha)

                        SatPrs = alpha*SatPrs_l+(1-alpha)*SatPrs_i

                        #Saturation MR 
                        SatMR = SatPrs/(Prs+epsilo*(Prs-SatPrs))
                        RH = SH/SatMR*100 #%
                        #Added 1-30-2023

                        #Calculate Integrated Water Vapor （IWV）
                        SH_layerM = (SH[:,1:]+SH[:,:-1])/2 #kg/kg
                        Prs_layerM= Prs[:,1:]-Prs[:,:-1]   #Pa
                        #Eliminate -999 values 
                        SH_layerNaN = np.where(SH_layerM<0,np.nan,SH_layerM)
                        Prs_layerNaN = np.where(Prs_layerM<0,np.nan,Prs_layerM)
                        #Integrated water vapor kg/m2
                        IWV = np.nansum(1/9.8*SH_layerNaN*Prs_layerNaN,axis=1)

                        SatMR_layerM = (SatMR[:,1:]+SatMR[:,:-1])/2 #kg/kg
                        #Integrated water vapor kg/m2
                        IWV_Sat = np.nansum(1/9.8*SatMR_layerM*Prs_layerNaN,axis=1)

                        #Calculation of dewpoint
                        MR = SH/(1-SH)
                        Vpr= vapor_pressure(Prs*units.Pa,MR*units('kg/kg'))
                        Td = dewpoint(Vpr)*units.K
                        
                        #Establish filters for various levels of PRS
                        #Total: Surface layer (1000-850 mb) and upper layer (300-150 mb)
                        #Low-level: Surface layer (1000-850 mb) and boundary layer (850-700 mb)
                        #Mid-level: Boundary layer (850-700 mb) and mid-troposphere (550-400 mb)
                        #Upper-level: mid-troposphere (550-400 mb) and upper troposphere (300-150 mb)
                        Prs150 = np.where(Prs>15000,Prs,np.nan)
                        f150prs= np.where(Prs150<30000,Prs150,np.nan)

                        Prs450 = np.where(Prs>40000,Prs,np.nan)
                        f450prs= np.where(Prs450<55000,Prs450,np.nan)

                        Prs850 = np.where(Prs>70000,Prs,np.nan)
                        f850prs= np.where(Prs850<85000,Prs850,np.nan)

                        Prs900 = np.where(Prs>85000,Prs,np.nan)
                        f900prs= np.where(Prs900<100000,Prs900,np.nan)

                        #Process Sounding data 
                        #surface (assumed here to be 1013 hPa), 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300, 200, and 100 hPa (a total of 16 levels).
                        Prs100 = np.where(Prs>7000,Prs,np.nan)
                        f100prs= np.where(Prs100<13000,Prs100,np.nan)

                        Prs200 = np.where(Prs>17000,Prs,np.nan)
                        f200prs= np.where(Prs200<23000,Prs200,np.nan)

                        Prs300 = np.where(Prs>27000,Prs,np.nan)
                        f300prs= np.where(Prs300<33000,Prs300,np.nan)

                        Prs400 = np.where(Prs>37000,Prs,np.nan)
                        f400prs= np.where(Prs400<43000,Prs400,np.nan)

                        Prs500 = np.where(Prs>47000,Prs,np.nan)
                        f500prs= np.where(Prs500<53000,Prs500,np.nan)

                        Prs550 = np.where(Prs>52000,Prs,np.nan)
                        f550prs= np.where(Prs550<58000,Prs550,np.nan)

                        Prs600 = np.where(Prs>57000,Prs,np.nan)
                        f600prs= np.where(Prs600<63000,Prs600,np.nan)

                        Prs650 = np.where(Prs>62000,Prs,np.nan)
                        f650prs= np.where(Prs650<68000,Prs650,np.nan)

                        Prs700 = np.where(Prs>67000,Prs,np.nan)
                        f700prs= np.where(Prs700<73000,Prs700,np.nan)

                        Prs750 = np.where(Prs>72000,Prs,np.nan)
                        f750prs= np.where(Prs750<78000,Prs750,np.nan)

                        Prs800 = np.where(Prs>77000,Prs,np.nan)
                        f800prs= np.where(Prs800<83000,Prs800,np.nan)

                        Prs850 = np.where(Prs>82000,Prs,np.nan)
                        f850prs0= np.where(Prs850<88000,Prs850,np.nan)

                        Prs900 = np.where(Prs>87000,Prs,np.nan)
                        f900prs0= np.where(Prs900<93000,Prs900,np.nan)

                        Prs950 = np.where(Prs>92000,Prs,np.nan)
                        f950prs= np.where(Prs950<98000,Prs950,np.nan)
                        #END OF ESTABLISH LAYER FILTERS 
                        
                        #DO WE NEED CLDCLASS READIN AT ALL?
                        #HEIGHT IS DEFINED AT ALL PIXELS, COMPARED TO ECMWF-AUX ONLY DEFINE OVERALL 
                        f = HDF(cldfile[0], SDC.READ)
                        vs = f.vstart()
                        vdata_lat  = vs.attach('Latitude')
                        vdata_long = vs.attach('Longitude')
                        vdata_elev = vs.attach('DEM_elevation')
                        vdata_flag = vs.attach('Navigation_land_sea_flag')
                        vdata_UTCstart = vs.attach('UTC_start')
                        vdata_proftime = vs.attach('Profile_time') #Seconds since the start of the granule.

                        lat = vdata_lat[:]
                        long = vdata_long[:]
                        flag = vdata_flag[:]
                        elev = vdata_elev[:]

                        UTC_start = vdata_UTCstart[:]
                        prof_time = vdata_proftime[:]

                        vdata_lat.detach() # "close" the vdata
                        vdata_long.detach() # "close" the vdata
                        vdata_flag.detach()
                        vdata_elev.detach()
                        vdata_UTCstart.detach()
                        vdata_proftime.detach()
                        vs.end() # terminate the vdata interface
                        f.close()

                        file = SD(cldfile[0], SDC.READ)
                        sds_obj = file.select('cloud_scenario') # select sds
                        cldscen = sds_obj.get()

                        sds_obj = file.select('CloudLayerType') # select sds
                        cldtype = sds_obj.get()

                        sds_obj = file.select('Height') # select sds
                        cldhght = sds_obj.get()
                        #----END OF 2B-CLDCLASS-----#

                        ###---CONVERT TO CLOUD TYPE
                        # 0 - No cloud
                        # 1 - cirrus
                        # 2 - Altostratus
                        # 3 - Altocumulus
                        # 4 - St
                        # 5 - Sc
                        # 6 - Cumulus
                        # 7 - Ns
                        # 8 - deep convection
                        cloudtype = np.zeros([len(cldhght),125])
                        for i in range(len(cldhght)):
                            for j in range(125):
                                bitmp= format(cldscen[i][j],'016b')[11:15]
                                cloudtype[i,j]= int(bitmp,2)

                        #Convert to array
                        cldlat = []
                        cldlon = []
                        cldflag= []
                        cldelev= []
                        for i in range(len(lat)):
                            cldlat.append(lat[i][0])
                            cldlon.append(long[i][0])
                            cldflag.append(flag[i][0])
                            cldelev.append(elev[i][0])
                        #Convert to array
                        intcldtype = np.array(cloudtype,dtype=int)
                        cldelev    = np.array(cldelev,dtype=float) #DEM_Height
                        cldflag    = np.array(cldflag,dtype=int)

                        ###--Change LON to 360
                        acldlon = np.array(cldlon)
                        cld360lon = np.where(acldlon<0,acldlon+360,acldlon)
                        ###--find the 5-point avg ground clutter
                        #filter out the -999 unassigned values
                        #DEM_height for the CO
                        groundorg  = np.where(cldelev<-100,0,cldelev)
                        ####NEED TO ADD FIVE POINT MAX FOR DEM_HEIGHT!!!
                        #to compensate the over-sampling of the ground
                        groundh    = slide_max(groundorg,window=5)
                        ###====END OF READIN CLDCLASS
                

                        #USED height from CLDCLASS!!

                        ###====CALCULATION OF TROPOPAUSE
                        #Thermo tropopause definition according to Holton et al (1995)
                        #Update to the lapse rate avg less than 2 K/km
                        #Calculate the temperature gradient
                        # (T_up - T_low)/del_grid - in K/km
                        #delTMK = (Tmk[:,:-1]-Tmk[:,1:])/240*1000
                        delTMK = -(Tmk[:,:-2]-Tmk[:,2:])/cldzgrid/2
                        SMTdelTMK = ndimage.filters.gaussian_filter(delTMK,1, mode='nearest')

                        #Ground and topography index array
                        longtopo = np.zeros(len(delTMK)) #tropopause index
                        ihghtdiff= np.zeros(len(delTMK)) #Find the difference between surface and height from Juliet's
                        groundidx= np.zeros(len(delTMK)) #search for ground indices
                        freezeidx= np.zeros(len(delTMK)) #freezing level indice
                        freezehgt= np.zeros(len(delTMK)) #freezing level height
                        topohght = np.zeros(len(delTMK)) #tropopause height
                        #Loop through all columns of the granule
                        for iclm in range(len(delTMK)):
                            ihghtdiff[iclm] = find_nearest(0,cldhght[iclm])
                            groundidx[iclm] = find_nearest(cldhght[iclm],groundh[iclm]) #Save topography vertical index
                            tmkcriteria = np.where(delTMK[iclm]<=2)[0] #Find the indices with less than 2K/km lapse rate
                            lapseratecri= tmkcriteria[np.where(tmkcriteria<max_idx)[0]] #Indices less than the max_idx [altitude higher than 7km]
                            tmptopo = [] #Temporary tropopause index
                            for i in np.arange(1,len(lapseratecri)):
                                #Loop from the lowest height to the highest     
                                idxinterest = lapseratecri[-i] #from lowest height index 
                                avglapserate= (delTMK[iclm][idxinterest]+delTMK[iclm][(idxinterest-num_levels_to_check):idxinterest])/2 
                                #Calculate the avg lapse rate between delTMK[index of interest] and delTMK[index of interest within 2 km ]
                                #If it's lapse rate is less than 2K/km within 2 km of the whole column
                                if len(np.where(avglapserate<=2)[0])==len(avglapserate):
                                    tmptopo.append(idxinterest)
                            #Default tropopause height 
                            if len(tmptopo)==0:
                                longtopo[iclm] = default_idx
                                topohght[iclm] = cldhght[iclm][default_idx]
                            else:
                                longtopo[iclm] = np.nanmax(tmptopo) #find the lowest height
                                topohght[iclm] = cldhght[iclm][int(np.nanmax(tmptopo))]

                            frezcriteria= np.where(Tmk[iclm]>=273)[0]  #Find the freezing level
                            #If there is no temperature >273 detected - use surface index 
                            if len(frezcriteria)==0:
                                freezeidx[iclm]= groundidx[iclm]
                                freezehgt[iclm]= cldhght[iclm][int(groundidx[iclm])]
                            else:
                                freezeidx[iclm]= int(np.min(frezcriteria))
                                freezehgt[iclm]= cldhght[iclm][int(np.min(frezcriteria))]
                        #####END OF TRACKING TROPOPAUSE
                        ###====START OF READIN JULIET's DATASET
                        source_filepath = Path_to_juliet+str(jdate).zfill(3)+'_'+str(granule[ig]).zfill(5)+'.nc.gz'
                        jlfile  = Path_to_juliet+str(jdate).zfill(3)+'_'+str(granule[ig]).zfill(5)+'.nc'
                        #Zip files
                        gunzip(source_filepath, jlfile)
                        print(jlfile)
                        ds = xr.open_dataset(jlfile)
                        xlon  = ds['Longitudes of CO']
                        ylat  = ds['Latitudes of CO']
                        hght  = ds['Heights']
                        stmtime  = ds['Time of Day']
                        #CO index with regard to AUX files 
                        ico_pixels = ds['Initial CO Profile Index']
                        fco_pixels = ds['Final CO Profile Index']

                        ini_core= ds['Initial Core Index'][:]
                        fin_core= ds['Final Core Index'][:]
                        colen   = ds['CO Length']
                        FullZ = ds['Full Z Profile']
                        freeze = ds['Mean Freezing Level']

                        #Eliminate CALIPSO only pixels
                        icali = ds['Initial CALIPSO-Only Region Index']
                        fcali = ds['Final CALIPSO-Only Region Index']
                        AMSR_rain    = ds['AMSR Rain Rates'] #(ngran,nCO, nCOpix)
                        CloudSat_rain= ds['CloudSat Rain Rates']
                        cttemp     = ds['94 GHz Brightness Temperatures'] #(ngran, nCO, narea, nCOpixext)
                        cttIR      = ds['11 micron Brightness Temperatures'] #(ngran, nCO, narea, nCOpixext)
                        clmIWP     = ds['Column-Integrated Ice Water Path']  #(ngran,nCO, nCOpix)
                        clmSWP     = ds['Column-Integrated Snow Water Path']  #(ngran,nCO, nCOpix)
                        clmSWPCERT = ds['Column-Integrated Snow Water Path Uncertainty']  #(ngran,nCO, nCOpix)
                        DEMhgt= ds['DEM Elevation']
                        os.remove(jlfile)
                        ####----LOOP THROUGH ALL CO----####
                        clmSWPnan = np.where(clmSWP>1E+9,np.nan,clmSWP)
                        clmSWPCERTnan = np.where(clmSWPCERT>1E+9,np.nan,clmSWPCERT)
                        #Filter out unwanted COs
                        for stid in range(len(colen)):
                            if (colen[stid]>=0):#Skip no CO cells
                                #Store the storm/core ID
                                stm_idx= np.ones(40)*-1
                                end_idx= np.ones(40)*-1
                                #IF there is more than zero pedestal
                                #MARK pedestal regions
                                #if conv_frac[stid]>0.02:
                                colen0 = int(colen[stid])

                                z_mod = np.ma.masked_all([len(ylat[stid]), len(FullZ[stid])])
                                ## CREATING A 2D ARRAY OF REFLECTIVITES FROM CLOUDSAT AND FAKE REFLECTIVITIES FROM CALIPSO
                                refl = np.where(FullZ[stid,0:colen0]>50,np.nan,FullZ[stid,0:colen0])
                                z_mod= np.where(refl>=-28,refl,np.nan) 

                                #Core index
                                icore = ini_core[stid]
                                fcore = fin_core[stid]
                                icore0= np.array(icore[icore>=0],dtype=int)
                                fcore0= np.array(fcore[fcore>=0],dtype=int)

                                #CALIPSO only index 
                                ical = icali[stid]
                                fcal = fcali[stid]
                                ical0= np.array(ical[ical>=0],dtype=int)
                                fcal0= np.array(fcal[fcal>=0],dtype=int)

                                #Check if CALIPSO has overlapping area
                                itest = np.where(ical0[1:]-fcal0[:-1]<0)[0]
                                itmpcali = ical0
                                ftmpcali = fcal0

                                #Eliminate the mistake indices in CALIPSO-only indices
                                if len(itest)>0:
                                    for c in np.arange(len(itest)):
                                        itmpcali = np.delete(itmpcali,itest[c]+1-c)
                                        ftmpcali = np.delete(ftmpcali,itest[c]-c)
                                    ical0 = itmpcali
                                    fcal0 = ftmpcali

                                #MARK CALIPSO ONLY region
                                if len(ical0)>0:
                                    #SPLIT CALIPSO PARTS
                                    if (fcal0[-1]<colen[stid]-1):
                                    #No CALIPSO at the end
                                        if ical0[0]!=0:
                                        #If there is no CALIPSO-only regionat the beginning
                                            for j in range(len(ical0)+1):
                                                #If there is CALIPSO-only region not at the beginning
                                                if j==0:
                                                    idx = 0
                                                    fdx = int(ical0[0])
                                                elif j ==len(ical0): #At the end
                                                    idx = int(fcal0[j-1])
                                                    fdx = int(colen0-1)
                                                else:
                                                    idx = int(fcal0[j-1])
                                                    fdx = int(ical0[j])

                                                if fdx-idx>1: #Eliminate short CloudSat-only region
                                                    for k in range(len(icore0)):
                                                        sidxcore = int(icore0[k])
                                                        fidxcore = int(fcore0[k])
                                                        if ylat[stid,0,sidxcore]>np.nanmin(ylat[stid,0,idx:fdx]) and ylat[stid,0,fidxcore]<=np.nanmax(ylat[stid,0,idx:fdx]):
                                                            stm_idx[j]=idx
                                                            end_idx[j]=fdx
                                        else:
                                        #If there is CALIPSO-only region at the beginning
                                            for j in range(len(ical0)):
                                                #If there is CALIPSO-only region not at the beginning
                                                if j ==(len(ical0)-1):
                                                    idx = int(fcal0[j])
                                                    fdx = int(colen0-1)
                                                else:
                                                    idx = int(fcal0[j])
                                                    fdx = int(ical0[j+1])

                                                if fdx-idx>1: #Eliminate short CloudSat-only region
                                                    for k in range(len(icore0)):
                                                        sidxcore = int(icore0[k])
                                                        fidxcore = int(fcore0[k])
                                                        if ylat[stid,0,sidxcore]>np.nanmin(ylat[stid,0,idx:fdx]) and ylat[stid,0,fidxcore]<=np.nanmax(ylat[stid,0,idx:fdx]):
                                                            stm_idx[j]=idx
                                                            end_idx[j]=fdx
                                    else:
                                        #There is CALIPSO at the end
                                        if ical0[0]!=0:
                                        #If there is no CALIPSO-only regionat the beginning
                                            for j in range(len(ical0)):
                                                if j==0:
                                                    idx = 0
                                                    fdx = int(ical0[0])
                                                else:
                                                    idx = int(fcal0[j-1])
                                                    fdx = int(ical0[j])
                                                if fdx-idx>1: #Eliminate short CloudSat-only region
                                                    for k in range(len(icore0)):
                                                        sidxcore = int(icore0[k])
                                                        fidxcore = int(fcore0[k])
                                                        if ylat[stid,0,sidxcore]>np.nanmin(ylat[stid,0,idx:fdx]) and ylat[stid,0,fidxcore]<=np.nanmax(ylat[stid,0,idx:fdx]):
                                                            stm_idx[j]=idx
                                                            end_idx[j]=fdx
                                        else:
                                        #If there is CALIPSO-only at the beginning
                                            for j in range(len(ical0)-1):
                                                idx= int(fcal0[j])
                                                fdx= int(ical0[j+1])
                                                if fdx-idx>1: #Eliminate short CloudSat-only region
                                                    for k in range(len(icore0)):
                                                        sidxcore = int(icore0[k])
                                                        fidxcore = int(fcore0[k])                                                        
                                                        if ylat[stid,0,sidxcore]>np.nanmin(ylat[stid,0,idx:fdx]) and ylat[stid,0,fidxcore]<=np.nanmax(ylat[stid,0,idx:fdx]):
                                                            stm_idx[j]=idx
                                                            end_idx[j]=fdx
                                else:
                                    #IF there is no CALIPSO PRESENTED
                                    idx = 0
                                    fdx = int(colen[stid])
                                    if fdx-idx>1: #Eliminate short CloudSat-only region
                                        if len(icore0)>0:
                                            for k in range(len(icore0)):
                                                sidxcore = int(icore0[k])
                                                fidxcore = int(fcore0[k])
                                                if ylat[stid,0,sidxcore]>np.nanmin(ylat[stid,0,idx:fdx]) and ylat[stid,0,fidxcore]<=np.nanmax(ylat[stid,0,idx:fdx]):
                                                    stm_idx[0]=idx
                                                    end_idx[0]=fdx

                                #MATCH THE LON/LAT OF EACH CLOUD OBJECT
                                #FROM SPLIT CALIPSO REGIONS
                                imprt_idx = np.where(stm_idx>=0)[0]
                                #LOOP THROUGH ALL CLOUDSAT ONLY REGION
                                for stm_srtidx in range(len(imprt_idx)):
                                    #The start and end indices of storm in Juliet's dataset
                                    #stmidx and stmfdx are within 0 and colen0
                                    #that are split by CALIPSO-only regions 
                                    stmidx = int(stm_idx[imprt_idx[stm_srtidx]])
                                    stmfdx = int(end_idx[imprt_idx[stm_srtidx]])

                                    #latitude/longitude indices wrt the whole granule
                                    #Location in the AUX file 
                                    #strt_ilat and end_ilat are corresponding to 0 and colen0
                                    strt_ilat = int(ico_pixels[stid])
                                    end_ilat  = int(fco_pixels[stid]+1)

                                    #find the longitude in 2B-CLDCLASS datase
                                    #convert longitude from -180,180 to 0,360
                                    tmplon  = cld360lon[strt_ilat:end_ilat]
                                    tmpxlon = np.where(xlon[stid,0,stmidx:stmfdx]<0,xlon[stid,0,stmidx:stmfdx]+360,xlon[stid,0,stmidx:stmfdx])

                                    #Match the corresponding Lon/Lat from Juliet's V1 to CLDCLASS index
                                    tmpa0  = np.where(np.isclose(np.nanmax(xlon[stid,0,stmidx:stmfdx]), acldlon[strt_ilat:end_ilat], rtol=10**-6, atol=10**-8)==True)[0]+strt_ilat
                                    tmpb0  = np.where(np.isclose(np.nanmin(xlon[stid,0,stmidx:stmfdx]), acldlon[strt_ilat:end_ilat], rtol=10**-6, atol=10**-8)==True)[0]+strt_ilat

                                    tmpa1  = np.where(np.isclose(np.nanmax(tmpxlon), tmplon, rtol=10**-6, atol=10**-8)==True)[0]+strt_ilat
                                    tmpb1  = np.where(np.isclose(np.nanmin(tmpxlon), tmplon, rtol=10**-6, atol=10**-8)==True)[0]+strt_ilat

                                    tmpa = np.array([tmpa0[0],tmpa1[0]])
                                    tmpb = np.array([tmpb0[0],tmpb1[0]])
                                    #If two methods does not give same results
                                    if abs(tmpa0[0]-tmpb0[0])!=abs(tmpa1[0]-tmpb1[0]):
                                        tmpidx = find_nearest(len(xlon[stid,0,stmidx:stmfdx]),[abs(tmpa0[0]-tmpb0[0]),abs(tmpa1[0]-tmpb1[0])])
                                    else:
                                        tmpidx = 0

                                    #Initial latitude indices wrt. AUX
                                    #Corresponding to stmidx and stmfdx
                                    if tmpa[tmpidx] > tmpb[tmpidx]:
                                        inilatidx = tmpb[tmpidx]
                                        endlatidx = tmpa[tmpidx]+1
                                    else:
                                        inilatidx = tmpa[tmpidx]
                                        endlatidx = tmpb[tmpidx]+1
                        #             print('Try the lon and lat index in AUX:',inilatidx,endlatidx,endlatidx-inilatidx)
                        #             print('CALIPSO-split:',stmidx,stmfdx,stmfdx-stmidx)
                        #             print('What about CO indices:',strt_ilat,end_ilat,end_ilat-strt_ilat,colen0)


                                    #Tropopause height indices from previous calculation
                                    #Traucate from AUX to CO length 
                                    tmptopoidx = longtopo[inilatidx:endlatidx]-ihghtdiff[inilatidx:endlatidx] + 75 #take into account for height different between full/partial
                                    tmptopoidx = np.where(tmptopoidx<0,0,tmptopoidx)
                                    tmptopoh   = topohght[inilatidx:endlatidx]/1000 #Tropopause height in km

                                    #DEM_height index for the CO
                                    tmpgroundidx = groundidx[inilatidx:endlatidx]-ihghtdiff[inilatidx:endlatidx] + 75
                                    tmpgroundidx = np.where(tmpgroundidx>75,75,tmpgroundidx)

                                    #Calcualte the distance between topography and tropopause
                                    Kthickness= tmpgroundidx-tmptopoidx
                                    #$$$ K_99= 6 pixels above the DEM_height
                                    K_99 = tmpgroundidx-6-1 #Take into account the additional pixel with python array
                                    #$$$ K_66 pixels; 50% of the troposphere thickness
                                    K_66 = np.array(tmptopoidx + Kthickness*0.5,dtype=int)
                                    #K_70 takes 60% of the tropospheric thickness 10km of 16km tropopause
                                    #according to Luo et al. (2008)
                                    K_70 = np.array(tmptopoidx + Kthickness*0.4,dtype=int)

                                    #Longitude/Latitude
                                    intcldlon = cldlon[inilatidx:endlatidx]
                                    intcldlat = cldlat[inilatidx:endlatidx]
                                    inttime   = prof_time[inilatidx:endlatidx]
                                    intflag   = cldflag[inilatidx:endlatidx]

                                    ###Mask out the non-cloud reflectivity
                                    #print('STMidx:',stmidx,stmfdx,'AUX:',inilatidx,endlatidx)
                                    tmprefl = np.where(z_mod[stmidx:stmfdx,:]>1E+8,-100,z_mod[stmidx:stmfdx,:])
                                    #Select reflectivity greater than -28dBZ
                                    ###USE CLDCLASS TO DEFINE CLOUD CLUSTERS
                                    filrefl = np.where(np.flip(tmprefl,axis=1)>-28,intcldtype[inilatidx:endlatidx,29:105],np.nan)
                                    #Threshold greater than 0 - No cloud in CLDCLASS
                                    #Find out the continous cloud clusters
                                    all_labels,num = define_cluster(filrefl,0)

                                    ###Find the cloud top height for the CO
                                    #Find the highest height with REFL>-28
                                    #Used CLDCLASS to define CTH
                                    tmpcth    = np.nanmin(np.where(filrefl>=0)[1])

                                    if tmpcth<=np.max(tmptopoidx)+trop_gap:
                                        #if the CO CTH is higher than the tropopause - gap
                                        #i.e.: CO CTH index<tropopause index+gap
                                        #Cloud top within 4km from the tropopause
                                        #Remember indices are in the opposite direction of height
                                        #Higher indices = lower altitude
                                        #Record the cloud label
                                        labels  = len(num)

                                        #LOOP THROUGH ALL LABELS
                                        #0 - is background $$$hopefully (when the CO is not the major area)
                                        for a in np.arange(1,labels):
                                            #Cloud indices wrt z_mod[stmidx:stmfdx,:]
                                            #Mark the beginning and ending indices of the CO
                                            strtcld = int(np.nanmin(np.where(all_labels==a)[0][:]))
                                            endcld  = int(np.nanmax(np.where(all_labels==a)[0][:]))

                                            #Cloud width has to be greater than 3 pixels
                                            #To find the indice with regard to ECWMF-AUX and MERRA2
                                            #only with the continously cloud pixel of label a
                                            if abs(endcld-strtcld)>3:
                                                if strtcld<endcld:
                                                    #STORE JULIET DATASET IDX
                                                    sjul_cld= int(strtcld+stmidx)
                                                    ejul_cld= int(endcld+stmidx)
                                                    #STORE THE CLOUDSAT IDX over the whole granule
                                                    sidx_aux= int(strtcld+inilatidx)
                                                    eidx_aux= int(endcld+inilatidx)
                                                    #STORE THE CLOUDS LON/LAT and TIME
                                                    slon_cld= intcldlon[strtcld]
                                                    elon_cld= intcldlon[endcld]
                                                    slat_cld= intcldlat[strtcld]
                                                    elat_cld= intcldlat[endcld]
                                                    time_cld= np.nanmean(inttime[strtcld:endcld]) #Second from the profile starts
                                                else:
                                                    #If the indices strtcld>endcld
                                                    #STORE JULIET DATASET IDX
                                                    sjul_cld= int(endcld+stmidx)
                                                    ejul_cld= int(strtcld+stmidx)
                                                    #STORE THE CLOUDSAT IDX over the whole granule
                                                    sidx_aux= int(endcld +inilatidx)
                                                    eidx_aux= int(strtcld+inilatidx)
                                                    #STORE THE CLOUDS LON/LAT and TIME
                                                    slon_cld= intcldlon[endcld]
                                                    elon_cld= intcldlon[strtcld]
                                                    slat_cld= intcldlat[endcld]
                                                    elat_cld= intcldlat[strtcld]
                                                    time_cld= np.nanmean(inttime[endcld:strtcld])

                                                #Search for this specific Cloud Label
                                                #If the lowest tropopause index+2 is lower than cloud top height index
                                                cthlabelcld = np.nanmin(np.where(all_labels==a)[1])
                                                topolabelcld= np.max(longtopo[sidx_aux:eidx_aux])+trop_gap

                                                #Search if the CTH of the current label
                                                #is greater than tropopause + gap
                                                ###Make sure the cloud reaches a specific height
                                                if cthlabelcld<=topolabelcld:
                                                    #The range is stmidx:stmfdx within 0-colen0
                                                    #Not for label=a region tho 
                                                    tmpcld    = np.where(all_labels==a,np.flip(z_mod[stmidx:stmfdx,:],axis=1),np.nan)
                                                    #Store height and temperature data
                                                    tmpTMK    = Tmk[inilatidx:endlatidx,29:105]
                                                    tmpauxhght= cldhght[inilatidx:endlatidx,29:105]

                                                    #DEM height from Juliet's dataset
                                                    #Eliminate negative values 
                                                    tmpDEMHght= np.where(DEMhgt[stid,stmidx:stmfdx]<0,0,DEMhgt[stid,stmidx:stmfdx])

                                                    #Rain Rate 
                                                    tmpAMSR0     = np.where(AMSR_rain[stid,stmidx:stmfdx]>1E5,0,AMSR_rain[stid,stmidx:stmfdx])
                                                    tmpAMSR      = np.where(tmpAMSR0<0,0,tmpAMSR0)

                                                    tmpCloudSatR0= np.where(CloudSat_rain[stid,stmidx:stmfdx]>1E5,0,CloudSat_rain[stid,stmidx:stmfdx])
                                                    tmpCloudSatR = np.where(tmpCloudSatR0<0,0,tmpCloudSatR0)


                                                    pedclm = [] #Store potential pedestal columns
                                                    #Search for less than three non-cloudy pixels
                                                    #Loop through all columns from stmidx to stmfdx
                                                    for icld in range(len(tmpcld)):
                                                        #Bottom to top indices 
                                                        Ktmp_66 = int(K_66[icld])
                                                        Ktmp_99 = int(K_99[icld])

                                                        #If there is continous cloud pixel till ground level=rain
                                                        #pedclm is with regard to stmidx:stmfdx and inilatidx:endlatidx
                                                        if ~np.all(np.isnan(tmpcld[icld,Ktmp_66:Ktmp_99])):
                                                            #find all continous cloudy pixels
                                                            howp    = group_consecutives(np.where(~np.isnan(tmpcld[icld,Ktmp_66:Ktmp_99]))[0])
                                                            #Find the beginning and end of a cloudy column within label=a
                                                            inanmin = np.nanmin(np.where(~np.isnan(tmpcld[icld,Ktmp_66:Ktmp_99]))[0])
                                                            inanmax = np.nanmax(np.where(~np.isnan(tmpcld[icld,Ktmp_66:Ktmp_99]))[0])

                                                            imax = 0 #Store the gap between cloudy pixels within each column
                                                            #If there is discontinuity within CO
                                                            if len(howp)>1:
                                                                for igroup in np.arange(1,len(howp)):
                                                                    #Calculate the gap between two consecutive cloud chunks
                                                                    tmph = np.nanmin(howp[igroup])-np.nanmax(howp[igroup-1])
                                                                    if imax<tmph:
                                                                        imax=tmph
                                                            #Find the pedestal columns
                                                            if imax<4 and len(howp)>1 and inanmax>=Ktmp_99-Ktmp_66-3 and inanmin<=3:
                                                                #if there is discont and non-cloudy pixel less than 3 or all continous
                                                                #The smallest and largest indices (lowest and highest height of the column)
                                                                #are near the K_99 and K_66
                                                                pedclm.append(icld)
                                                            elif imax==0 and len(howp)==1 and len(howp[0])>=Ktmp_99-Ktmp_66-2:
                                                                # if there is only one group from Ktmp_99 to Ktmp_66
                                                                # excluding the potential non-cloudy pixels at the edge
                                                                pedclm.append(icld)

                                                    #Find each pedclm width is greater than 3
                                                    hellp  =group_consecutives(pedclm)
                                                    pedlen =np.empty(len(hellp))
                                                    pedlenidx = []
                                                    for h in range(len(hellp)):
                                                        pedlen[h]=len(hellp[h])
                                                        if len(hellp[h])>3:
                                                            pedlenidx.extend(hellp[h])
                                                    pedlenidx = np.array(pedlenidx,dtype=int)

                                                    #Convective core definition by Hanii!
                                                    #Loop through all pedestal region
                                                    #convective core ETH10dBZ greater than 52% troposphere thickness
                                                    conv = []
                                                    for iconv in range(len(pedlenidx)):
                                                        pedidx  = pedlenidx[iconv]
                                                        #62.5% of the troposphere thickness 
                                                        #According to Luo et al. (2008) estimation
                                                        Kped_70 = int(K_70[pedidx]) 
                                                        ifthere = np.where(tmpcld[pedidx]>=10)[0]
                                                        #print('K_70 idx, 10dBZ index:',Kped_70,ifthere)
                                                        if len(ifthere)>0: #If there is REFL>10dBZ
                                                            #Highest ETH10dBZ height is greater than 62.5% of tropospheric thickness
                                                            #a small index value corresponds to a large height
                                                            if np.nanmin(ifthere)<=Kped_70:
                                                                conv.append(pedidx)                                
                                                    #Save the number of core and core width
                                                    contCov = group_consecutives(np.array(conv,dtype=int))
                                                    nCore   = len(contCov)
                                                    Core_wid= [] 
                                                    for iconvcore in range(nCore):
                                                        if len(contCov[iconvcore])>=2:
                                                            Core_wid.append(len(contCov[iconvcore])) 
                                                    #Save the continuos convective cores 
                                                    if len(Core_wid)==0:
                                                        nCore=0
                                                        Core_wid=0

                                                    #Save the number of pedestals and pedestal width 
                                                    contPed = group_consecutives(pedlenidx)
                                                    nPed    = len(contPed)
                                                    Ped_wid = [] 
                                                    for iped in range(nPed):
                                                        Ped_wid.append(len(contPed[iped]))
                                                    #Define K_85 within the pedestal columns;Kmax
                                                    #Loop through all pedstal columns
                                                    #Try to find Kmax, i.e., K85
                                                    if len(pedlenidx)>0:
                                                        #CTH and CBH within pedestal region
                                                        DEM_85     = [0]*len(pedlenidx)
                                                        CTHidx_85  = [0]*len(pedlenidx)
                                                        CBHidx_85  = [0]*len(pedlenidx)
                                                        AMSR_85    = [0]*len(pedlenidx)
                                                        CldRainR_85= [0]*len(pedlenidx)
                                                        CTH_85  = np.empty(len(pedlenidx))
                                                        CBH_85  = np.empty(len(pedlenidx))
                                                        K99_85  = np.empty(len(pedlenidx))
                                                        Ktmp_85 = np.empty(len(pedlenidx))
                                                        CTT_85  = np.empty(len(pedlenidx))
                                                        HGHT_85 = np.empty([len(pedlenidx),76])
                                                        FRZ_85  = np.empty(len(pedlenidx))
                                                        topo_85 = [99999]*len(pedlenidx)


                                                        #Store variables within the pedestal region 
                                                        for i85 in range(len(pedlenidx)):
                                                            CTHidx_85[i85] = int(np.nanmin(np.where(tmpcld[pedlenidx[i85]]>=-28)[0]))
                                                            CBHidx_85[i85] = int(np.nanmax(np.where(tmpcld[pedlenidx[i85]]>=-28)[0]))
                                                            #Temperature at cloud top 
                                                            CTT_85[i85] = tmpTMK[pedlenidx[i85],CTHidx_85[i85]]
                                                            CTH_85[i85] = tmpauxhght[pedlenidx[i85],CTHidx_85[i85]]
                                                            CBH_85[i85] = tmpauxhght[pedlenidx[i85],CBHidx_85[i85]]
                                                            HGHT_85[i85]= tmpauxhght[pedlenidx[i85]]
                                                            K99_85[i85] = K_99[pedlenidx[i85]]
                                                            DEM_85[i85] = tmpDEMHght[pedlenidx[i85]]
                                                            AMSR_85[i85]= tmpAMSR[pedlenidx[i85]]
                                                            CldRainR_85[i85]= tmpCloudSatR[pedlenidx[i85]]
                                                            topo_85[i85]    = tmptopoh[pedlenidx[i85]]
                                                            #$$$ 30% of the thickness from K_99
                                                            #$$$ DEM_height (=K99+7)+40%of thickness
                                                            Ktmp_85[i85]= np.array(K_99[pedlenidx[i85]] + 7 - Kthickness[pedlenidx[i85]]*0.3,dtype=int)
                                                            #Freezehght is across the whole AUX file 
                                                            FRZ_85[i85] = freezehgt[pedlenidx[i85]+inilatidx] 

                                                        #Use the highest height between Ktmp_85 and K99
                                                        #Find the smallest index= the highest height
                                                        K_85 = int(np.nanmin(np.concatenate((K99_85,Ktmp_85))))

                                                        ###Calculate the cutoff height index
                                                        icutoff = def_cutoff(tmpcld,K_85,0)

                                                        #If the cutoff height is defined in the CO
                                                        if ~np.isnan(icutoff):
                                                            icutoff = int(icutoff)

                                                            #Figure out the anvil width
                                                            labelcthidx = np.nanmin(np.where(tmpcld[:,:icutoff]>=-28)[1])  #Find CTH
                                                            maskcld   = np.where(~np.isnan(tmpcld[:,labelcthidx:icutoff]),1,0)

                                                            anvilwidth= len(np.where(np.sum(maskcld,axis=1)>0)[0]) #only account for the column with cloudy pixel

                                                            ianvilmin = np.nanmin(np.where(tmpcld[:,labelcthidx:icutoff]>=-28)[0])+stmidx
                                                            ianvilmax = np.nanmax(np.where(tmpcld[:,labelcthidx:icutoff]>=-28)[0])+stmidx

                                                            #strtcld and endcld are the label=a indices 
                                                            #CO DEM height max and mean
                                                            COmaxDEMhght = np.round(np.nanmax(tmpDEMHght[strtcld:endcld]),1)
                                                            COmeanDEMhght= np.round(np.nanmean(tmpDEMHght[strtcld:endcld]),1)

                                                            #CO rain rate max and mean
                                                            COmeanAMSRR  = np.round(np.nanmean(tmpAMSR[strtcld:endcld]),4)
                                                            COmeanCldRR  = np.round(np.nanmean(tmpCloudSatR[strtcld:endcld]),4)

                                                            #Assume the extreme values of CBH and CTH are 
                                                            #only within pedestal regions 
                                                            Pedmaxtopo   = np.round(np.nanmax(topo_85),1)
                                                            Pedmeantopo  = np.round(np.nanmean(topo_85),1)
                                                            COmincldbase = np.round(np.nanmin(CBH_85),1)
                                                            COmaxcldtop  = np.round(np.nanmax(CTH_85),1)
                                                            COcldbase = np.round(np.nanmean(CBH_85),1)
                                                            COcldtop  = np.round(np.nanmean(CTH_85),1)
                                                            COcutoff  = np.round(np.nanmean(HGHT_85[:,icutoff]),1)
                                                            COanvilw  = anvilwidth*cldgrid
                                                            COpedw    = len(pedlenidx)*cldgrid
                                                            COAUXCTT     = np.round(np.nanmean(CTT_85),1)
                                                            COAUXminCTT  = np.round(np.nanmin(CTT_85),1)
                                                            COAUXfreeze  = np.round(np.nanmean(FRZ_85),1)
                                                            PedmeanDEM   = np.round(np.nanmean(DEM_85),1)
                                                            PedmaxDEM    = np.round(np.nanmax(DEM_85),1)

                                                            #Pedestal average rain rate, max rain rate 
                                                            PedmeanAMSRR = np.round(np.nanmean(AMSR_85),4)
                                                            PedmaxAMSRR  = np.round(np.nanmax(AMSR_85),4)
                                                            PedmeanCldRR = np.round(np.nanmean(CldRainR_85),4)
                                                            PedmaxCldRR  = np.round(np.nanmax(CldRainR_85),4)

                                                            #AUX within label=a region
                                                            Meanflag= np.round(np.nanmean(cldflag[sidx_aux:eidx_aux]),2)
                                                            flagBin = np.bincount(list(cldflag[sidx_aux:eidx_aux].ravel()),minlength=4)
                                                            mLON  = np.round(np.nanmean(xlon[stid,0,sjul_cld:ejul_cld]),2)
                                                            mLAT  = np.round(np.nanmean(ylat[stid,0,sjul_cld:ejul_cld]),2)

                                                            #Convert from -180 to 180 in longitude for ERA5
                                                            if slon_cld<0:
                                                                tmp360slon=slon_cld+360
                                                            else:
                                                                tmp360slon=slon_cld

                                                            if elon_cld<0:
                                                                tmp360elon=elon_cld+360
                                                            else:
                                                                tmp360elon=elon_cld

                                                            #Check the time
                                                            cldtime    =  int(np.round((time_cld+UTC_start[0][0])/3600))
                                                            clditime   =  find_nearest(cldtime,ERAtime)

                                                            #Environment variables from ECMWF-AUX
                                                            mSkinT   = np.round(np.nanmean(SkinT[sidx_aux:eidx_aux]),2)
                                                            mIWV      = np.round(np.nanmean(IWV[sidx_aux:eidx_aux]),2)
                                                            mIWVRatio = np.round(np.nanmean(IWV[sidx_aux:eidx_aux]/IWV_Sat[sidx_aux:eidx_aux])*100,2)

                                                            #Environmental Variable calculation from ERA5
                                                            #Define surface variables
                                                            icnt_sst = 0
                                                            icounter = 0
                                                            icnt_LCL = 0
                                                            icnt_CIN = 0
                                                            #Variable names
                                                            mCAPE= 0 
                                                            mCIN = 0
                                                            mLCL = 0 
                                                            mMSLP= 0
                                                            mSFLX= 0
                                                            mLFLX= 0
                                                            momega=0
                                                            mDEW2m=0
                                                            mTMK2m=0
                                                            mSST  =0
                                                            mAOD  =0

                                                            #List surface variables
                                                            vegeperc = np.zeros(21)
                                                            soil1perc= np.zeros(8)
                                                            soil2perc= np.zeros(8)
                                                            soil3perc= np.zeros(8)
                                                            soil4perc= np.zeros(8)
                                                            STYPE    =[] #Create empty list for soily type
                                                            LOWVTYPE =[] #create empty list for low vegetation
                                                            HIGHVTYPE=[] #Create empty list for high vegetation

                                                            #List surface variables for pedestal region
                                                            #over the pedestal region
                                                            Pedvegeperc = np.zeros(21)
                                                            Pedsoil1perc= np.zeros(8)
                                                            Pedsoil2perc= np.zeros(8)
                                                            Pedsoil3perc= np.zeros(8)
                                                            Pedsoil4perc= np.zeros(8)
                                                            PedSTYPE    =[] #Create empty list for soily type
                                                            PedLOWVTYPE =[] #create empty list for low vegetation
                                                            PedHIGHVTYPE=[] #Create empty list for high vegetation

                                                            #Calculate Pedestal region soil and vegetation cover 
                                                            #Check the time
                                                            for i85 in range(len(pedlenidx)):
                                                                #With regard to the whole AUX file
                                                                pedi85 = int(pedlenidx[i85])+inilatidx
                                                                #Find nearest ERA5 indices
                                                                if cldlon[pedi85]<0:
                                                                    era5lon = find_nearest(cldlon[pedi85]+360,lons)
                                                                else:
                                                                    era5lon = find_nearest(cldlon[pedi85],lons)
                                                                era5lat = find_nearest(cldlat[pedi85],lats)

                                                                #Save the soil and vegetation information
                                                                PedintSOIL = int(np.round(SOILTYPE[clditime,era5lat,era5lon]))
                                                                PedSTYPE.append(PedintSOIL)

                                                                #If the surface is not ocean
                                                                #greater than 0.5
                                                                if ERAMask[era5lat,era5lon]>0.5:
                                                                    #Soil coverage 
                                                                    Pedsoil1perc[PedintSOIL] = Pedsoil1perc[PedintSOIL] + np.round(soil_percent1[clditime,era5lat,era5lon],2)
                                                                    Pedsoil2perc[PedintSOIL] = Pedsoil2perc[PedintSOIL] + np.round(soil_percent2[clditime,era5lat,era5lon],2)
                                                                    Pedsoil3perc[PedintSOIL] = Pedsoil3perc[PedintSOIL] + np.round(soil_percent3[clditime,era5lat,era5lon],2)
                                                                    Pedsoil4perc[PedintSOIL] = Pedsoil4perc[PedintSOIL] + np.round(soil_percent4[clditime,era5lat,era5lon],2)

                                                                    #Vegetation coverage over the Pedestal region
                                                                    intLOW = int(np.round(LOWTYPE[clditime,era5lat,era5lon]))
                                                                    intHIGH= int(np.round(HIGHTYPE[clditime,era5lat,era5lon]))
                                                                    PedLOWVTYPE.append(intLOW)
                                                                    PedHIGHVTYPE.append(intHIGH)
                                                                    Pedvegeperc[intLOW]  = Pedvegeperc[intLOW]+LOWVEG[clditime,era5lat,era5lon]
                                                                    Pedvegeperc[intHIGH] = Pedvegeperc[intHIGH]+HIGHVEG[clditime,era5lat,era5lon]

                                                            mPedSOILTYPE = np.bincount(list(PedSTYPE),minlength=8)
                                                            mPedVEGTYPE  = np.bincount(list(np.array([PedLOWVTYPE,PedHIGHVTYPE]).ravel()),minlength=21) #there is 20 types of vegetation, including 0 (non-veg)
                                                            PedbinLOWVEG = np.bincount(list(PedLOWVTYPE),minlength=21)
                                                            PedbinHIGHVEG= np.bincount(list(PedHIGHVTYPE),minlength=21)

                                                            #Loop through all columns
                                                            merUTC     =  find_nearest(clditime,merTIME/60)
                                                            #Label==a within the Aux file 
                                                            for igrid in range(sidx_aux,eidx_aux):
                                                                icounter = icounter + 1

                                                                #Find nearest MERRA2 grid 
                                                                imerlon = find_nearest(merLON,cldlon[igrid])
                                                                imerlat = find_nearest(merLAT,cldlat[igrid])
                                                                mAOD    = mAOD + AODANA[merUTC,imerlat,imerlon]

                                                                #Find nearest ERA5 indices
                                                                if cldlon[igrid]<0:
                                                                    era5lon = find_nearest(cldlon[igrid]+360,lons)
                                                                else:
                                                                    era5lon = find_nearest(cldlon[igrid],lons)
                                                                era5lat = find_nearest(cldlat[igrid],lats)

                                                                #Add the environmental variables 
                                                                mCAPE = mCAPE+ CAPE[clditime,era5lat,era5lon]
                                                                mMSLP = mMSLP+ MSLP[clditime,era5lat,era5lon]
                                                                mSFLX = mSFLX+ SFLX[clditime,era5lat,era5lon]
                                                                mLFLX = mLFLX+ LFLX[clditime,era5lat,era5lon]
                                                                momega= momega+Omega[clditime,era5lat,era5lon]
                                                                mDEW2m= mDEW2m+DEW2m[clditime,era5lat,era5lon]
                                                                mTMK2m= mTMK2m+TMK2m[clditime,era5lat,era5lon]

                                                                if np.isnan(LCLnan[clditime,era5lat,era5lon]):
                                                                    mLCL  = mLCL
                                                                else:
                                                                    mLCL = mLCL+ LCLnan[clditime,era5lat,era5lon]
                                                                    icnt_LCL = icnt_LCL + 1

                                                                if np.isnan(CIN[clditime,era5lat,era5lon]):
                                                                    mCIN  = mCIN
                                                                else:
                                                                    mCIN = mCIN + CIN[clditime,era5lat,era5lon]
                                                                    icnt_CIN = icnt_CIN + 1

                                                                intSOIL = int(np.round(SOILTYPE[clditime,era5lat,era5lon]))
                                                                STYPE.append(intSOIL)

                                                                #If the surface is not ocean
                                                                #greater than 0.5
                                                                if ERAMask[era5lat,era5lon]>0.5:
                                                                    #Soil coverage 
                                                                    soil1perc[intSOIL] = soil1perc[intSOIL] + soil_percent1[clditime,era5lat,era5lon]
                                                                    soil2perc[intSOIL] = soil2perc[intSOIL] + soil_percent2[clditime,era5lat,era5lon]
                                                                    soil3perc[intSOIL] = soil3perc[intSOIL] + soil_percent3[clditime,era5lat,era5lon]
                                                                    soil4perc[intSOIL] = soil4perc[intSOIL] + soil_percent4[clditime,era5lat,era5lon]

                                                                    #Vegetation coverage 
                                                                    intLOW = int(np.round(LOWTYPE[clditime,era5lat,era5lon]))
                                                                    intHIGH= int(np.round(HIGHTYPE[clditime,era5lat,era5lon]))
                                                                    LOWVTYPE.append(intLOW)
                                                                    HIGHVTYPE.append(intHIGH)
                                                                    vegeperc[intLOW]  = vegeperc[intLOW]+LOWVEG[clditime,era5lat,era5lon]
                                                                    vegeperc[intHIGH] = vegeperc[intHIGH]+HIGHVEG[clditime,era5lat,era5lon]
                                                                else: #If over the ocean surface!
                                                                    icnt_sst = icnt_sst + 1
                                                                    mSST=mSST+SSTnan[clditime,era5lat,era5lon] 

                                                            #Average Environmental variables:
                                                            mCAPE = np.round(mCAPE/icounter,2)
                                                            mMSLP = np.round(mMSLP/icounter,2)
                                                            mSFLX = np.round(mSFLX/icounter,2)
                                                            mLFLX = np.round(mLFLX/icounter,2)
                                                            momega= np.round(momega/icounter,2)
                                                            mDEW2m= np.round(mDEW2m/icounter,2)
                                                            mTMK2m= np.round(mTMK2m/icounter,2)
                                                            mAOD  = np.round(mAOD/icounter,2)

                                                            if icnt_sst==0 or Meanflag==2:
                                                                mSST=-999
                                                            else:
                                                                mSST  = np.round(mSST/icnt_sst,2)

                                                            if icnt_LCL==0:
                                                                mLCL=-999
                                                            else:
                                                                mLCL  = np.round(mLCL/icnt_LCL,2)

                                                            if icnt_CIN==0:
                                                                mCIN=-999
                                                            else:
                                                                mCIN  = np.round(mCIN/icnt_CIN,2)
                                                            #7 kind of soil, including 0
                                                            mSOILTYPE = np.bincount(list(STYPE),minlength=8)
                                                            mVEGTYPE  = np.bincount(list(np.array([LOWVTYPE,HIGHVTYPE]).ravel()),minlength=21) #there is 20 types of vegetation, including 0 (non-veg)
                                                            binLOWVEG = np.bincount(list(LOWVTYPE),minlength=21)
                                                            binHIGHVEG= np.bincount(list(HIGHVTYPE),minlength=21)

                                                            #GET INFORMATION FROM ECMWF-AUX
                                                            mRH  = RH[sidx_aux:eidx_aux]
                                                            mSH  = SH[sidx_aux:eidx_aux]*1000 #convert from kg/kg to g/kg
                                                            mUspd= Uspd[sidx_aux:eidx_aux]
                                                            mVspd= Vspd[sidx_aux:eidx_aux]

                                                            mttlRH  = np.round(np.nanmean(mRH[:,29:105]),1)
                                                            mupRH   = np.round(np.nanmean(mRH[~np.isnan(f450prs[sidx_aux:eidx_aux])]),1) #450-550
                                                            mmidRH  = np.round(np.nanmean(mRH[~np.isnan(f850prs[sidx_aux:eidx_aux])]),1) #750-850
                                                            mlowRH  = np.round(np.nanmean(mRH[~np.isnan(f900prs[sidx_aux:eidx_aux])]),1) #900-1000

                                                            mttlSH  = np.round(np.nanmean(mSH[:,29:105]),4)
                                                            mupSH   = np.round(np.nanmean(mSH[~np.isnan(f450prs[sidx_aux:eidx_aux])]),4) #450-550
                                                            mmidSH  = np.round(np.nanmean(mSH[~np.isnan(f850prs[sidx_aux:eidx_aux])]),4) #750-850
                                                            mlowSH  = np.round(np.nanmean(mSH[~np.isnan(f900prs[sidx_aux:eidx_aux])]),4) #900-1000

                                                            #Get direction of VWS
                                                            Uspd150 = np.nanmean(mUspd[~np.isnan(f150prs[sidx_aux:eidx_aux])])
                                                            Uspd450 = np.nanmean(mUspd[~np.isnan(f450prs[sidx_aux:eidx_aux])])
                                                            Uspd850 = np.nanmean(mUspd[~np.isnan(f850prs[sidx_aux:eidx_aux])])
                                                            Uspd900 = np.nanmean(mUspd[~np.isnan(f900prs[sidx_aux:eidx_aux])])

                                                            Vspd150 = np.nanmean(mVspd[~np.isnan(f150prs[sidx_aux:eidx_aux])])
                                                            Vspd450 = np.nanmean(mVspd[~np.isnan(f450prs[sidx_aux:eidx_aux])])
                                                            Vspd850 = np.nanmean(mVspd[~np.isnan(f850prs[sidx_aux:eidx_aux])])
                                                            Vspd900 = np.nanmean(mVspd[~np.isnan(f900prs[sidx_aux:eidx_aux])])


                                                            mttlVWS = np.round(np.nanmean(np.sqrt((Uspd900-Uspd150)**2+(Vspd900-Vspd150)**2)),3) #200-300mb 925-1000mb
                                                            mlowVWS = np.round(np.nanmean(np.sqrt((Uspd900-Uspd850)**2+(Vspd900-Vspd850)**2)),3) #700-850mb 925-1000mb
                                                            mmidVWS = np.round(np.nanmean(np.sqrt((Uspd850-Uspd450)**2+(Vspd850-Vspd450)**2)),3) #300-500mb 700-850mb
                                                            mupVWS  = np.round(np.nanmean(np.sqrt((Uspd450-Uspd150)**2+(Vspd450-Vspd150)**2)),3) #300-500mb 200-300mb

                                                            #Direction of the VWS angle
                                                            #Direction ranges from -180 to 180 deg...
                                                            mdirttlVWS = np.round(np.arctan2(np.nanmean(Uspd150-Uspd900),np.nanmean(Vspd150-Vspd900))/np.pi*180,3)
                                                            mdirlowVWS = np.round(np.arctan2(np.nanmean(Uspd850-Uspd900),np.nanmean(Vspd850-Vspd900))/np.pi*180,3)
                                                            mdirmidVWS = np.round(np.arctan2(np.nanmean(Uspd450-Uspd850),np.nanmean(Vspd450-Vspd850))/np.pi*180,3)
                                                            mdirupVWS  = np.round(np.arctan2(np.nanmean(Uspd150-Uspd450),np.nanmean(Vspd150-Vspd450))/np.pi*180,3)

                                                            #Sounding Temperature, dewpoint, U, V
                                                            #Temperature
                                                            Tmk_pts     = Tmk[sidx_aux:eidx_aux]
                                                            Tmk_pts950  = np.round(np.nanmean(Tmk_pts[~np.isnan(f950prs[sidx_aux:eidx_aux])]),2)
                                                            Tmk_pts900  = np.round(np.nanmean(Tmk_pts[~np.isnan(f900prs0[sidx_aux:eidx_aux])]),2)
                                                            Tmk_pts850  = np.round(np.nanmean(Tmk_pts[~np.isnan(f850prs0[sidx_aux:eidx_aux])]),2)
                                                            Tmk_pts800  = np.round(np.nanmean(Tmk_pts[~np.isnan(f800prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts750  = np.round(np.nanmean(Tmk_pts[~np.isnan(f750prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts700  = np.round(np.nanmean(Tmk_pts[~np.isnan(f700prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts650  = np.round(np.nanmean(Tmk_pts[~np.isnan(f650prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts600  = np.round(np.nanmean(Tmk_pts[~np.isnan(f600prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts550  = np.round(np.nanmean(Tmk_pts[~np.isnan(f550prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts500  = np.round(np.nanmean(Tmk_pts[~np.isnan(f500prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts400  = np.round(np.nanmean(Tmk_pts[~np.isnan(f400prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts300  = np.round(np.nanmean(Tmk_pts[~np.isnan(f300prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts200  = np.round(np.nanmean(Tmk_pts[~np.isnan(f200prs[sidx_aux:eidx_aux])]),2) 
                                                            Tmk_pts100  = np.round(np.nanmean(Tmk_pts[~np.isnan(f100prs[sidx_aux:eidx_aux])]),2)                                     
                                                            #Dewpoint 
                                                            Td_pts     = Td[sidx_aux:eidx_aux]
                                                            Td_pts950  = np.round(np.nanmean(Td_pts[~np.isnan(f950prs[sidx_aux:eidx_aux])]),2)
                                                            Td_pts900  = np.round(np.nanmean(Td_pts[~np.isnan(f900prs0[sidx_aux:eidx_aux])]),2)
                                                            Td_pts850  = np.round(np.nanmean(Td_pts[~np.isnan(f850prs0[sidx_aux:eidx_aux])]),2)
                                                            Td_pts800  = np.round(np.nanmean(Td_pts[~np.isnan(f800prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts750  = np.round(np.nanmean(Td_pts[~np.isnan(f750prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts700  = np.round(np.nanmean(Td_pts[~np.isnan(f700prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts650  = np.round(np.nanmean(Td_pts[~np.isnan(f650prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts600  = np.round(np.nanmean(Td_pts[~np.isnan(f600prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts550  = np.round(np.nanmean(Td_pts[~np.isnan(f550prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts500  = np.round(np.nanmean(Td_pts[~np.isnan(f500prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts400  = np.round(np.nanmean(Td_pts[~np.isnan(f400prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts300  = np.round(np.nanmean(Td_pts[~np.isnan(f300prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts200  = np.round(np.nanmean(Td_pts[~np.isnan(f200prs[sidx_aux:eidx_aux])]),2) 
                                                            Td_pts100  = np.round(np.nanmean(Td_pts[~np.isnan(f100prs[sidx_aux:eidx_aux])]),2)
                                                            #Specific humidity  
                                                            SH_pts     = SH[sidx_aux:eidx_aux]*1000
                                                            SH_pts950  = np.round(np.nanmean(SH_pts[~np.isnan(f950prs[sidx_aux:eidx_aux])]),4)
                                                            SH_pts900  = np.round(np.nanmean(SH_pts[~np.isnan(f900prs0[sidx_aux:eidx_aux])]),4)
                                                            SH_pts850  = np.round(np.nanmean(SH_pts[~np.isnan(f850prs0[sidx_aux:eidx_aux])]),4)
                                                            SH_pts800  = np.round(np.nanmean(SH_pts[~np.isnan(f800prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts750  = np.round(np.nanmean(SH_pts[~np.isnan(f750prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts700  = np.round(np.nanmean(SH_pts[~np.isnan(f700prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts650  = np.round(np.nanmean(SH_pts[~np.isnan(f650prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts600  = np.round(np.nanmean(SH_pts[~np.isnan(f600prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts550  = np.round(np.nanmean(SH_pts[~np.isnan(f550prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts500  = np.round(np.nanmean(SH_pts[~np.isnan(f500prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts400  = np.round(np.nanmean(SH_pts[~np.isnan(f400prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts300  = np.round(np.nanmean(SH_pts[~np.isnan(f300prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts200  = np.round(np.nanmean(SH_pts[~np.isnan(f200prs[sidx_aux:eidx_aux])]),4) 
                                                            SH_pts100  = np.round(np.nanmean(SH_pts[~np.isnan(f100prs[sidx_aux:eidx_aux])]),4)                                     
                                                            #U 
                                                            Uspd_pts     = Uspd[sidx_aux:eidx_aux]
                                                            Uspd_pts950  = np.round(np.nanmean(Uspd_pts[~np.isnan(f950prs[sidx_aux:eidx_aux])]),2)
                                                            Uspd_pts900  = np.round(np.nanmean(Uspd_pts[~np.isnan(f900prs0[sidx_aux:eidx_aux])]),2)
                                                            Uspd_pts850  = np.round(np.nanmean(Uspd_pts[~np.isnan(f850prs0[sidx_aux:eidx_aux])]),2)
                                                            Uspd_pts800  = np.round(np.nanmean(Uspd_pts[~np.isnan(f800prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts750  = np.round(np.nanmean(Uspd_pts[~np.isnan(f750prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts700  = np.round(np.nanmean(Uspd_pts[~np.isnan(f700prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts650  = np.round(np.nanmean(Uspd_pts[~np.isnan(f650prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts600  = np.round(np.nanmean(Uspd_pts[~np.isnan(f600prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts550  = np.round(np.nanmean(Uspd_pts[~np.isnan(f550prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts500  = np.round(np.nanmean(Uspd_pts[~np.isnan(f500prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts400  = np.round(np.nanmean(Uspd_pts[~np.isnan(f400prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts300  = np.round(np.nanmean(Uspd_pts[~np.isnan(f300prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts200  = np.round(np.nanmean(Uspd_pts[~np.isnan(f200prs[sidx_aux:eidx_aux])]),2) 
                                                            Uspd_pts100  = np.round(np.nanmean(Uspd_pts[~np.isnan(f100prs[sidx_aux:eidx_aux])]),2)
                                                            #V 
                                                            Vspd_pts     = Vspd[sidx_aux:eidx_aux]
                                                            Vspd_pts950  = np.round(np.nanmean(Vspd_pts[~np.isnan(f950prs[sidx_aux:eidx_aux])]),2)
                                                            Vspd_pts900  = np.round(np.nanmean(Vspd_pts[~np.isnan(f900prs0[sidx_aux:eidx_aux])]),2)
                                                            Vspd_pts850  = np.round(np.nanmean(Vspd_pts[~np.isnan(f850prs0[sidx_aux:eidx_aux])]),2)
                                                            Vspd_pts800  = np.round(np.nanmean(Vspd_pts[~np.isnan(f800prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts750  = np.round(np.nanmean(Vspd_pts[~np.isnan(f750prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts700  = np.round(np.nanmean(Vspd_pts[~np.isnan(f700prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts650  = np.round(np.nanmean(Vspd_pts[~np.isnan(f650prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts600  = np.round(np.nanmean(Vspd_pts[~np.isnan(f600prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts550  = np.round(np.nanmean(Vspd_pts[~np.isnan(f550prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts500  = np.round(np.nanmean(Vspd_pts[~np.isnan(f500prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts400  = np.round(np.nanmean(Vspd_pts[~np.isnan(f400prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts300  = np.round(np.nanmean(Vspd_pts[~np.isnan(f300prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts200  = np.round(np.nanmean(Vspd_pts[~np.isnan(f200prs[sidx_aux:eidx_aux])]),2) 
                                                            Vspd_pts100  = np.round(np.nanmean(Vspd_pts[~np.isnan(f100prs[sidx_aux:eidx_aux])]),2) 

                                                            #Combine the Variables into an array
                                                            Tmk_arr = np.array([Tmk_pts950,Tmk_pts900,Tmk_pts850,Tmk_pts800,Tmk_pts750,Tmk_pts700,Tmk_pts650,Tmk_pts600
                                                                                ,Tmk_pts550,Tmk_pts500,Tmk_pts400,Tmk_pts300,Tmk_pts200,Tmk_pts100])
                                                            Td_arr = np.array([Td_pts950,Td_pts900,Td_pts850,Td_pts800,Td_pts750,Td_pts700,Td_pts650,Td_pts600
                                                                            ,Td_pts550,Td_pts500,Td_pts400,Td_pts300,Td_pts200,Td_pts100])
                                                            U_arr = np.array([Uspd_pts950,Uspd_pts900,Uspd_pts850,Uspd_pts800,Uspd_pts750,Uspd_pts700,Uspd_pts650,Uspd_pts600
                                                                            ,Uspd_pts550,Uspd_pts500,Uspd_pts400,Uspd_pts300,Uspd_pts200,Uspd_pts100])
                                                            V_arr = np.array([Vspd_pts950,Vspd_pts900,Vspd_pts850,Vspd_pts800,Vspd_pts750,Vspd_pts700,Vspd_pts650,Vspd_pts600
                                                                            ,Vspd_pts550,Vspd_pts500,Vspd_pts400,Vspd_pts300,Vspd_pts200,Vspd_pts100])
                                                            SH_arr = np.array([SH_pts950,SH_pts900,SH_pts850,SH_pts800,SH_pts750,SH_pts700,SH_pts650,SH_pts600
                                                                            ,SH_pts550,SH_pts500,SH_pts400,SH_pts300,SH_pts200,SH_pts100])

                                                            #GET ENVIRONMENTAL VARIABLE FROM JULIETS DATASET
                                                            #If there is no ice
                                                            if np.all(clmIWP[stid][sjul_cld:ejul_cld]>1E9):
                                                                COmeanIWP=0
                                                            else:
                                                                COmeanIWP  = np.round(np.nanmean(clmIWP[stid][sjul_cld:ejul_cld]),3)

                                                            #If there is no snow
                                                            if np.all(clmSWP[stid][sjul_cld:ejul_cld]>1E9):
                                                                COmeanSWP=0
                                                                COmeanSWPC = 0
                                                            else:
                                                                COmeanSWP  = np.round(np.nanmean(clmSWPnan[stid][sjul_cld:ejul_cld]),3)
                                                                COmeanSWPC = np.round(np.nanmean(clmSWPCERTnan[stid][sjul_cld:ejul_cld]),3)

                                                            if ~np.all(cttemp[stid,0,sjul_cld:ejul_cld]>1E9):
                                                                COmeanCTT  = np.round(np.nanmean(cttemp[stid,0,sjul_cld:ejul_cld]),3)
                                                                COminCTT   = np.round(np.nanmin(cttemp[stid,0,sjul_cld:ejul_cld]),3)                                                    
                                                            else:
                                                                COmeanCTT  =0
                                                                COminCTT   =0

                                                            if ~np.all(cttIR[stid,0,sjul_cld:ejul_cld]>1E9):
                                                                cttmeanIR  = np.round(np.nanmean(cttIR[stid,0,sjul_cld:ejul_cld]),3)
                                                                cttminIR   = np.round(np.nanmin(cttIR[stid,0,sjul_cld:ejul_cld]),3)
                                                            else:
                                                                cttmeanIR  =0
                                                                cttminIR   =0

                                                            #csv
                                                            add_SFC_csv(sfcfile,jdate,granule[ig],stid,ERAtime[clditime],int(stmtime[stid]),mLON,mLAT,Meanflag,flagBin,
                                                                        Pedsoil1perc,Pedsoil2perc,Pedsoil3perc,Pedsoil4perc,mPedSOILTYPE,
                                                                        PedbinLOWVEG,PedbinHIGHVEG,mPedVEGTYPE,
                                                                        soil1perc,soil2perc,soil3perc,soil4perc,mSOILTYPE,
                                                                        binLOWVEG,binHIGHVEG,mVEGTYPE,mTMK2m,mDEW2m,mSkinT)                                 

                                                            add_ENV_csv(envfile,jdate,granule[ig],stid,ERAtime[clditime],int(stmtime[stid]),mLON,mLAT,Meanflag,flagBin,
                                                                        mCAPE,mCIN,mLCL,mSFLX,mLFLX,mMSLP,mSST,momega,mlowVWS, mmidVWS,mupVWS, mttlVWS,
                                                                        mdirlowVWS, mdirmidVWS, mdirupVWS, mdirttlVWS,mlowSH, mmidSH, mupSH, mttlSH,
                                                                        mlowRH, mmidRH, mupRH, mttlRH,mIWV,mIWVRatio,mAOD,COmeanIWP,COmeanSWP,COmeanSWPC,
                                                                        COmeanAMSRR,PedmeanAMSRR,PedmaxAMSRR,COmeanCldRR,PedmeanCldRR,PedmaxCldRR)

                                                            add_CUTOFF_csv(txtfile,jdate,granule[ig],stid,ERAtime[clditime],int(stmtime[stid]),mLON,mLAT,Meanflag,flagBin,
                                                                        COcutoff,np.round(np.array(freeze[stid]*1000,dtype=float),2),COAUXfreeze,COmincldbase,COmaxcldtop,COcldbase,
                                                                        COcldtop,COpedw,COanvilw,COAUXCTT,COAUXminCTT,cttmeanIR,cttminIR,COmeanCTT,COminCTT,PedmeanDEM,
                                                                        PedmaxDEM,COmeanDEMhght,COmaxDEMhght,Pedmeantopo,Pedmaxtopo,nPed,Ped_wid,nCore,Core_wid)

                                                            add_SOUNDING_csv(soundingfile,jdate,granule[ig],stid,ERAtime[clditime],int(stmtime[stid]),mLON,mLAT,Meanflag,flagBin,
                                                                            Tmk_arr,Td_arr,SH_arr,U_arr,V_arr) 