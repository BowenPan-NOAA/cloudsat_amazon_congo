import gzip
import numpy as np

def gunzip(source_filepath, dest_filepath, block_size=65536):
#from https://stackoverflow.com/questions/52332897/how-to-extract-a-gz-file-in-python
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)

def slide_max(array,window):
    #Establish Five-point max elevation array
    #Window much be even...
    #Array - to do slide maximum
    #Window- the window width to the selected maximum
    copy = np.copy(array)
    gap_length = int(np.floor(window/2))   #gap at the boundary
    Complete_length = int(len(array)-np.floor(window/2)*2) #Remove the windows on both sides
    for i in range(Complete_length):
        idx = gap_length+i
        copy[idx] = np.nanmax(array[idx-gap_length:idx+gap_length+1])
    return(copy)

def cal_start_end_julian_days(month,yr):
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