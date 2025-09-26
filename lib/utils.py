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