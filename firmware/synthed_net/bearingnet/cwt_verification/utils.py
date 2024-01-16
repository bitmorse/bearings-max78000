import sys
import os
import numpy as np
#add wavelib to python path
wavelib_path = os.path.join(os.path.dirname(__file__), '../../../../../wavelib/python/')
print("Wavelib path: %s "% wavelib_path)

sys.path.append(wavelib_path)

from wavelibh import WavelibH
wavelib_as = WavelibH(wavelib_path+'libwavelibwrapper.so')

DECIMATION = 1 #by how much to decimate the signal of 20khz

INTERP_SIZE = 50
SAMPLE_LEN = 200

SCALES = 10 #cwt scales
MORLET = 4
DT = 1
DJ = 0.1*3 #spacing between scales
S0 = 2.5*DT #smallest scale
POW = 2 
REFERENCE = 0.05

'''
s0 is the smallest scale, 
while dj is the separation between scales. 
Dj can also be seen as a measure of resolution which is calculated as dj = 1.0 
Number of subscales so smaller value of dj corresponds to higher resolution within a scale.
type accepts “pow”/”power” or “lin”/”linear” as input values, power is the base of power
 if “pow”/”power' is selected and is ignored if the input is “lin”. Power of N scale calculation.
'''
def get_image(timeseries):
    assert len(timeseries) == SAMPLE_LEN
    linear_scale_type = 0
    
    cwt = wavelib_as.cwt(timeseries, S0, DJ, linear_scale_type, POW, MORLET, DT, SCALES)
    cwt = (cwt/REFERENCE)
    cwt = np.repeat(cwt, 5, axis=0)
    cwt = cwt[:,::4]
    
    assert cwt.shape == (INTERP_SIZE, INTERP_SIZE)
    
    return cwt


def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initial condition

    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t-1]

    return smoothed_data

