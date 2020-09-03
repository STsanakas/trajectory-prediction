import sys
import math 
import pandas as pd
import numpy as np
def calculate_initial_compass_bearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def convert_to_distance_bearing(pointAlat, pointAlon, pointBlat, pointBlon):
	pointA=(pointAlat,pointAlon)
	pointB=(pointBlat,pointBlon)
	A=calculate_initial_compass_bearing(pointA, pointB)
	from math import sin, cos, sqrt, atan2, radians
	# approximate radius of earth in km
	R = 6373.0
	lat1 = radians(pointAlat)
	lon1 = radians(pointAlon)
	lat2 = radians(pointBlat)
	lon2 = radians(pointBlon)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	distance = R * c
	return distance, A

def transform_dataset(datasetfile):
	dataset_coords = pd.read_csv(datasetfile, engine='python').values.astype('float32')
	dataset=[0,0]
	for i in range(len(dataset_coords)-1):
		j=dataset_coords[i]
		k=dataset_coords[i+1]
		dis, bear = convert_to_distance_bearing(j[0], j[1], k[0], k[1])		
		results=[dis, bear]
		dataset=np.vstack((dataset,results))
	dataset=np.delete(dataset, 0, 0)
	return 	dataset
