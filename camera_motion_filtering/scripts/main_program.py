# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------
# DIGITAL SIGNAL PROCESSING
# CAMERA MOTION ESTIMATION BY FEATURE MASKING AND KALMAN FILTER FOR VIDEO STABILIZATION
# Author: Jay Neil C. Gapuz
# Notes:
# 		Code written in Python Version 2.7 (for v3.6 optimize)
#		Libraries include: 
#			OpenCV, scipy, matplotlib.pyplot (available free online)
#			stabilize _functions (author predefined functions)
#           filter_procedure (author defined functions for filtering)
# 			stabilze_movement (author defined functions for motion compensation)
#-----------------------------------------------------------------------------------------
from os import system
system('cls')
x = ('''
		 ______________________________________________
		|                                              |
		|                   ─▄▀─▄▀                     |
		|                   ──▀──▀                     |
		|                   █▀▀▀▀▀█▄                   |
		|                   █░░░░░█─█                  |
		|                   ▀▄▄▄▄▄▀▀                   |
		|                                              |
		|______________________________________________|
		CAMERA MOTION ESTIMATION BY FEATURE MASKING AND 
		     KALMAN FILTER FOR VIDEO STABILIZATION
		-----------------------------------------------
                    
	''')
print(x.decode('utf-8'))
import sys
sys.stdout.write('\r                               IMPORTING LIBRARIES')
sys.stdout.flush()
import cv2
import scipy.signal
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
sys.stdout.write('\r                             IMPORTING CUSTOM FUNCTIONS')
sys.stdout.flush()
from stabilize_functions import *
from filter_procedure import *
from stabilize_movement import *


sys.stdout.write('\r                                                            \n')
sys.stdout.flush()

#-----------------------------------------------------------------------------------------
# CONVERTING VIDEO TO FRAMES
#-----------------------------------------------------------------------------------------

filename = raw_input('INPUT VIDEO FILENAME: ')
FRAMES_ARR,FPS = CONVERT_VIDEO_TO_FRAMES(filename)
VIDLENGTH = len(FRAMES_ARR)/float(FPS) 
print('FPS             = %d' %FPS)
print('VIDEO LENGTH    = %0.1f s' %VIDLENGTH)

# GET THE DIMENSION OF IMAGE
IMAGE = cv2.imread(FRAMES_ARR[0],0) #Open image in grayscale
height,width = IMAGE.shape
print('FRAME DIMENSION = %d x %d\n' %(height,width))


#-----------------------------------------------------------------------------------------
# FEATURES TO TRACK FROM THRESHOLD IMAGE
#-----------------------------------------------------------------------------------------
'''
	Use Improved Harris Filter to track good features 
	from the processed image using the Sobel Operator
'''
def track_features(IMAGE):

	img_mask = np.zeros(IMAGE.shape[:2], dtype="uint8")

	corners = cv2.goodFeaturesToTrack(IMAGE,250,0.1,20)
	corners = np.int0(corners)

	
	for corner in corners:
		x,y = corner.ravel()
		#plt.scatter(x,y,marker ='o',color='black')
		cv2.circle(img_mask,(x,y),3,255,-1)
	#plt.imshow(img)
	#SHOW_IMAGE(mask)	
	
	return img_mask

#-----------------------------------------------------------------------------------------
# PERFORM COMPARISON OF FRAMES
#-----------------------------------------------------------------------------------------
'''
	Code has been optimized so that it can compare various
	non adjacent frames, by changing the FRAME_DISTANCE
	parameter
'''
#Input parameters	
count = -1
FRAMES_ARR_LEN = len(FRAMES_ARR)
OVERLAP = 2  #default = 2, compare 2 frames at a time
FRAME_DISTANCE = 1  #compare every n frame


SIGNIFICANT_FRAMES =[0]
cnt = 0
while cnt < len(FRAMES_ARR):
	SIGNIFICANT_FRAMES.append(cnt)
	cnt = cnt + FRAME_DISTANCE


# For logging and debugging 	
#temp = open('TEMP_POINTS.csv','w')

MOVEMENT_ARR_X = []
MOVEMENT_ARR_Y = []

MOTION_ARR_X = []
MOTION_ARR_Y = []

for frame_num in SIGNIFICANT_FRAMES:

	count = count + 1
	init = True
	X_AXIS_ARR = np.zeros(OVERLAP).tolist()
	Y_AXIS_ARR = np.zeros(OVERLAP).tolist()
	
	
	for lap in (0,FRAME_DISTANCE):
		try:
			frame = FRAMES_ARR[frame_num + lap]
		except:
			break
			
		
		IMAGE = cv2.imread(frame,0) #Open image in grayscale
		IMAGE = SOBEL_FILTER(NORMALIZE_IMAGE(IMAGE))
		features_mask = track_features(IMAGE)

		IMAGE_1 = features_mask

		if init:
			#x,y = convolve_image(IMAGE_1,IMAGE_1) 
			# init = False
			# IMAGE_2 = IMAGE_1
			# X_AXIS_ARR[0] = y[1]
			# Y_AXIS_ARR[0] = y[0]
			
			#Convolution result for the same frame located at the center
			IMAGE_2 = IMAGE_1
			init = False
			X_AXIS_ARR[0] = width/2
			Y_AXIS_ARR[0] = height/2
			
			
		else:
			x,y = convolve_image(IMAGE_1,IMAGE_2)
			X_AXIS_ARR[1] = (y[1]-X_AXIS_ARR[0])
			Y_AXIS_ARR[1] = (y[0]-Y_AXIS_ARR[0])


	X_AXIS_AVE = X_AXIS_ARR[1]
	Y_AXIS_AVE = Y_AXIS_ARR[1]
	
	if frame_num == 0:
		X_PREV = 0 
		Y_PREV = 0 
		X_1 = X_AXIS_AVE
		Y_1 = X_AXIS_AVE
		continue
	elif frame_num == 1:
		X_PREV = X_AXIS_AVE + X_1
		Y_PREV = Y_AXIS_AVE + Y_1
	else:
		X_PREV = X_AXIS_AVE + X_PREV
		Y_PREV = Y_AXIS_AVE + Y_PREV
		
	# For debugging purpose	
	#print X_AXIS_AVE,Y_AXIS_AVE,X_PREV,Y_PREV
	#temp.write('%s,%s,%s,%s\n' %(X_AXIS_AVE,Y_AXIS_AVE,X_PREV,Y_PREV))
	
	#Store 
	MOTION_ARR_X.append(X_AXIS_AVE)
	MOTION_ARR_Y.append(Y_AXIS_AVE)
	MOVEMENT_ARR_X.append(X_PREV)
	MOVEMENT_ARR_Y.append(Y_PREV)
	
	
	sys.stdout.write('\r  STATUS: FRAME %s OUT OF %s' %(frame_num,FRAMES_ARR_LEN))
	sys.stdout.flush()

sys.stdout.write('\r   ESTIMATING CAMERA MOTION........')
sys.stdout.flush()
#temp.close()

#-----------------------------------------------------------------------------------------
# FILTERING 
#-----------------------------------------------------------------------------------------


# plt.plot(MOVEMENT_ARR_X,'--')
# plt.plot(MOVEMENT_ARR_Y,'--')

MOVEMENT_ARR_X_NEW,MOVEMENT_ARR_Y_NEW = KLM_FILTER_(MOVEMENT_ARR_X,MOVEMENT_ARR_Y,ITERATION = 5)
# plt.plot(MOVEMENT_ARR_X_NEW,'--')
# plt.plot(MOVEMENT_ARR_Y_NEW,'--')

MOVEMENT_ARR_X_NEW,MOVEMENT_ARR_Y_NEW = SMOOTHING_FILTER(MOVEMENT_ARR_X_NEW,MOVEMENT_ARR_Y_NEW,ORDER = 36)

# plt.plot(MOVEMENT_ARR_X_NEW)
# plt.plot(MOVEMENT_ARR_Y_NEW)
# plt.xlabel('frame')
# plt.ylabel('pixel shift')
# plt.show()


sys.stdout.write('\r  APPLYING STABILIZATION..........')
sys.stdout.flush()


#-----------------------------------------------------------------------------------------
# MOTION COMPENSATION AND FRAME TRANSFORMATION
#-----------------------------------------------------------------------------------------
FRAMES_ARR = AFFINE_TRANSFORM(MOVEMENT_ARR_X,MOVEMENT_ARR_Y,MOVEMENT_ARR_X_NEW,MOVEMENT_ARR_Y_NEW)

sys.stdout.write('\r  SAVING VIDEO..........')
sys.stdout.flush()
filename = ('%s_stabilized.avi' %(filename.split('.')[0]))
CONVERT_FRAMES_TO_VIDEO(FRAMES_ARR,FPS,filename)

sys.stdout.write('\r  FINISHED                                                                            ')
sys.stdout.flush()
