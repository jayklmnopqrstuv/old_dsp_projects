#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PROJECT PYTHON CODE
#  Background Noise Reduction Based on Wiener Suppression
#  Using A Priori Signal-to-Noise Ratio Estimate in Python
#This code is optimized for Python 2.7, see necessary libraries
#[CODING TIMELINE] Start: 12.05.2017 End: 12.12.2017
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import struct              
import soundfile as sf
from scipy.signal import butter, lfilter,filtfilt,freqs,freqz
from scipy.signal import firwin, kaiser_atten, kaiser_beta
import math
import sys
import os


'''In the command window, type main.py <filename> <type>
**filename should be a wav filetype saved in the folder
where this script is saved, otherwise indicate the directory
**type is a selection between narrowband voice signal 
and wideband voice signal, default is narrowband'''

filename = (sys.argv[1]).split('.wav')[0]
try:
	type = sys.argv[2]
except:
	type = 'narrowband'

if type in ['narrowband','wideband']:
	pass
else:
	print 'UNKNOWN TYPE, will use NARROWBAND'
	type = 'narrowband'
print 'FILENAME: ', filename
print 'TYPE: ', type , '\n'



####################################################################################
#LOAD SIGNAL
####################################################################################
#Load signal and find its sampling frequency
#[Fs,SIG_ORIG] = wavfile.read('flight_587.wav')
print 'LOADING FILE......'
[SIG_ORIG,Fs] = sf.read('%s.wav' %filename)
SIG_ORIG = SIG_ORIG.tolist()
SIG_LEN  = len(SIG_ORIG)


'''this code isolates the different type of wav files
and save the resulting to a single array'''
SIG_ORIG_TEMP=[] 
try:
	for i in SIG_ORIG:
		SIG_ORIG_TEMP.append(i[0])
	SIG_ORIG=SIG_ORIG_TEMP

except:
	pass


####################################################################################
#SEGMENTATION PROCESS
####################################################################################
print 'SEGMENTATION PROCESS.....'
#---------------------------------------------------------------
#(A) ZERO PADDING
#---------------------------------------------------------------
print '     ZERO PADDING.....'
'''Purpose: Make all short time wav samples equal to a certain
duration (Transform to a 15sec sound)'''

#functions to return different params
def ZERO_ARR(k):return [0.0 for x in range(k)]
def ZERO_PAD_END(INPUT_SIG,x):	return  ZERO_ARR(x) + INPUT_SIG
def ZERO_PAD_START(INPUT_SIG,x): return INPUT_SIG + ZERO_ARR(x)


#```````````````````````````````````````````````````````````````
'''Add initial padding at the start of the signal
   Initial samples will not have any points for comparison
   after the segmentation and filtering process
   For visualization:
   first chunk   [1 2 3 4 5 6 7 8 9 10 ] 
   second chunk  [          6 7 8 9 10 11 12 13 14 15 ]
   third chunk   [                     11 12 13 14 15 16...]
   Notice how the first five samples dont have any match
   Hence, let's pad zeros at the start of the signal
   equivalent to half of the length of the chunk
   first chunk   [0 0 0 0 0 1 2 3 4 5 ]   
   second chunk  [          1 2 3 4 5 6 7 8 9 10 ]
   third chunk   [                    6 7 8 9 10 ...]  '''
#``````````````````````````````````````````````````````````````

'''this function computes the number of length of zeros
to pad at the end of the signal''' 
def ZERO_END_LEN(Fs,length,t=15.0):
	ZERO_LEN = int(t*Fs)
	ZERO_LEN = ZERO_LEN - length - 45
	return ZERO_LEN


CHUNK_LEN = int(0.020 * Fs) #/no.of samples per segment of 20ms/

SIG_NEW = ZERO_PAD_END(SIG_ORIG, ZERO_END_LEN(Fs, SIG_LEN))
SIG_NEW = ZERO_PAD_START(SIG_NEW,int(CHUNK_LEN/2)) #/ + 10ms/

'''bug found, need to compensate the START PADDING by 
adding the same zeros at the end hence total time = 15s + 20ms'''
SIG_NEW = ZERO_PAD_END(SIG_NEW,int(CHUNK_LEN/2))

#plot the resulting time domain of the signal then superimpose it
plt.subplot(3,1,1)
plt.plot(SIG_NEW,'b')
plt.title('ORIGINAL SIGNAL')


#---------------------------------------------------------------
#FIR FILTERS
#(*)BANDPASS FILTERING 
# for narrowband voice signal use ~200 - 3.4KHz
# fow wideband voice signal use 50 - 7~8KHz
#(*)SMOOTHING FILTER (moving average in the time domain)
#---------------------------------------------------------------


def BANDPASS_KAISER(ntaps,data, lowcut, highcut, fs, width):
	
	nyq = 0.5 * fs
	atten = kaiser_atten(ntaps, width / nyq)
	beta = kaiser_beta(atten)
	order = firwin(ntaps, [lowcut, highcut], nyq=nyq,
                  pass_zero=False,
                  window=('kaiser', beta), scale=False)
	y=lfilter(order,1,data)

	#/*for plotting purposes of the filter used*/
	#w, h = freqz(order, 1)
	#plt.plot((fs * 0.5 / np.pi) * w,
    #        abs(h), label="Kaiser window")	
	#plt.show()

	return y

def MOVING_AVERAGE(SIGNAL,WIDTH):
	SMOOTH_SIGNAL =[]
	for i in range(len(SIGNAL)):
		try:
			AVE=sum(SIGNAL[i:i + WIDTH-1])/WIDTH
			SMOOTH_SIGNAL.append(AVE)
		except:
			SMOOTH_SIGNAL.append(SIGNAL[i])

	return SMOOTH_SIGNAL

'''for debugging'''	
#SIG_NEW = BANDPASS_KAISER(200,SIG_NEW,300,3000,Fs,100);
#SIG_LEN= len(SIG_NEW)
#sf.write('%s_bandpass.wav' %filename,SIG_NEW, Fs)
#plt.plot(SIG_NEW,'g')

#---------------------------------------------------------------
#(B) DIVIDING INTO CHUNKS
#---------------------------------------------------------------
print '     DIVIDE INTO SEGMENTS.....'
'''Return a list of chunks stored in separate arrays'''

CHUNK_ARR=[]
COUNTER = 0

while COUNTER <= len(SIG_NEW):
	CHUNKY = (SIG_NEW[COUNTER: (COUNTER + CHUNK_LEN)])
	if len(CHUNKY) == CHUNK_LEN/2:
		CHUNK_ARR.append(CHUNKY + ZERO_ARR(CHUNK_LEN/2))
		#print len([CHUNKY] + ZERO_ARR(CHUNK_LEN/2))
	if len(CHUNKY)== 0: #/*solution to bug*/
		pass
	else:
		CHUNK_ARR.append(CHUNKY)
	COUNTER += CHUNK_LEN/2

#check the number of chunks stored:
CHUNKS_TOTAL = len(CHUNK_ARR)


'''how elements of the chunk will be synthesized later
CHUNK 1 1 2 3 4 5 6 7 8 9 10
CHUNK 2         1 2 3 4 5 6 7 8 9 10       '''


####################################################################################
#SNR COMPUTATION AND ATTENUATION
####################################################################################

'''functions to calculate different params'''
def CALCULATE_FFT(SIGNAL):
	FFT = np.fft.fft(SIGNAL)
	FFT_ABS = (np.real(FFT)).tolist()
	FFT_IM  = (np.imag(FFT)).tolist()
	return FFT,FFT_ABS,FFT_IM


def PLOT_SIGNAL(TIME_SIGNAL,FFT_SIGNAL,SIZE):
	plt.subplot(2,1,1)
	plt.plot(TIME_SIGNAL)
	plt.xlim(0,SIZE)
	plt.subplot(2,1,2)
	plt.plot(abs(FFT_SIGNAL))
	#plt.xlim(0,SIZE/2)
	#plt.xlim(0,2)	
	plt.show()

def FFT_BIN_ENERGY(FFT_REAL,FFT_IMAG,LENGTH):
	ENERGY_ARR=[]
        for i in range(LENGTH):
                ENERGY_ARR.append((FFT_REAL[i]**2)+(FFT_IM[i]**2))
        return ENERGY_ARR


#---------------------------------------------------------------
#SNR COMPUTATION FUNCTIONS
#---------------------------------------------------------------

table = open('%s_table.csv' %filename,'wb')
table.write('CHUNK,SIGNAL ENERGY, NOISE ENERGY, SNR, ATTENUATION\n')

def MEAN_SD(ENERGY_ARR):
        cal = sum(ENERGY_ARR)/(len(ENERGY_ARR))
        return (cal + (0.3)(cal))
    
        
#https://labrosa.ee.columbia.edu/~dpwe/tmp/nist/doc/stnr.txt    
def COMPUTE_SNR_ESTIMATE(ENERGY_ARR,LENGTH):
    
    #	SIGNAL_ENERGY = sum of the maximum energy values
	#	NOISE_ENERGY   = noise floor =  minimum * len(fft bins)
	#	SNR = (TOTAL_SIGNAL - NOISE)/NOISE'''

	ENERGY_NEW_ARR=[]
	for ENERGY in ENERGY_ARR[0:LENGTH-1]:
		if ENERGY < 0.5:  #remove the zero pad values
			pass
		else:
				ENERGY_NEW_ARR.append(ENERGY)		
	#Rough estimation
	N = min(ENERGY_NEW_ARR) * (len(ENERGY_NEW_ARR))
	S = sum(ENERGY_ARR[0:LENGTH-1])
                           
	try:
			return abs((S - N)/N)
                           
	except: #bug due to zero padding since 0/0 = undet
		return 0

                                                 

def COMPUTE_SNR_RAYLEIGH(ENERGY_ARR,LENGTH):
	global counter
	global table
	SIGNAL_ARR = []
	'''Method 1: Top 5 FFT bins with the highest energy are the signal components'''
	#SIGNAL_ARR = (sorted(ENERGY_ARR[0: LENGTH-1],reverse=True)[0:4])
	'''Method 2: If component is less than the  mean +  sigma of the energy arr,
	   treat as part of the noise floor'''
	#get the sigma of the energy arr then add to the mean:
	LIMIT = np.mean(ENERGY_ARR[0:LENGTH-1]) + np.std(ENERGY_ARR[0:LENGTH-1])
	
        NOISE_ARR =[]
            
	for ENERGY in ENERGY_ARR[0:LENGTH-1]:
		#if ENERGY in SIGNAL_ARR:  #Method 1
		if ENERGY > LIMIT:
			#pass #Method 1
			SIGNAL_ARR.append(ENERGY) #Method 2
		else:
			NOISE_ARR.append(ENERGY)
			
	'''for plotting purposes'''		
	#plt.bar([x for x in range(len(ENERGY_ARR))],10 * np.log(ENERGY_ARR))
	#plt.plot([x for x in range(len(ENERGY_ARR))],[10 * np.log(LIMIT) for y in range(len(ENERGY_ARR))])
	#if LIMIT > 0:
	#	plt.xlim(0,LENGTH-1)
	#	plt.ylim(0,120)
	#	plt.title('SPECTRUM DISTRIBUTION')
	#	plt.ylabel('ENERGY')
	#	plt.xlabel('FFT BIN')
	#else:
	#	plt.close()

	try:
		N = sum(NOISE_ARR) /LENGTH
		S=  sum(SIGNAL_ARR)/LENGTH
		#print S,N
		table.write('%s,%s,'%(S,N))
		SNR_VAL = abs(S/N)
		if S < 1.0:   #silence discriminator 
			return 0.0 , SIGNAL_ARR
		else:
			return abs(S /N), SIGNAL_ARR
	
	except: #bug due to zero padding since 0/0 = undet
		return 0, SIGNAL_ARR

	

#---------------------------------------------------------------
#SPECTRAL ATTENUATION FUNCTION
#---------------------------------------------------------------

def REC_TO_POLAR(x,y):
        r = np.sqrt(x**2 + y**2)
        p = math.degrees(np.arctan2(y,x))
        return r,p
    
def POLAR_TO_REC(r,p):
        x = r * math.cos(p)
        y = r * math.sin(p)
        return x,y

def GAIN_ATTENUATE(FFT_VAL, GAIN):
        mag,ang = REC_TO_POLAR(np.real(FFT_VAL),np.imag(FFT_VAL))
        mag = mag * GAIN
        x,y = POLAR_TO_REC(mag,math.radians(ang))
        return np.complex(x,y)

	
'''https://www.asee.org/documents/zones/zone1/2008/student/ASEE12008_0044_paper.pdf	
A reasonable generalization is that if the zero-crossing rate is high,
the speech signal is unvoiced, while if the zero-crossing rate is low, the speech signal is voiced
Note: Not implemented in this program, yet'''

def SILENCE_DISCRIMINATOR(CHUNK, ENERGY_ARR):
	zero_crossing = (np.diff(np.sign(CHUNK)) != 0).sum() - (CHUNK == 0).sum()
	zero_rate = len(zero_crossing)
	total_energy = sum(ENERGY_ARR[0:len(ENERGY_ARR)/2])
	
	'''need to decide the zero crossing rate and E''' 
	
#---------------------------------------------------------------
#SPECTRAL ATTENUATION FUNCTION
#---------------------------------------------------------------
print 'PERFORMING SNR AND SPECTRAL ATTENUATION......'
counter = 0
FINAL_CHUNK_ARR=[]
SNR_ARRAY=[]
for CHUNK in CHUNK_ARR:
	try:
		counter = counter + 1
		table.write('%s,' %counter)
		POST_CHUNK=[]
		SPECTRUM_ARR = []
		FFT,FFT_REAL,FFT_IM = CALCULATE_FFT(CHUNK)
		FFT_BIN_ENERGY_ARR= FFT_BIN_ENERGY(FFT_REAL,FFT_IM,CHUNK_LEN)
        #val = MEAN_SD(ENERGY_ARR)
		try:
			'''if Method 1 (SNR ROUGH ESTIMATION) is used'''
            #SNR = COMPUTE_SNR_ESTIMATE(FFT_BIN_ENERGY_ARR,CHUNK_LEN/2)
			'''if Method 2 (SNR USING RAYLEIGH's) is used'''
			SNR,REL_BINS = COMPUTE_SNR_RAYLEIGH(FFT_BIN_ENERGY_ARR,CHUNK_LEN/2)
			SNR_ARRAY.append(SNR)
			ATTENUATION = (SNR - 1)/SNR
			#ATTENUATION = (1.0 + math.sqrt((SNR - 1)/SNR))*0.5 #Method 2
            #print SNR, SNR2,ATTENUATION,ATTENUATION2
						
		except:
            #debugging (SNR < 1)
			ATTENUATION = 0
		table.write('%s,%s\n' %(SNR, ATTENUATION))
		for sig in FFT:
			'''if Method 1 (SNR ROUGH ESTIMATION) is used'''
			#POST_CHUNK.append(GAIN_ATTENUATE(sig,ATTENUATION))
			'''if Method 2 (SNR USING RAYLEIGH's) is used'''
			if ATTENUATION == 0:
				POST_CHUNK.append(GAIN_ATTENUATE(sig,0.15))
				SPECTRUM_ARR.append((np.real(sig) * 0.15) ** 2 + (np.imag(sig) * 0.15) ** 2)
			else:
				if ((np.real(sig))**2 + (np.imag(sig))**2) in REL_BINS:
					POST_CHUNK.append(GAIN_ATTENUATE(sig,1))
					SPECTRUM_ARR.append((np.real(sig) * 1) ** 2 + (np.imag(sig) * 1) ** 2)
				else:
					POST_CHUNK.append(GAIN_ATTENUATE(sig,ATTENUATION))
					SPECTRUM_ARR.append((np.real(sig) * ATTENUATION) ** 2 + (np.imag(sig) * ATTENUATION) ** 2)
					
		#plt.bar([x for x in range(len(SPECTRUM_ARR))],10 * np.log(SPECTRUM_ARR),color ='red')
		#plt.savefig('%s.png' %counter)
		#plt.close()
		POST_CHUNK = np.real(np.fft.ifft(POST_CHUNK))
		FINAL_CHUNK_ARR.append(POST_CHUNK)
		
		
	except:
		pass
		
	counter = counter + 1

table.close()
		
		
#####################################################################################
#SYNTHESIS PROCESS
#####################################################################################
print 'PERFORMING SYNTHESIS......'
WAV_SYNTHESIS=[]
counter = 0
for j in range(len(FINAL_CHUNK_ARR)):
	try:
		CHUNK1 = FINAL_CHUNK_ARR[j]
		#CHUNK1 = MOVING_AVERAGE(CHUNK1,3)
		CHUNK2 = FINAL_CHUNK_ARR[j+1]
		#CHUNK2 = MOVING_AVERAGE(CHUNK2,3)
		
		C1=[]
		C2=[]
		#
		for pos in range(len(CHUNK1)):
			if pos < CHUNK_LEN/2-1:
				pass
			else:
				#print CHUNK2[pos-(CHUNK_LEN/2)],CHUNK1[pos]
				WAV_SYNTHESIS.append((CHUNK2[pos-(CHUNK_LEN/2)]+CHUNK1[pos])/2)
				C1.append(CHUNK2[pos-(CHUNK_LEN/2)])
				C2.append(CHUNK1[pos])
		#for plotting purposes
		#if counter > 500:
			#plt.plot(C1,'r')
			#plt.plot(C2,'b')
			#print counter
			#plt.show()
		counter = counter +1
	except:
		pass
		
plt.subplot(3,1,2)
plt.title('AFTER SPECTRAL ATTENUATION')
plt.plot(WAV_SYNTHESIS,'g')
#sf.write('%s_initial.wav' %filename, WAV_SYNTHESIS, Fs)
#final,x,y = CALCULATE_FFT(WAV_SYNTHESIS)
#PLOT_SIGNAL(WAV_SYNTHESIS,final,len(final)/4)


'''write output in a file, for raw data'''
f= open('%s_output_raw.txt'%filename,'wb')
for item in WAV_SYNTHESIS:
	f.write('%s,' %item)
	f.write('\n')
f.close()


#####################################################################################
#PERFORM SIGNAL CONDITIONING (Filter the signal for other noise)
#####################################################################################
print 'SIGNAL CONDITIONING.....'
if type == 'wideband':
	'''wideband signal'''
	WAV_SYNTHESIS = BANDPASS_KAISER(200,WAV_SYNTHESIS,50,Fs/2-50,Fs,50)
	WAV_SYNTHESIS = MOVING_AVERAGE(WAV_SYNTHESIS,3)
else: 
	'''narrowband signal'''
	WAV_SYNTHESIS = BANDPASS_KAISER(200,WAV_SYNTHESIS,280,3300,Fs,100)
	WAV_SYNTHESIS = MOVING_AVERAGE(WAV_SYNTHESIS,2)
plt.subplot(3,1,3)
plt.title('AFTER FILTERING/SMOOTHING')
plt.plot(WAV_SYNTHESIS,'r')
plt.ylim(-1.0,1.0)
plt.show()
#final,x,y = CALCULATE_FFT(WAV_SYNTHESIS)
#PLOT_SIGNAL(WAV_SYNTHESIS,final,len(final)/4)
sf.write('%s_filtered.wav' %filename, WAV_SYNTHESIS, Fs)



	
