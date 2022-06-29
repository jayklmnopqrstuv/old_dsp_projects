# Background Noise Reduction Based on Wiener Suppression using a Priori Signal-to-Noise Ratio Estimate

*Abstract* - Random, additive noise is a form of degradation a major problem of all analog communication system. In an
audio file, listeners can hear it as “hiss” sounds that usually came from different sources. These noises came from inside the
device itself and ambient noise coming from the environment which is not correlated to the signal itself. This study focuses
on the suppression of background noise in a mono-channel audio file using spectral analysis. Initially, the input audio file
was segmented. Then, each segment was analyzed using Fast Fourier Transform and approximation of its SNR was
calculated. The signal will then be attenuated by multiplying a suppression value calculated using the Wiener suppression
formula in the frequency domain. The original signal was then restored using Inverse Fourier Transform and pass through a
moving-average for smoothing. Data were gathered from actual noisy audio files which were the input to the system. The
researchers compared the output audio signal to the input audio signal by evaluating the signal to noise ratio of the two.

#### NOTES:
Code to run : `main.py` \
This code is optimized for Python 2.7  <br> 
Libraries include:   <br>
<ol>
 <li> matplotlib.pyplot  <br>
	- MATLAB function for plotting  <br>
 <li> scipy  <br>
	Scilab-like library includes fft functions   <br>
	and signal filtering tools  <br>
	Specific functions for import:  <br>
	- *scipy.io  <br>
	- *scipy.fft  <br>
	- *scipy.signal  <br>
 <li> numpy  <br>
	Handling arrays and faster processing  <br>
 <li> math   <br>
   functions for computing  <br>
 <li> soundfile  <br>
	read and write wav files  <br>
</ol>
