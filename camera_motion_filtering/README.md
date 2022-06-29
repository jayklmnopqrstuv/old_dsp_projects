# Camera Motion Estimation by Feature Masking and Kalman Filter for Video Stabilization

*Abstract*— Handheld devices used for capturing videos are susceptible to shaky motion during recording process. Various post processing techniques can correct and stabilize the unwanted movement the video had suffered. This paper introduces a simple and robust method to stabilize video by means of feature masking to compare frames and estimate the camera motion of the captured video. The calculated camera flow is smoothened using a Kalman filter, and the resulting motion provides compensation on the pixel shift relative to the obtained estimate. Effectiveness of this method was evaluated by comparing the result of simulated shaky video to the stabilized video. The system has provided significant reduction on the unwanted camera movement when applied on the actual sample video files.

#### Notes:

A. Scripts are written in Python 2.7.  These are not optimized yet for Python 3.x versions.  
B. Libraries include:  
* scipy
* numpy
* matplotlib.pyplot
* openCV
- Install these libraries from https://www.python.org/downloads/

C. Run main_program.py from the folder containing the other python scripts and the sample video.
