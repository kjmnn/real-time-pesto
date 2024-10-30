# Real-time pitch estimation with PESTO
A little script I made to run [PESTO](https://github.com/SonyCSLParis/pesto) in real time and display the output as an animated graph. 

Dependencies are in `requirements.txt`; it should probably run on GPU as well, but I only tested it on CPU on my Linux machine.

For usage run `python real_time_pesto.py -h` or check the source. To exit, just close the plot window.

For better performance, decrease the sampling rate (decreases accuracy), increase step size (decreases temporal resolution) or increase steps per chunk (decreases responsiveness).  
For better prediction accuracy, increase the sampling rate (decreases performance), increase step size (decreases temporal resolution) or increase steps per chunk (decreases responsiveness).  
For best accuracy, just record yourself and use the `pesto` module's CLI instead of this 😎