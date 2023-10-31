#!/bin/bash

#/home/shubham/MATLAB/bin/matlab -nosplash -nodisplay -r "run Time_Optimal_PathPlanning_Launch.m;quit;"
# Have to be in the path of the files
# Make an alias matlab="/home/shubham/MATLAB/bin/matlab"

# 1. Buzzbay edits:
    # a. istr = 240; iend = 420; jstr = 80; jend = 200;
    # b. Mask_p(41:60, 81:100) = 1; # Figure out many masks
    # c. save name -> '_B08_VS06'; # iterate through something to change each time
cd /home/shubham/Documents/MATLAB/TOPP/trunk/SandBox/Cases/SeaExercise
exec /bin/bash
matlab="/home/shubham/MATLAB/bin/matlab"
    # matlab -nosplash -nodisplay -r "buzzBay_extract_interp(240, 420, 80, 200, 81, 100, 41, 60, '_B08_VS09')"
matlab -nodesktop -r "buzzBay_extract_interp(240, 420, 80, 200, 81, 100, 41, 60, '_B08_VS09')"
 