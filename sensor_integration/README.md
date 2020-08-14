# Getting sensors measurements
In this section we want to integrate real sensors measurements to the nengo model. We use the STERIOLABS ZED camera which conatains an IMU unit (which has an accelerometer and a gyroscope sensors) to do this task.

# Installation and dependencies
## Prerequisites
* Install python 3.7 (stable version for Nengo)
* Download and install cuda 11 (or latest) on https://developer.nvidia.com/cuda-downloads
* Download and install the latest version of the ZED SDK on https://www.stereolabs.com/developers/release/
NOTE that the cuda version provided in the ZED SDK installation is not sufficient, so you have to manually install cuda.

Test your ZED camera following the steps provided on https://www.stereolabs.com/docs/installation/

## ZED - Python API
Follow the instructions provided on https://github.com/stereolabs/zed-python-api

### Troubleshooting
- ZED python module installer is looking for not found ```version.txt``` file:

  To solve this problem you can create a new version.txt file in the ZED SDK installation folder and copy the output of the command ```/usr/local/cuda/bin/nvcc --version``` to the new file. 
  NOTE that you have to use admin mode to create this file.

Test the python API running the toutorials on https://github.com/stereolabs/zed-examples/tree/master/tutorials
