# EECS 4422 Project - Brandon Loy
Naive implementation of a video tracker based on Channel and Spatial Reliability - Discriminant Correlation Filter Tracker (CSR-DCF) [1]. A region containing a target object is manually selected. Histogram of Gradients is used to extract features from video frames. Correlation filters are used to to localize the target object in subsequent frames. 
 
### Getting Started
Install the required python libraries
```
pip install -r requirements.txt
```

### Usage
The video should be broken into frames, saved in a directory. This directory should be in the same directory as track.py. From an ipython terminal, run the following:
```
from track import csrTrack
```
Once the tracker is imported, the directory name should be passed as the first argument. 
```
csrTrack('yourDirectory')
```
Optionally, you can view the feature channel responses by setting the 2nd argument to True
```
csrTrack('yourDirectory', True)
```

### Preparing Video Files
The vid2jpg function can be used to turn your video file into a directory of jpgs.
```
from vid2jpg import vid2jpg
```
Pass the path to the video file in argument 1. Argument 2 is the downsampling factor, this is optional and is by default set to 1. In the following example, the frames will be shrunk by a factor of 3.
```
vid2jpg(<pathString>, 3)
```
It will create a new directory, with the video filename, in the current working directory. This directory can now be used with csrTrack



[1] Alan Lukežič, Tomáš Vojíř, Luka Čehovin, Jiří Matas and Matej Kristan. ''Discriminative Correlation Filter with Channel and Spatial Reliability.'' In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
