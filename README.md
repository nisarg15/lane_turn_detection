       
### To run this code following libraries are required
* OpenCV  
* NumPy
* os
* glob
* matplotlib.pyplot


### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````
* NumPy
	````
	pip install numpy
	````
### Running code in ubuntu
After changing the path of the video source file, images folder, path for the 
write images and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command

````
lane detection.py
````
* Run the following command which will provide the output as
  series of images which will show lane detection on the orginal frame.
  
````
Turn detection.py
````
It is important to note that if both python files are in different directory
we have to change to the correct directory again.


### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the video and images
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
  
### Maintainers ###
	Nisarg Upadhyay (nisargupadhyay1@gmail.com)
