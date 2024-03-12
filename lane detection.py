# Import the Libraries
import cv2
import numpy as np

# Storing a video in a variable
video = cv2.VideoCapture("C:/Users/nisar/Desktop/Projects/2.mp4")

# Function to find the region of intrest(ROI)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to get the points to plot after we find the best fitting line by polyfit
def make_points(line_values):
    slope,intercept = line_values
    height = 600
    y1 = int(height)
    y2 = int((y1*3.0)/5)
    x1 = int(((y1-intercept)/slope))
    x2 = int((y2-intercept)/slope)
    
    return[x1,y1,x2,y2]

# Function to find the Best line using polyfit function    
def average_slope_intercepts(lines1):
    Lines = []

    for line in lines1:
        for x1,y1,x2,y2 in line:
            fit1 = np.polyfit((x1,x2),(y1,y2),1) 
        
            
    Lines.append((fit1[0],fit1[1]))
    Linesaverage = np.average(Lines,axis = 0)
    curve = make_points(Linesaverage)
    return curve

while video.isOpened():
    
# Looping on the video frame bt frame till no frames are left    
    is_sucess, orig_frame = video.read()
    if is_sucess is False:
        break
    
    height = orig_frame.shape[0]
    width = orig_frame.shape[1]

    region_of_interest_vertices = [(0, height),(width/2, height/2),(width, height)]

# Cropping the frame based on the region of intrest    
    frame = region_of_interest(orig_frame,
                np.array([region_of_interest_vertices], np.int32),) 
    
# Converting the cropped frame to HSV color space    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
# Filtering the white channel    
    sensitivity = 30
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])


    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    edges = cv2.Canny(mask, 50, 150) # Finding edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50 ,minLineLength =50,maxLineGap=10) # Finding the lines based on the edges
    lines = [average_slope_intercepts(lines)] # Calling the function to best fit the line

# Ploting the line on the orginal frame    
    if lines is not None:
     for line in lines:
          x1, y1, x2, y2 = line
          if (x1-y1) >50 and (x2-y2) >50 :
              cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
         
         
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    
    edges2 = cv2.Canny(mask2, 50, 150) # Finding the lines based on the edges
    lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 10 ,minLineLength =50,maxLineGap=2000) # Calling the function to best fit the line
    
# Plotting the dased lines on orginal frame
    if lines2 is not None:
     for line in lines2:
          x1, y1, x2, y2 = line[0]
          if (x1-y1) <50 and (x2-y2) < 50 :
              cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

# Displaying the orginal frame with the Lanes on it represented by green and red colors for solid and dashed lanes respectivily 
    cv2.imshow("frame2", orig_frame)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()