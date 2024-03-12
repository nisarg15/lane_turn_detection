# Import the Libraries
import cv2
import numpy as np

# Storing a video in a variable
video = cv2.VideoCapture("C:/Users/nisar/Desktop/Projects/3.mp4")

# Defining the variables
record_white = []
record_yellow = []
dest_orignal = []

# Function to find and plot the windows on the image
def curve_pts(img):


    window_image = np.dstack((img, img, img))*255

    histogram = np.sum(mask[mask.shape[0]//2:,:],axis = 0)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    yellow_low = np.argmax(histogram[:midpoint])
    white_low = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/10)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    n_z = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = yellow_low
    rightx_current = white_low
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_pts = []
    right_pts = []

    # Step through the windows one by one
    for window in range(9):
        # Identify window boundaries in x and y (and right and left)
        bottom = img.shape[0] - (window+1)*window_height
        top = img.shape[0] - window*window_height
        left_bottom = leftx_current - 100
        left_top = leftx_current + 100
        right_bottom = rightx_current - 100
        right_top = rightx_current + 100
        # Draw the windows on the visualization image

        cv2.rectangle(window_image,(left_bottom,bottom),(left_top,top),
        (0,255,0), 3) 
        cv2.rectangle(window_image,(right_bottom,bottom),(right_top,top),
            (0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= bottom) & (nonzeroy < top) & 
        (n_z >= left_bottom) &  (n_z < left_top)).nonzero()[0]
        good_right_inds = ((nonzeroy >= bottom) & (nonzeroy < top) & 
        (n_z >= right_bottom) &  (n_z < right_top)).nonzero()[0]
        # Append these indices to the lists
        left_pts.append(good_left_inds)
        right_pts.append(good_right_inds)
        
        if len(good_left_inds) > 1:
            leftx_current = np.int(np.mean(n_z[good_left_inds]))
        if len(good_right_inds) > 1:        
            rightx_current = np.int(np.mean(n_z[good_right_inds]))
        
# Points for curve fitting
    # Concatenate the arrays of indices
    left_pts = np.concatenate(left_pts)
    right_pts = np.concatenate(right_pts)

    # Extract left and right line pixel positions
    leftx = n_z[left_pts]
    lefty = nonzeroy[left_pts] 
    rightx = n_z[right_pts]
    righty = nonzeroy[right_pts] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    window_image[nonzeroy[left_pts], n_z[left_pts]] = [255, 0, 100]
    window_image[nonzeroy[right_pts], n_z[right_pts]] = [0, 100, 255]
   
    return window_image, left_fit, right_fit 
  
# Function to get the area of intrest and the 4 vertices to perfom homohraphy
def region_selection(image):

    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.90]
    top_left     = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.87, rows * 0.90]
    top_right    = [cols * 0.57, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image , vertices


while video.isOpened():

# Looping on the video frame bt frame till no frames are left       
    is_sucess, orig_frame = video.read()
    if is_sucess is False:
        break


    
    frame , vertices = region_selection(orig_frame) # Image with ROI
        
    src = []    
    for x in range(0,4) :
       point1 = list((vertices[0][x][0],vertices[0][x][1]))
       src.append(point1)
    src = np.float32(np.array(src)) # Source points for Wrapping the image


    height =vertices[0][1][1]  - vertices[0][0][1]

    width = vertices[0][2][0]  - vertices[0][1][0]

    
    dst = np.float32(np.array([[0,540], [0,0],[960,0],[960,540]])) # Destination points for Wrapping the image

    
    matrix = cv2.getPerspectiveTransform(src, dst) # Homography Matrix
    wrap_output = cv2.warpPerspective(frame, matrix,(960,540)) # Wrapping the image to get the bird's view
    
    intensity = np.ones(wrap_output.shape,dtype = "uint8")*70
    wrap_output = cv2.subtract(wrap_output,intensity) # Reducing the brightness of the wrapped image
    
# Converting the cropped frame to HSV color space     
    hsv = cv2.cvtColor(wrap_output, cv2.COLOR_BGR2HSV)
    
# Filtering the Yellow Channel         
    lower_yellow = np.array([18,94,140])
    upper_yellow = np.array([48,255,255])


    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow) # Mask of Yellow Channel
    
# Converting the cropped frame to HSV color space      
    hsv2 = cv2.cvtColor(wrap_output, cv2.COLOR_BGR2HSV)

# Filtering the white channel from the image    
    sensitivity = 100
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])


    mask1 = cv2.inRange(hsv2, lower_white, upper_white)# Mask of white channel


    mask = cv2.bitwise_or(mask1, mask2) # Combining both the masks
    
    window_image, left_value,right_value = curve_pts(mask) # Getting the image with boxes and the values of left and right points
    
    lspace = np.linspace(0, window_image .shape[0]-1, window_image.shape[0])
    line_fitx = left_value[0]*lspace**2 + left_value[1]*lspace+ left_value[2]
    
    lspace2 = np.linspace(0, window_image .shape[0]-1, window_image.shape[0])
    line_fitx2 = right_value[0]*lspace2**2 + right_value[1]*lspace2+ right_value[2]
    
    color_img = np.zeros_like(wrap_output)

# Drawing the polynomial lines on the image     
    left = np.array([np.transpose(np.vstack([line_fitx, lspace]))])
    right = np.array([np.flipud(np.transpose(np.vstack([line_fitx2, lspace2])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(wrap_output, np.int_(points), (0,200,255)) # Fill the space between the lines with polygon
    
    matrix = cv2.getPerspectiveTransform(dst,src) # Homography matrix for inverse Wrapping
    wrap_output_back = cv2.warpPerspective(wrap_output, matrix,(1280,720)) # Inverse wrapping
    superimposed_frame = cv2.bitwise_or(wrap_output_back, orig_frame) # Combing the wrapped image with the orginal image

# Finding the direction of the lanes to take turns    
    y_min = 0
    l_line = left_value[0] * y_min ** 2 + left_value[1] * y_min + left_value[2]
    r_line = right_value[0] * y_min ** 2 + right_value[1] * y_min + right_value[2]

    # Finding center of top of polygon
    m_linetop = l_line + (r_line - l_line) / 2

    y_max = mask.shape[1] - 1
    l_line = left_value[0] * y_max ** 2 + left_value[1] * y_max + left_value[2]
    r_line = right_value[0] * y_max ** 2 + right_value[1] * y_max + right_value[2]

    # Finding center of bottom of polygon
    m_linebottom = l_line + (r_line - l_line) / 2

    # Calculating the deviation of mid-line to predict turns
    deviation = int(m_linetop) - int(m_linebottom)


    if deviation > 5 :
        cv2.putText(superimposed_frame, 'Turn left', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)

    elif deviation < -5:
        cv2.putText(superimposed_frame, 'Turn right', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)

    else:
        cv2.putText(superimposed_frame, 'Go Straight', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)

    

# Displaying the final result
    cv2.imshow("frame1",superimposed_frame)

    cv2.waitKey(1)
video.release()
cv2.destroyAllWindows()