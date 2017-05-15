import numpy as np
import glob
import cv2
import pickle

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
		
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
	"""Sobel gradient function"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a template of zeros and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
    # Set value to 1 for pixels in between the thresholds 
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
	return binary_output

def color_threshold(img, r_thresh=(0,255), hlsthresh=(0, 255), hsvthresh=(0,255)):
	"""Function that thresholds the R, S and V-channels of HLS"""
	# Filter out the r channel
	r_channel = img[:,:,2]
	# Create a template of zeros the size of the image
	r_binary_output = np.zeros_like(r_channel)    
	# Threshold all pixels to 1 in between the threshold min and max
	r_binary_output[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
	# Convert RGB to HLS color space
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	# Filter out the s channel
	s_channel = hls[:,:,2]
	# Create a template of zeros the size of the image
	s_binary_output = np.zeros_like(s_channel)
	# Threshold all pixels to 1 in between the threshold min and max
	s_binary_output[(s_channel > hlsthresh[0]) & (s_channel <= hlsthresh[1])] = 1
	
	# Convert RGB to HSV color space
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	# Filter out the v channel
	v_channel = hsv[:,:,2]
	# Create a template of zeros the size of the image
	v_binary_output = np.zeros_like(v_channel)
	# Threshold all pixels to 1 in between the threshold min and max
	v_binary_output[(v_channel > hsvthresh[0]) & (v_channel <= hsvthresh[1])] = 1	
	
	comb_output = np.zeros_like(s_channel)	
	comb_output[((s_binary_output == 1) & (v_binary_output == 1)) | 
                ((r_binary_output == 1) & (v_binary_output == 1)) ] = 1
	# Return the results
	return comb_output

def rad_of_curvature(ploty, left_fit, right_fit, leftx, rightx):
	""" Function that calculates the radius of curvature of the road"""
	leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
	rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
	
	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(
					2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(
					2*right_fit[0])
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	
	return (left_curverad, right_curverad)

def smoothing_func(recent_fits, prev_fits = 5):
	""" Funtion to smoothen the lanes frame-to-frame and reduce flickering"""
	recent_fits = np.squeeze(recent_fits)
	avg_fit = np.zeros(720)
	
	for i, fit in enumerate(reversed(recent_fits)):
		if i == prev_fits:
			break
		avg_fit = avg_fit + fit

	avg_fit = avg_fit / prev_fits
	
	return avg_fit
	
def blind_search(binary_warped, left_line, right_line):
	""" Blind search function: Used for 1st frame or if no info from last frame"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 65
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Create  array to be used in next step
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Send the polynomial coefficients to the line class
	left_line.current_fit = left_fit
	right_line.current_fit = right_fit	
	
	# ax^2 + bx + c
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
	
	# Send the fitx for each line to the line class
	left_line.recent_xfitted.append(left_fitx)
	right_line.recent_xfitted.append(right_fitx)
	
	if len(left_line.recent_xfitted) > 10:
		avg_left_line = smoothing_func(left_line.recent_xfitted, 10)
		avg_left_fit = np.polyfit(ploty, avg_left_line, 2)
		avg_left_fitx = avg_left_fit[0] * ploty ** 2 + avg_left_fit[1] * ploty + avg_left_fit[2]
		left_line.current_fit = avg_left_fit
		left_line.allx = avg_left_fitx
		left_line.ally = ploty
	else:
		left_line.current_fit = left_fit
		left_line.allx = left_fitx
		left_line.ally = ploty
		
	if len(right_line.recent_xfitted) > 10:
		avg_right_line = smoothing_func(right_line.recent_xfitted, 10)
		avg_right_fit = np.polyfit(ploty, avg_right_line, 2)
		avg_right_fitx = avg_right_fit[0] * ploty ** 2 + avg_right_fit[1] * ploty + avg_right_fit[2]
		right_line.current_fit = avg_right_fit
		right_line.allx = avg_right_fitx
		right_line.ally = ploty
	else:
		right_line.current_fit = right_fit
		right_line.allx = right_fitx
		right_line.ally = ploty
	"""
	left_line.current_fit = left_fit
	left_line.allx = left_fitx
	left_line.ally = ploty
	
	right_line.current_fit = right_fit
	right_line.allx = right_fitx
	right_line.ally = ploty
	"""
	
	left_line.line_base_pos = left_line.allx[len(left_line.allx)-1]
	right_line.line_base_pos = right_line.allx[len(right_line.allx)-1]
	
	left_line.detected = True
	right_line.detected = True
	
	left_curverad, right_curverad = rad_of_curvature(ploty, left_fit, right_fit, left_fitx, right_fitx)
	
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	#print("Blind search")
	return (out_img, left_curverad, right_curverad)

def not_blind_search(binary_warped, left_line, right_line):
	"""not blind search function: Used when there is info available from last frame"""
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 65
		
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		
	# Receive the polynomial coefficients from line class
	left_fit = left_line.current_fit
	right_fit = right_line.current_fit
		
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
		
	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)	
	
	# ax^2 + bx + c
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# Send the fitx for each line to the line class
	left_line.recent_xfitted.append(left_fitx)
	right_line.recent_xfitted.append(right_fitx)
	
	if len(left_line.recent_xfitted) > 10:
		avg_left_line = smoothing_func(left_line.recent_xfitted, 10)
		avg_left_fit = np.polyfit(ploty, avg_left_line, 2)
		avg_left_fitx = avg_left_fit[0] * ploty ** 2 + avg_left_fit[1] * ploty + avg_left_fit[2]
		left_line.current_fit = avg_left_fit
		left_line.allx = avg_left_fitx
		left_line.ally = ploty
	else:
		left_line.current_fit = left_fit
		left_line.allx = left_fitx
		left_line.ally = ploty
		
	if len(right_line.recent_xfitted) > 10:
		avg_right_line = smoothing_func(right_line.recent_xfitted, 10)
		avg_right_fit = np.polyfit(ploty, avg_right_line, 2)
		avg_right_fitx = avg_right_fit[0] * ploty ** 2 + avg_right_fit[1] * ploty + avg_right_fit[2]
		right_line.current_fit = avg_right_fit
		right_line.allx = avg_right_fitx
		right_line.ally = ploty
	else:
		right_line.current_fit = right_fit
		right_line.allx = right_fitx
		right_line.ally = ploty
	"""
	left_line.current_fit = left_fit
	left_line.allx = left_fitx
	left_line.ally = ploty
	
	right_line.current_fit = right_fit
	right_line.allx = right_fitx
	right_line.ally = ploty
	"""
	left_line.line_base_pos = left_line.allx[len(left_line.allx)-1]
	right_line.line_base_pos = right_line.allx[len(right_line.allx)-1]
	
	# Switch to blind search if std. dev of lane lines is high
	
	std_dev = np.std(right_line.allx - left_line.allx)
	
	if std_dev > 100:
		left_line.detected = False
	
	left_curverad, right_curverad = rad_of_curvature(ploty, left_fit, right_fit, left_fitx, right_fitx)
	
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	#print("Not Blind search")
	return (out_img, left_curverad, right_curverad)
	
def draw_lines(img, Minv, left_line, right_line):
	""" Recast the lanes found onto the original image """
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(img).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	# Get relevant data from line class
	left_fitx = left_line.allx
	right_fitx = right_line.allx
	ploty = left_line.ally
	
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	result = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
	
	return result
	
def print_road_info(img, left_line, right_line):
	""" Function to print road info like curvature and deviation on the frames """
	curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
	lane_center = (left_line.line_base_pos + right_line.line_base_pos) / 2
	
	# Define conversions in x from pixels space to meters	
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	img_center = 1280/2
	
	deviation = abs(lane_center - img_center)*xm_per_pix
	if lane_center > img_center:
		deviation = 'Right --> {0:0.3f} meters'.format(deviation)
	elif lane_center < img_center:
		deviation = 'Left <-- {0:0.3f} meters'.format(deviation)
	else:
		deviation = 'Center'
		
	curve_info = 'Curvature of road: {0:0.3f} meters'.format(curvature)
	deviation_info = 'Deviation from lane center: ' + deviation

	cv2.putText(img, 'Road Info', (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(img, curve_info, (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	cv2.putText(img, deviation_info, (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	
	return img
	
	
	
	
	
