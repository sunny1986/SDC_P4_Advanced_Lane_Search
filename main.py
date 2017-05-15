import numpy as np
import glob
import cv2
import pickle
from moviepy.editor import VideoFileClip
from image_generator import Line, abs_sobel_thresh, old_color_threshold, color_threshold, print_road_info
from image_generator import blind_search, not_blind_search, draw_lines

# Import pickle data
pickle_data = pickle.load(open("./camera_cal/calibration_pickle.p","rb"))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

# Get data based on type of data requested
#type_of_data = 'image' 				# To work on image data
type_of_data = 'video' 			# To work on video data

#data = './test_images/test5.jpg'	# Image source
data = 'project_video.mp4' 			# Video source 
									#'project_video.mp4'
									#'challenge_video.mp4' 
									#'harder_challenge_video.mp4'

# Setup left and right lines
left_line = Line()
right_line = Line()

# Where all the magic happens
def process_frame(frame):
	img = frame
	#Undistort the frame	
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	
	filteredImage = np.zeros_like(undist[:,:,0])
	# Apply Sobel filter on the image in x direction	
	sobelx = abs_sobel_thresh(undist,orient = 'x', thresh_min = 50, thresh_max = 255)
	sobely = abs_sobel_thresh(undist,orient = 'y', thresh_min = 25, thresh_max = 255)
	
	# Apply color filter on the image
	c_binary = color_threshold(undist, r_thresh=(50,255), hlsthresh=(100,255), hsvthresh=(220,255))
	
	# Combine the filter results
	filteredImage[((sobelx == 1) & (sobely == 1)) | (c_binary == 1)] = 255
	#cv2.imshow('Filtered Image',filteredImage)
	
	# Define perspective transform
	img_size = (img.shape[1],img.shape[0])
	bottom_width = 0.75
	top_width = 0.10
	height = 0.63
	chop_bottom = 0.94			# Chop the bottom portion of image that has the car hood
	offset = img.shape[1]*0.10 	# Offset ratio controls the width between the lanes
	# Source Points
	src = np.float32([[(0.5*img.shape[1] - 0.5*top_width*img.shape[1]),height*img.shape[0]],	# Point 1
			[(0.5*img.shape[1] + 0.5*top_width*img.shape[1]),height*img.shape[0]],	# Point 2
			[(0.5*img.shape[1] - 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 3
			[(0.5*img.shape[1] + 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 4		
			])
	
	# Destination Points
	dst = np.float32([[offset,0],[img.shape[1]-offset,0],[offset,img.shape[0]],[img.shape[1]-offset,img.shape[0]]])
	
	# Apply perspective transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	binary_warped = cv2.warpPerspective(filteredImage, M, img_size, flags=cv2.INTER_LINEAR)
	#cv2.imshow('Warped Image',binary_warped)
	
	# Search for L&R lines		
	if left_line.detected == False:
		lines_img, curve_l, curve_r = blind_search(binary_warped, left_line, right_line)
	else:
		lines_img, curve_l, curve_r = not_blind_search(binary_warped, left_line, right_line)
	
	#cv2.imshow('Lines Image',lines_img)
		
	# Pass this info to the line class
	left_line.radius_of_curvature = curve_l
	right_line.radius_of_curvature = curve_r
	
	recast_img = draw_lines(binary_warped, Minv, left_line, right_line)
	
	# Combine the recast_image with the undistorted image
	combined = cv2.addWeighted(undist, 1, recast_img, 0.3, 0)
	
	result = print_road_info(combined, left_line, right_line)	
	#cv2.imshow('Recast Image',result)
	
	return result

# If type of data is images use this block of code
if type_of_data == 'image':
	print("Processing Image data")
	# Read an image
	img = cv2.imread(data)
		
	#Undistort the image	
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	
	filteredImage = np.zeros_like(undist[:,:,0])
	# Apply Sobel filter on the image in x direction	
	sobelx = abs_sobel_thresh(undist,orient = 'x', thresh_min = 12, thresh_max = 255)
	sobely = abs_sobel_thresh(undist,orient = 'y', thresh_min = 25, thresh_max = 255)
	# Apply color filter on the image
	c_binary = color_threshold(undist, hlsthresh=(100,255), hsvthresh=(50,255))
	
	# Combine the filter results
	filteredImage[((sobelx == 1) & (sobely == 1)) | (c_binary == 1)] = 255
	
	# Define perspective transform
	img_size = (img.shape[1],img.shape[0])
	bottom_width = 0.75
	top_width = 0.10
	height = 0.70
	chop_bottom = 0.94			# Chop the bottom portion of image that has the car hood
	# Source Points
	src = np.float32([[(0.5*img.shape[1] - 0.5*top_width*img.shape[1]),height*img.shape[0]],	# Point 1
			[(0.5*img.shape[1] + 0.5*top_width*img.shape[1]),height*img.shape[0]],	# Point 2
			[(0.5*img.shape[1] - 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 3
			[(0.5*img.shape[1] + 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 4		
			])
	offset = img.shape[1]*0.10 # Offset ratio controls the width between the lanes
	# Destination Points
	dst = np.float32([[offset,0],[img.shape[1]-offset,0],[offset,img.shape[0]],[img.shape[1]-offset,img.shape[0]]])
	
	# Apply perspective transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	binary_warped = cv2.warpPerspective(filteredImage, M, img_size, flags=cv2.INTER_LINEAR)
	
	# To save images
	#wname = './transf_images/warpy_image.jpg'
	#cv2.imwrite(wname, result)

# If type of data is video use this block of code
# which process the frames and saves the final result as an mp4 file
elif type_of_data == 'video':
	print("Processing video data ...")
	
	clip_in = VideoFileClip(data)
	clip_out = clip_in.fl_image(process_frame)
	clip_out.write_videofile("output_video.mp4", audio = False)