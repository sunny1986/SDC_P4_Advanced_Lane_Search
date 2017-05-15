import numpy as np
import glob
import cv2
import pickle

#prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

# Fetch a bunch of images for calibration
images = glob.glob('./camera_cal/calibration*.jpg')

# Process each image
for idx, filename in enumerate(images):
	#print(filename)
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# Find corners of the chessboard
	ret, corners = cv2.findChessboardCorners(gray,(9,6), None)
	
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
		
		# Draw the corners on the chessboard and save the images
		cv2.drawChessboardCorners(img,(9,6), corners, ret)
		wname = './camera_cal/chesscorners'+str(idx)+'.jpg'
		cv2.imwrite(wname, img)
		
img = cv2.imread('./camera_cal/calibration1.jpg')
size = (img.shape[1],img.shape[0])

# Generate required matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

undist = cv2.undistort(img, mtx, dist, None, mtx)

wname = './output_images/undistorted.jpg'
cv2.imwrite(wname, undist)
	
# Save the required matrices in a pickle file
pickle_data = {}
pickle_data["mtx"] = mtx
pickle_data["dist"] = dist
pickle.dump(pickle_data, open("./camera_cal/calibration_pickle.p","wb"))
