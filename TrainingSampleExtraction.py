
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


np.set_printoptions(threshold=np.nan)
sift = cv2.xfeatures2d.SIFT_create()

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
#given training sample path, return extracted features and corresponding feature label
def TrainingSampleFeaturesGenerator(train_path):
	training_names = mylistdir(train_path)
	image_paths = []
	image_classes = []
	class_id = 0
	for training_name in training_names:
	    dir = os.path.join(train_path, training_name)
	    class_path = imutils.imlist(dir)
	    image_paths+=class_path
	    image_classes+=[training_name]*len(class_path)

	# List where all the descriptors are stored
	des_list = []

	for image_path in image_paths:
	    im = cv2.imread(image_path)
	    kpts, des = sift.detectAndCompute(im, None)
	    des_list.append((image_path, des))   
	    
	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[1:]:
	    descriptors = np.vstack((descriptors, descriptor))  

	# Perform k-means clustering
	k = 100
	voc, variance = kmeans(descriptors, k, 1) 

	# Calculate the histogram of features
	im_features = np.zeros((len(image_paths), k), "float32")
	for i in xrange(len(image_paths)):
	    words, distance = vq(des_list[i][1],voc)
	    for w in words:
	        im_features[i][w] += 1
	# Scaling the words
	stdSlr = StandardScaler().fit(im_features)
	im_features = stdSlr.transform(im_features)

	# Save the SVM
	joblib.dump((stdSlr, k, voc), "bof.pkl", compress=3)   
	return im_features,  image_classes


#given test sample, return test sample features,
def TestSampleFeaturesGenerator(image_path):
	stdSlr, k, voc = joblib.load("bof.pkl")

	image_paths = imutils.imlist(image_path)
# List where all the descriptors are stored
	des_list = []

	for image_path in image_paths:
	    im = cv2.imread(image_path)
	    if im == None:
	        print "No such file {}\nCheck if the file exists".format(image_path)
	        exit()
	    kpts, des = sift.detectAndCompute(im, None)
	    des_list.append((image_path, des))   

	# Stack all the descriptors vertically in a numpy array
	# print des_list
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[0:]:
	    descriptors = np.vstack((descriptors, descriptor)) 
	# 
	test_features = np.zeros((len(image_paths), k), "float32")
	for i in xrange(len(image_paths)):
	    words, distance = vq(des_list[i][1],voc)
	    for w in words:
	        test_features[i][w] += 1

	# Scale the features
	test_features = stdSlr.transform(test_features)

	return test_features

im_features, image_classes = TrainingSampleFeaturesGenerator("dataset/train")
test_features = TestSampleFeaturesGenerator("dataset/test")
print im_features
print image_classes
print test_features
	    