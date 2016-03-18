
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import csv
import requests
import ast
import urllib, cStringIO

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# r = requests.get('http://wmcalyj.pythonanywhere.com/getImages')
# r = r.text
# r = ast.literal_eval(r)
# for pill in r:
# 	for url in r[pill]:
# 		print url


def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

np.set_printoptions(threshold=np.nan)
sift = cv2.xfeatures2d.SIFT_create()


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


def TrainingSampleFeaturesGeneratorPath(train_path):
	training_names = mylistdir(train_path)
	image_paths = []
	image_classes = []
	class_id = 0
	for training_name in training_names:
	    dir = os.path.join(train_path, training_name)
	    class_path = imutils.imlist(dir)
	    print class_path
	    image_paths+=class_path
	    image_classes+=[training_name]*len(class_path)

	image_names = np.reshape(image_paths, (-1,1))

	des_list = []
	HH = []



	for image_path in image_paths:
		im = url_to_image(image_path)
		if im == None:
			print "No such file {}\nCheck if the file exists".format(image_path)
			exit()
		kernel = np.ones((50,50),np.float32)/2500
		im = cv2.filter2D(im, -1, kernel)
		kpts, des = sift.detectAndCompute(im, None)
		hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
		hsv = cv2.filter2D(hsv,-1,kernel)
		h_hue = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
		H = []
		n_hue = sum(h_hue)
		for h in h_hue:
			hh = np.float32(float(h)/float(n_hue))
			H.append(hh)
		h_sat = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
		temp = []
		temp.append(np.std(H, ddof = 1))
		# H = []
		n_sat = sum(h_sat)
		for h in h_sat:
			hh = np.float32(float(h)/float(n_sat))
			H.append(hh)
		temp.append(np.std(H,ddof = 1))
		HH.append(H)
		des_list.append((image_path, des))   

	
	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[0:]:
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
	image_classes = np.reshape(image_classes, (-1,1))
	im_features = np.append(im_features, HH, axis = 1)
	res = np.append(im_features, image_classes, axis = 1)
	# res = np.append(image_names, res, axis = 1)
	fl = open('FeatureSample.csv', 'w')

	writer = csv.writer(fl)
	for values in res:
	    writer.writerow(values)

	fl.close() 
	return im_features,  image_classes

#given training sample path, return extracted features and corresponding feature label
def TrainingSampleFeaturesGenerator():
	image_paths = []
	image_classes = []
	class_id = 0
	# for training_name in training_names:
	#     dir = os.path.join(train_path, training_name)
	#     class_path = imutils.imlist(dir)
	#     print class_path
	#     image_paths+=class_path
	#     image_classes+=[training_name]*len(class_path)
	# List where all the descriptors are stored

	r = requests.get('http://wmcalyj.pythonanywhere.com/getImages')
	r = r.text
	r = ast.literal_eval(r)


	for pill in r:
		for pill_path in r[pill]:
			image_paths.append(pill_path)
			image_classes.append(pill)

	# image_names = np.reshape(image_paths, (-1,1))
	# print "image_names is "
	# print image_names
	print r
	des_list = []
	HH = []



	for image_path in image_paths:
		im = url_to_image(image_path)
		if im == None:
			print "No such file {}\nCheck if the file exists".format(image_path)
			exit()
		kpts, des = sift.detectAndCompute(im, None)
		hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
		kernel = np.ones((50,50),np.float32)/2500
		hsv = cv2.filter2D(hsv,-1,kernel)
		h_hue = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
		H = []
		n_hue = sum(h_hue)
		for h in h_hue:
			hh = np.float32(float(h)/float(n_hue))
			H.append(hh)
		h_sat = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
		temp = []
		temp.append(np.std(H, ddof = 1))
		# H = []
		n_sat = sum(h_sat)
		for h in h_sat:
			hh = np.float32(float(h)/float(n_sat))
			H.append(hh)
		temp.append(np.std(H,ddof = 1))
		HH.append(H)
		des_list.append((image_path, des))   

	
	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[0:]:
		print image_path
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
	image_classes = np.reshape(image_classes, (-1,1))
	im_features = np.append(im_features, HH, axis = 1)
	res = np.append(im_features, image_classes, axis = 1)
	# res = np.append(image_names, res, axis = 1)
	fl = open('FeatureSample.csv', 'w')

	writer = csv.writer(fl)
	for values in res:
	    writer.writerow(values)

	fl.close() 
	return im_features,  image_classes






def TestSampleFeaturesGeneratorWithLabel(train_path):
	stdSlr, k, voc = joblib.load("bof.pkl")
	training_names = mylistdir(train_path)
	image_paths = []
	image_classes = []
	class_id = 0
	for training_name in training_names:
	    dir = os.path.join(train_path, training_name)
	    class_path = imutils.imlist(dir)
	    image_paths+=class_path
	    image_classes+=[training_name]*len(class_path)
	des_list = []
	HH = []
	print image_paths
	image_names = np.reshape(image_paths, (-1,1))
	print image_names
	for image_path in image_paths:
	    im = cv2.imread(image_path)
	    if im == None:
	        print "No such file {}\nCheck if the file exists".format(image_path)
	        exit()
	    kpts, des = sift.detectAndCompute(im, None)
	    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
	    kernel = np.ones((50,50),np.float32)/2500
	    hsv = cv2.filter2D(hsv,-1,kernel)
	    h_hue = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
	    H = []
	    n_hue = sum(h_hue)
	    for h in h_hue:
	        hh = np.float32(float(h)/float(n_hue))
	        H.append(hh)
	    h_sat = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
	    temp = []
	    temp.append(np.std(H, ddof = 1))
	    # H = []
	    n_sat = sum(h_sat)
	    for h in h_sat:
	        hh = np.float32(float(h)/float(n_sat))
	        H.append(hh)
	    temp.append(np.std(H,ddof = 1))
	    HH.append(H)
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
	image_classes = np.reshape(image_classes, (-1,1))
	test_features = np.append(test_features, HH, axis = 1)
	res = np.append(test_features, image_classes, axis = 1)
	res = np.append(image_names, res, axis = 1)
	fl = open('TestFeatureWithLabel.csv', 'w')

	writer = csv.writer(fl)
	for values in res:
	    writer.writerow(values)

	fl.close() 
	return res




#given test sample, return test sample features,
def TestSampleFeaturesGenerator(image_path):
	stdSlr, k, voc = joblib.load("bof.pkl")

	image_paths = imutils.imlist(image_path)
# List where all the descriptors are stored
	des_list = []
	HH = []
	for image_path in image_paths:
	    im = cv2.imread(image_path)
	    if im == None:
	        print "No such file {}\nCheck if the file exists".format(image_path)
	        exit()
	    kpts, des = sift.detectAndCompute(im, None)
	    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
	    kernel = np.ones((50,50),np.float32)/2500
	    hsv = cv2.filter2D(hsv,-1,kernel)
	    h_hue = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
	    H = []
	    n_hue = sum(h_hue)
	    for h in h_hue:
	        hh = np.float32(float(h)/float(n_hue))
	        H.append(hh)
	    
	    h_sat = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
	    n_sat = sum(h_sat)
	    for h in h_sat:
	        hh = np.float32(float(h)/float(n_sat))
	        H.append(hh) 
	    HH.append(H)
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
	test_features = np.append(test_features, HH, axis = 1)
	fl = open('TestFeature.csv', 'w')

	writer = csv.writer(fl)
	for values in test_features:
	    writer.writerow(values)

	fl.close() 
	return test_features


im_features, image_classes = TrainingSampleFeaturesGenerator()
# test_features = TestSampleFeaturesGenerator("dataset/test")
