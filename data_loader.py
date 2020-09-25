import _pickle as cPickle
import gzip
import cv2
import os
import numpy as np
from skimage.filters import threshold_otsu

def get_root_directory():
	current_dir = os.path.dirname(os.path.realpath(__file__))
	dir_split = os.path.split(current_dir)
	root_directory = dir_split[0]
	return root_directory

def read_training_data(training_directory):
	letters = [
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
		'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
		'U', 'V', 'W', 'X', 'Y', 'Z'
	]
	image_data = []
	image_testdata = []
	target_data = []
	target_testdata = []
	
	for each_letter in letters:
		for each in range(10):
			img_details = cv2.imread(training_directory+'/'+each_letter+'/'+each_letter+'_'+str(each)+'.jpg', cv2.IMREAD_GRAYSCALE)

			
			#print(type(img_details))
            #cv2.imshow("Image",img_details)
			#cv2.waitKey(0)
			#_,thresh_image = cv2.threshold(img_details, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			#cv2.imshow("thresh",thresh_image)
			#cv2.waitKey(0)
			
			#binary_image = img_details < threshold_otsu(img_details)
			
						
			flat_bin_image = np.reshape(img_details,(400,1))
			flat_bin_image=flat_bin_image.astype(np.float32)/255
			if(each >= 7 and each <=9):
				image_testdata.append(flat_bin_image)
				target_testdata.append(each_letter)
			else:
				image_data.append(flat_bin_image)
				target_data.append(each_letter)
	return (np.array(image_data), np.array(target_data),np.array(image_testdata), np.array(target_testdata))

def load_data():
	root_directory = get_root_directory()

	training_20X20_dir = os.path.join(root_directory, 'training_data', 'train20X20')
	
	
	tr_d,tr_dl,te_d,te_dl = read_training_data(training_20X20_dir)
	
	training_results = [vectorized_result(y) for y in tr_dl]
	training_data = zip(tr_d, training_results)
#	test_inputs = [for x in te_d[0]]
	test_data = zip(te_d, te_dl)
	print(training_data)	

#	print(tr_d[0])
#	cnt=0
#	for values in training_data:
#		print(values)
#		cnt+=1
#	print("training data:",cnt)
#	cnt=0
#	for values in test_data:
#		print(values)
#		cnt+=1
#	print("testing data:",cnt)
	
	return (training_data, test_data)

def vectorized_result(j):
	if(j>='A' and j<='Z'):
		j=ord(j)-65+10
	e = np.zeros((36, 1))
	e[int(j)] = 1
	#print(e)
	return e
#training_data,test_data=data_loader.load_data()
