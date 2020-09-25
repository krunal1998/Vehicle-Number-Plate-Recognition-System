# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
from PIL import Image
import pytesseract as tess
import network

def preprocess(img):
	cv2.imshow("Input",img)
	cv2.waitKey(0)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray",gray)
	cv2.waitKey(0)
	imgBlurred = cv2.GaussianBlur(gray, (5,5), 0)
	cv2.imshow("blur",imgBlurred)
	cv2.waitKey(0)

	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
	cv2.imshow("Sobel",sobelx)
	cv2.waitKey(0)
#	threshold_img = cv2.adaptiveThreshold(sobelx,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,43,2)
	ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow("Threshold",threshold_img)
	cv2.waitKey(0)
	return threshold_img

def cleanPlate(plate):
	print ("CLEANING PLATE. . .")
	cv2.imshow("candidate for plate",plate)
	cv2.waitKey(0)
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	thresh= cv2.dilate(gray, kernel, iterations=1)

#	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,43,2)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#	cv2.imshow("adaptive",thresh)
#	cv2.waitKey(0)
	temp=thresh.copy()
	
	character_dimensions = (0.30*plate.shape[0], 0.80*plate.shape[0], 0.02*plate.shape[1], 0.15*plate.shape[1])
	min_height, max_height, min_width, max_width = character_dimensions
	
	im1,contours,hierarchy = cv2.findContours(temp,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects = [cv2.boundingRect(ctr) for ctr in contours]
	
	characters = []
	
	if contours:
		for rect in reversed(rects):
#rect[0]=X, rect[1]=Y, rect[2]=width, rect[3]=height
#			if (rect[2] < rect[3] and rect[2]>3	 and rect[2]<30 and rect[3]>10 and rect[3]<35):
			if (rect[2] < rect[3] 	 and rect[2]<max_width and rect[3]>min_height and rect[3]<max_height):
				
				cleaned_final = plate[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
#				resized_img=cv2.resize(cleaned_final,(20,20))
				gray_image = cv2.cvtColor(cleaned_final, cv2.COLOR_BGR2GRAY)
				resized_img=cv2.resize(gray_image,(20,20))
				cv2.imshow("resized",resized_img)
				cv2.waitKey(0)
				
#				cv2.rectangle(plate, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0),1)
#				cv2.imshow("temp",plate)
#				cv2.waitKey(0)

				
				flat_bin_image = np.reshape(resized_img,(400,1))
				flat_bin_image=flat_bin_image.astype(np.float32)/255
				characters.append(flat_bin_image)
				#cv2.imshow("Function Test",resized_img)
				#cv2.waitKey(0)
		height, width = plate.shape[:2]
		return plate,[0,0,width,height],np.array(characters)

	else:
		return plate,None,None


def extract_contours(threshold_img):
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
	morph_img_threshold = threshold_img.copy()
	cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
	cv2.imshow("Morphed",morph_img_threshold)
	cv2.waitKey(0)

	im2,contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
	return contours


def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	min = 15*aspect*15  
	max = 125*aspect*125  

	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

def isMaxWhite(plate):
	avg = np.mean(plate)
	if(avg>=115):
		return True
	else:
 		return False

def validateRotationAndRatio(rect):
	(x, y), (width, height), rect_angle = rect

	if(width>height):
		angle = rect_angle
	else:
		angle = 90 + rect_angle

	if angle>15:
	 	return False

	if height == 0 or width == 0:
		return False

	area = height*width
	if not ratioCheck(area,width,height):
		return False
	else:
		return True



def cleanAndRead(img,contours,net):
	for i,cnt in enumerate(contours):
		min_rect = cv2.minAreaRect(cnt)

		if validateRotationAndRatio(min_rect):

			x,y,w,h = cv2.boundingRect(cnt)
			plate_img = img[y:y+h,x:x+w]
			


			if(isMaxWhite(plate_img)):
				clean_plate, rect, characters = cleanPlate(plate_img)

#				plate_im = Image.fromarray(clean_plate)
#				text = tess.image_to_string(plate_im, lang='eng')
#				print(text)
                					
				#cv2.imshow("clean",clean_plate)
				#cv2.waitKey(0)
				if rect:
					x1,y1,w1,h1 = rect
					x,y,w,h = x+x1,y+y1,w1,h1
					cv2.imshow("Cleaned Plate",clean_plate)
					cv2.waitKey(0)
					
					plate_im = Image.fromarray(clean_plate)
					text = tess.image_to_string(plate_im, lang='eng')
					unexpected_text = "!,:@#$&*./[]/'- "
					for x in text:
						
						if x in unexpected_text:
							text = text.replace(x,"") 
						if ( ord(x) == 8216 or ord(x) == 8217 or ord(x) == 8218):
							text = text.replace(x,"")
					if text is not "":
						print("Detected text: ")
						print(text)
#					print ("Detected Text : ")
#					predicted_char=""
#					for x in characters:
#						predicted_char = predicted_char+ net.predict(x)
#					print(predicted_char)
#					img = cv2.rectangle(img , (x,y), (x+w,y+h), (0,255,0), 1 )
#					cv2.imshow("ori",img)
#					cv2.waitKey(0)



if __name__ == '__main__':
	print ("DETECTING PLATE . . .")
	img = cv2.imread("C:/Users/Admin/Desktop/project/ml_code/car_image/47.JPG")
	cv2.imshow("Input",img)
	net = network.Network([400, 100, 36])
	threshold_img = preprocess(img)
	contours= extract_contours(threshold_img)
	cleanAndRead(img,contours,net)
