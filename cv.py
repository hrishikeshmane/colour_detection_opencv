import numpy as np
import cv2

# for time being, green colour only
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
#print(lowerBound,upperBound)

#initialize our camera object
cam=cv2.VideoCapture(0)

# to clean noise 
kernalOpen=np.ones((5,5))
kernalClose=np.ones((20,20))

while True:
	ret, img=cam.read()

	# window size
	img=cv2.resize(img,(340,220))

	imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	# to make a white mask on Green Colour
	mask=cv2.inRange(imgHSV,lowerBound,upperBound)
	
	#cv2.imshow("mask",mask)
	#cv2.imshow("cam",img)
	#cv2.waitKey(10)
	
	maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen) 
	maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_OPEN,kernalClose)
	
	# to draw contour
	#maskFinal=maskClose
	maskFinal=maskOpen

	_,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	
	cv2.drawContours(img,conts,-1,(255,0,0),3)
	#print("41")
	for i in range(len(conts)):
		x,y,w,h=cv2.boundingRect(conts[i])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
		#cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    

	cv2.imshow("cam",img)
	cv2.imshow("mask",mask)
	cv2.imshow("maskClose",maskClose)
	cv2.imshow("maskOpen",maskOpen)
	cv2.waitKey(10)
