import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while(True):

	_, frame = cap.read()
	hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_black = np.array([0, 5, 50])
	upper_black = np.array([179, 50, 255])

	mask = cv2.inRange(hsv, lower_black, upper_black)
	
	res = cv2.bitwise_and(frame, frame, mask = mask)
	
	#cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)
	time.sleep(.5)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
