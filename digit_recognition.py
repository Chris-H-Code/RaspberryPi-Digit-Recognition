import cv2
import time
from tensorflow import keras
from tensorflow. keras.models import load_model
from keras.utils import np_utils
import numpy as np
import imutils
from sense_hat import SenseHat

SenseHat = SenseHat()
cap = cv2.VideoCapture(0)
model = load_model('mnist_keras_convnet.h5')
num_classes = 10


def Loop():
	while(True):

		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.imshow('frame', frame)
		time.sleep(.5)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			cap.release()
			cv2.destroyAllWindows()
			break
		#elif cv2.waitKey(1) & 0xFF == ord('t'):
		cv2.imwrite('/home/pi/RaspberryPiProjects/image.png', frame)
		image = cv2.imread('/home/pi/RaspberryPiProjects/image.png')
		PredictImage(image)

def PredictImage(image):
	ret, thresh = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
	kernel = np.ones((5, 5), np.uint8)
	erode = cv2.erode(thresh, kernel, iterations=1)
	result = cv2.bitwise_or(image, erode)
	cv2.imwrite('/home/pi/RaspberryPiProjects/test.png', result)
	image = cv2.imread('/home/pi/RaspberryPiProjects/test.png')
	image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.bitwise_not(image)
	image = image.astype('float32')/255
	image = image.reshape(1, 28, 28, 1)
	yhat = model.predict(image)
	yhat_2 = model.predict_classes(image)
	yhat_2_list = keras.utils.to_categorical(yhat_2, num_classes)
	predicted = np.max(yhat)
	if predicted >= 0.65:
		print(f'Digit predicted! The predicted digit is: {yhat_2}')
		print(f'The accuracy is: {predicted*100}')
		prediction = int(yhat_2[0])
		prediction = str(prediction)
		#SenseHat.clear()
		SenseHat.show_letter(prediction, text_colour=[0, 0, 255])
		time.sleep(.5)
	Loop()

Loop()

#prediction = int(yhat[0])

#prediction = str(prediction)

#SenseHat.show_letter(prediction, text_colour=[0, 0, 255])
