import os
import shutil
import random
import numpy
import imutils
import cv2 as cv


def main():

	# Clear out the preprocessed dataset folder
	try:
		shutil.rmtree("./preprocessedTrainingSet")
	except FileNotFoundError:
		pass	

	os.mkdir("./preprocessedTrainingSet")

	categories = os.listdir("./trainingSet")

	for folder in categories:
		print("Processing folder: {}".format(folder))

		# Get the file paths to all the input images
		files = os.listdir("./trainingSet/{}".format(folder))

		# Create an output folder
		os.mkdir("./preprocessedTrainingSet/{}".format(folder))
 
		# Start the output file names counter (for naming files 1.jpg, 2.jpg, etc)
		fileName = 1

		for picture in files:
			# print("File directory: ./trainingSet/{}/{}".format(folder,picture))
			img = cv.imread("./trainingSet/{}/{}".format(folder,picture))

			if img is None:
				continue

			# Crop image
			img1 = cropImage(img, random.uniform(0, 0.1), random.uniform(0, 0.1))
			img2 = cropImage(img, random.uniform(0, 0.1), random.uniform(0, 0.1))
			img3 = cropImage(img, random.uniform(0, 0.1), random.uniform(0, 0.1))

			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img1)
			fileName = fileName + 1
			
			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img2)
			fileName = fileName + 1

			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img3)
			fileName = fileName + 1


			# Rotate image
			img4 = rotateImage(img, random.uniform(0, 360))
			img5 = rotateImage(img, random.uniform(0, 360))
			img6 = rotateImage(img, random.uniform(0, 360))

			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img4)
			fileName = fileName + 1
			
			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img5)
			fileName = fileName + 1

			cv.imwrite("./preprocessedTrainingSet/{}/{}.jpg".format(folder,fileName), img6)
			fileName = fileName + 1


	cv.destroyAllWindows()


def cropImage(img, xOffset, yOffset):	

	height = len(img)
	width = len(img[0])

	xStart = xOffset * width
	xEnd = xOffset + width * 0.9
	yStart = yOffset * height
	yEnd = yOffset + height * 0.9

	return img[int(yStart):int(yEnd), int(xStart):int(xEnd)]


def rotateImage(img, rotationAngle):

	img = imutils.rotate_bound(img, rotationAngle)

	return img


if __name__ == '__main__':
	main()