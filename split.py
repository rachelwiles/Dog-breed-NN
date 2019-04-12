import os
import shutil
import random
import sys
import math
import cv2 as cv

# NB: This will take all classes and split each class the % split defined. As the 
# classes are not even in size, preprocessing will need to occur on the training set 
# classes so that there is now an even number of images across all classes. This is to 
# prevent training a biased model towards a particular class with more images in the 
# training set. 

def main():

	# Clear out the dataset folders
	try:
		shutil.rmtree("./trainingSet")
	except FileNotFoundError:
		pass	

	os.mkdir("./trainingSet")

	try:
		shutil.rmtree("./testSet")
	except FileNotFoundError:
		pass	

	os.mkdir("./testSet")

	try:
		shutil.rmtree("./validationSet")
	except FileNotFoundError:
		pass	

	os.mkdir("./validationSet")

	trainingSplit = input("Enter the % training split:")
	validationSplit = input("Enter the % validation split:")
	testSplit = input("Enter the % test split:")


	if int(trainingSplit) + int(validationSplit) + int(testSplit) == 100:
		print("Splitting dataset...")
	else:
		print("Error: Invalid Split")
		quit()

		
	categories = os.listdir("./originalImages")


	for folder in categories:

		print("-------------------------------------")
		print("Processing class {}".format(folder))

		# Get the file paths to all the input images
		files = os.listdir("./originalImages/{}".format(folder))

		# Get number of files in each folder
		numberOfFiles = len([name for name in files if os.path.isfile(os.path.join("./originalImages/{}".format(folder), name))])
		print("Total images: {}".format(numberOfFiles))

		numberTrainingImages = math.floor(numberOfFiles / 100* int(trainingSplit))
		numberValidationImages = math.floor(numberOfFiles / 100 * int(validationSplit)) 
		numberTestImages = math.floor(numberOfFiles / 100 * int(testSplit))
		print("Number of training images: {}".format(numberTrainingImages))
		print("Number of validation images: {}".format(numberValidationImages))
		print("Number of test images: {}".format(numberTestImages))

		#-------------------------
		# SOMETHING HERE TO CONVERT % SPLIT TO NUMBER OF FILES FOR EACH FOLDER TO OUTPUT
		#-------------------------

		# Create an training set output folders
		os.mkdir("./trainingSet/{}{}-{}".format(trainingSplit,"%",folder))

		# Create a validation set output folder
		os.mkdir("./validationSet/{}{}-{}".format(validationSplit,"%",folder))

		# Create a test set output folder
		os.mkdir("./testSet/{}{}-{}".format(testSplit,"%",folder))

		fileName = 1

		for pictureName in files:
			rnumber = random.randint(1,101)
			picture = cv.imread("./originalImages/{}/{}".format(folder,pictureName))


			if rnumber <= int(trainingSplit):
				cv.imwrite("./trainingSet/{}{}-{}/{}.jpg".format(trainingSplit,"%",folder,fileName), picture)
				fileName = fileName + 1

			elif rnumber <= int(trainingSplit) + int(validationSplit):
				cv.imwrite("./validationSet/{}{}-{}/{}.jpg".format(validationSplit,"%",folder,fileName), picture)
				fileName = fileName + 1

			else:
				cv.imwrite("./testSet/{}{}-{}/{}.jpg".format(testSplit,"%",folder,fileName), picture)
				fileName = fileName + 1
		


if __name__ == '__main__':
	main()