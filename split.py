import os
import shutil
import random
import sys


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


	categories = os.listdir("./dataset")

	for folder in categories:
		print("Processing folder: {}".format(folder))

		# Get the file paths to all the input images
		files = os.listdir("./dataset/{}".format(folder))

		# Creqte an training set output folders
		os.mkdir("./trainingSet/{}{}-{}".format(trainingSplit,"%",folder))

		# Create a validation set output folder
		os.mkdir("./validationSet/{}{}-{}".format(validationSplit,"%",folder))

		# Create a test set output folder
		os.mkdir("./testSet/{}{}-{}".format(testSplit,"%",folder))

		
		# Start the output file names counter (for naming files 1.jpg, 2.jpg, etc)
		fileName = 1


if __name__ == '__main__':
	main()