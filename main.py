import argparse
import os 
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.datasets as datasets

def main(doAugs, split, lr, model):

    print("Using model: {}".format(model))
    print("Using data augmentation: {}".format(doAugs))
    print("Using split: {}".format(split))
    print("Learning rate: {}".format(lr))


    loadingTrain, loadingVal, loadingTest = getDataLoaders(doAugs, split)

    n_classes = 5

    model = buildModel(model, n_classes)

    
def buildModel(model, n_classes):

    if model == "alexnet":
        model = models.alexnet(pretrained=True)

    elif model == "vgg19":
        model = models.vgg19(pretrained=True)

    elif model == "googlenet":
        model = models.googlenet(pretrained=True)

    else:
        raise("Model not supported")


    # Freeze parameters
    for param in model.features.parameters():
        param.requires_grad = False


    #Replace the last fully connected layer with a linear layer
    model.fc = nn.Linear(512, n_classes)

    model.cuda() 

    return model



def getDataLoaders(doAugs, split):

    trainingFolder = os.path.join(os.getcwd(),"{}-{}-{}/trainingSet".format(split[0], split[1], split[2]))
    validationFolder = os.path.join(os.getcwd(),"{}-{}-{}/validationSet".format(split[0], split[1], split[2]))
    testFolder = os.path.join(os.getcwd(),"{}-{}-{}/testSet".format(split[0], split[1], split[2]))
    

    # From Stanford Cats and Dogs dataset...
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]


    trainTrans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
    ])


    sameSizeTrans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
    ])


    datasetTrain = datasets.ImageFolder(trainingFolder, trainTrans if doAugs else sameSizeTrans)
    datasetVal = datasets.ImageFolder(validationFolder, sameSizeTrans)
    datasetTest = datasets.ImageFolder(testFolder, sameSizeTrans)


    loadingTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=4, shuffle=True, num_workers=8)
    loadingVal = torch.utils.data.DataLoader(datasetVal, batch_size=4, shuffle=False, num_workers=8)
    loadingTest =  torch.utils.data.DataLoader(datasetTest, batch_size=4, shuffle=False, num_workers=8)

    return loadingTrain, loadingVal, loadingTest




if __name__ == '__main__':

    arguments = argparse.ArgumentParser()
    arguments.add_argument("--a", default=1, type=int, help="Do data augmentation 1 for Y, 0 for N")
    arguments.add_argument("--s", default=1, type=int, help="Split to use 1 for 80-10-10, 0 for 50-25-25")
    arguments.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    arguments.add_argument("--mod", default="alexnet")
    arguments = arguments.parse_args()

    doAugs = True if arguments.a==1 else False
    split = [80,10,10] if arguments.s==1 else [50,25,25]

    main(doAugs, split, arguments.lr, arguments.mod)




