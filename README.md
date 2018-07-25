# Deeplearning4j-Hackathon

This repository contains a skeleton project for the Deeplearning4j-Hackathon. The hackathon challenge is to implement a Convolutional Neural Network for image classification which is able to distinct images of cats and dogs.
Please follow the installation instructions and the data download (Step 0 and Step 1) in front of the Hackathon!

If you get stuck at any point in this hackathon don't hesitate to ask the friendly OPITZ guys which are happy to help!
The Deeplearning4j website also provides great documentation with many examples and tutorials. Just have a look:
https://deeplearning4j.org/documentation

#Step 0 - Setup your machine learning Dev environment

## Install JDK 8 64 Bit:
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

## Download and Install a Java IDE.
We highly recommend IntelliJ IDEA which also has a free Community Edition. It is very comfortable to use:
www.jetbrains.com/idea/download				

## Download Apache Maven:
https://maven.apache.org/download.cgi

Follow the Maven installation instructions for your system:
https://maven.apache.org/install.html

For Windows:
Unzip the folder and add the "bin" subfolder to your Path environment variable

Ensure that "JAVA_HOME" environment variable points to your JDK.

## Install Git 
https://git-scm.com/download

Then Clone this repository:
git clone https://github.com/thtemme/Deeplearning4j-Hackathon.git

## OPTIONAL: If you have a Nvidia graphic card, install Nvidia CUDA v9.0:
https://developer.nvidia.com/cuda-90-download-archive

You also have to install cuDNN for CUDA v9.0:
https://developer.nvidia.com/cudnn

Add the bin folder of cudnn installation to your PATH environment variable.


## Import Project in your IDE as Maven project

In IntelliJ IDEA follow these steps:
1. Select "File -> New -> Project from existing sources"
2. Select root folder of Git project.
3. Import project from external model -> Maven -> Next
4. Set option "Import Maven projects automatically"
5. Leave all other standard options and click Next until import is finished

### After importing run mvn clean install.

If you have CUDA installed, select "CUDA" as Maven profile. Otherwise set "CPU" as Maven profile.

In IntelliJ go to "View -> Tool Windows -> Maven"
and select under Profiles the relevant Profile.
Then select "Deeplearning4j-Hackathon -> Lifecycle -> install"

Maven dependencies are downloaded for now. This may take a while (up to an hour depending on your internet connection).

Your dev environment is now successfully setup for the Hackathon. Next we have to care about the data.

# Step 1 - Download the data (Images with cats and dogs) from Microsoft:
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

Create resources directory under "src/main/resources" of Git project. Copy the downloaded zip file into the resources folder and unzip it there so that PetImages is a direct child of folder "resources".

Keep the original zip file as backup!


# Step 2 - Preprocessing: Validate and Split Up Input data
Now we are starting to code. The project you have imported has implemented the base functionality for this hackathon. However, there are some ToDos in the code which are a good point for you to start.

Have a look at DataPreprocessing.java

## Validate Input data:
First of all we have to validate the input data. Some of the image data is corrupted. Have a look at Cat\10404.jpg for example. Deeplearning4j will crash during the training phase, if corrupt image files are fed into the pipeline. Therefore we have to delete it. Look at the method "deleteCorruptJpegData()" and fullfil the implementation. You will find hints in the code.

## Split up training and test set

Next we have to split up training and test data. To compare different neural network architectures fairly, we have to create a fixed test set for the whole hackathon. Suggestion is to use 20 % of the data as test data.
Fullfill implementation of method "splitUpTrainingAndTestSet()".

## Execute DataPreprocessing

Finally start the main method in DataPreprocessing. Check if everything works fine. If yes, you will have two directories under resources: "PetImages" for our Training Set and "ValidationPetImages" for our test set.
There should be ~10.000 Cats and ~10.000 Dogs for the training set and ~2500 Cats and ~2500 Dogs for the test set.

# Step 3 - Set up training pipeline

Now we have to setup our training pipeline. Have a look at "CatsDogsClassification". First we have to create an iterator for the training data we want to use. Implement a method generateRecordReaderDataSetIterator which takes a directory as input and returns a "RecordReaderDataSetIterator" as output. You will find hints in the code.

A basic neural network for image classification is returned as "MultiLayerNetwork" in method "hackathonBasicNetwork". As you can see the whole network architecture is setup easily via Builder pattern.
Use this network as starting point to see if everything is working fine. The network is not very good. We will improve it in the next step.

As you can see in the method trainModel() the model is trained easily just by calling network.fit(). 

After a training iteration we would like to see the performance of the trained model. Therefore we have to implement a method "evaluateModel" which prints out the performance of the trained model on the Validation set!


# Step 4 - Run the example and check if everything works fine.

The training phase should start up now. You can watch the training progress by having a look at the Score (=Error rate) after the iterations. You can also call "http://localhost:9000" for a graphical interface of the training progress.
If you receive an error message, please ask one of the OPITZ guys. :-)

# Step 5 - Serialize the trained model after each training epoch:
You may want to save your trained model for later evaluation or further training. Therefore you can use the Deeplearning4j ModelSerializer to save the model after each training iteration.

# Step 6 - Improve the neural network
The hackathon basic network is a very simple one and not really good. There are much better network architecturs out there. These are some examples:
You may have a look at AlexNet, which is a well described one to start: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
Try to setup an improved network for yourself.

If you get stuck at this point, you can also have a look at the Deeplearning4j ModelZoo (https://deeplearning4j.org/model-zoo). You can also load complete Networkconfigurations as dependency in the project. 

# Step 7 - Train your improved network

If you are done with network setup, start the training. Please notice that effective training will take some time (hours until days), especially if you are not using CUDA. So start the training and then it may be time for a bigger break at this point.
You can have a look at the network performance on the validation set from time to time.

# Step 8 - Still not have enough?

You may have a look at other types of neural networks for other learning tasks in Deeplearning4j. Therefore clone the deeplearning4j-examples project, import it to your IDE and see the various samples of DeepLearning in deeplearning4j:
https://github.com/deeplearning4j/dl4j-examples

