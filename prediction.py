#Janavi Kumar
#7976608

from PIL import Image
import numpy
import sys
import os 
import tflearn
import sklearn

from tflearn.layers.core import input_data
from tflearn.layers.core import dropout
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from sklearn.model_selection import train_test_split

#set up all the training data
trainData = []
trainLabel = []
for i in range(0, 10):
    for j in range(0,1000):
        trainingLabel = [0] * 10
        trainingLabel[i] = 1
        trainingLabel = [float(k) for k in trainingLabel]

        #load all the images
        image = Image.open('./hw4_train/'+str(i)+'/'+str(i)+'_'+str(j)+'.png')
        img = numpy.asarray(image.getdata())
        img = img.astype(float)
        trainData.append(img)
        trainLabel.append(trainingLabel)

trainData, X, trainLabel, Y = train_test_split(trainData, trainLabel, test_size = 0.1)
finalTrainData = numpy.array(trainData)
finalTrainLabel = numpy.array(trainLabel)
finalX = numpy.array(X)
finalY = numpy.array(Y)
finalTrainData = finalTrainData.reshape([-1,28,28,1])
finalX = finalX.reshape([-1,28,28,1])

#load all the testing data
testData = []
for i in range(0, 10000):
    image = Image.open('./hw4_test/'+str(i)+'.png')
    img = numpy.asarray(image.getdata())
    img = img.astype(float)
    testData.append(img)
finalTestData = numpy.array(testData)
finalTestData = finalTestData.reshape([-1,28,28,1])

#create a convolutional neural network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')

network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')

network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network,name='placeholder_scope')

#train the model and predict
#FOR REPEATABILITY: make sure your model is saved and reaccessed if it already exists

model = tflearn.DNN(network, tensorboard_verbose = 0)
if os.path.isfile('DNNModel.tflearn') or os.path.isfile('DNNModel.tflearn.index'):
    model.load('DNNModel.tflearn')
else:
    model.fit({'input': finalTrainData}, {'placeholder_scope': trainLabel},
              n_epoch=50,
              batch_size=100,
              validation_set=({'input': finalX}, {'placeholder_scope': Y}))
    model.save('DNNModel.tflearn')

pred = numpy.array(model.predict(finalTestData))
pred = numpy.argmax(pred, axis=1)
fstream = open('prediction.txt', 'w+')
for i in range(0, len(pred)):
    predVal = str(pred[i])
    fstream.write(predVal + '\n')
fstream.close()


