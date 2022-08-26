# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:31:36 2022

@author: zachh
"""

# written by Zach Hughes  6/13/2022

import tensorflow as tf
import numpy as np
import time
import os

path = os.getcwd()
#importing mnist dataset
(train_x,train_y),(test_x,test_y) = tf.keras.datasets.mnist.load_data()
#train_x has shape (60000,28,28)
#test_x has shape (10000,28,28)

#now splitting train data into
#training data and validation data
x_val = train_x[50000:60000]
x_train = train_x[:50000]
y_val = train_y[50000:60000]
y_train = train_y[:50000]


#now reshaping. taking 28x28 matrix and making
#a vector of length 784 for NN 
x_train = x_train.reshape(50000,784)
x_val = x_val.reshape(10000,784)
x_test = test_x.reshape(10000,784) #renaming to match naming convensions

#now converting data to float 32 (not certain why we do this outside of possible overflow)
#also  normalizing data. input data is in grayscale, on range from 0(white) to 255(black)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

grayscale = 255
x_train /= grayscale
x_val /= grayscale
x_test /= grayscale

#now using 
#using tf.keras.utils.to_categorical turns
# a numpy array (or) a vector which has integers that represent different categories, 
#can be converted into a numpy array (or) a matrix which has binary values and
# has columns equal to the number of categories in the data.
num_classes = 10 # 0-9 are the classes possible
y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_train = y_train.reshape(50000,10,1)
y_val = tf.keras.utils.to_categorical(y_val,num_classes)
y_test = tf.keras.utils.to_categorical(test_y, num_classes) #renaming again to match convention


# now creating the neural network
# i want one hidden layer in the shape (784,300,10)
size = (784,300,10)
#first, initialize weights
w1 = np.transpose(np.random.uniform(-1,1,size=size[:2])) #weights in layer 1
w2 = np.transpose(np.random.uniform(-1,1,size=size[1:3])) #weights in layer 2
#w3 = np.transpose(np.random.uniform(-1,1,size=size[2:]))
num_layers = 2 #number of layers
ndatapoints = len(x_train)#int(len(x_train)) #number of images
weights = [w1,w2]
inputs = [np.zeros((784,1)),np.zeros((300,1)),np.zeros((100,1))] #
#x_train = x_train[0:10001] #simplifying the problem for the time being

outputs = [np.zeros((300,1)),np.zeros((100,10)),np.zeros((10,1))] #intializing outputs and biases
biases = outputs
pre_activate = outputs
epochs = 2
LR = 0.2 #learning rate

# activation function
def Sigmoid(x):
    activated = 1/(1+np.exp(-x))
    return(activated)

def SigmoidDeriv(x):
    derivative = Sigmoid(x) * (1 - Sigmoid(x))
    return(derivative)



# def load_png(png):
#     img = misc.imread(png)
#     res = np.zeros(28*28)
#     for i, _ in enumerate(img):
#         for j, px in enumerate(img[i]):
#             res[28*i + j] = str(int(round(abs(px[1]-255)/255.)))
#     return res

def backprop(weights,pre_activate,output,error,x_train):
    #pre_activate = np.insert(pre_activate,0,x_train)
    x_train = x_train.reshape(784,1)
    lr = 0.01
    deltaE3 = error * SigmoidDeriv(pre_activate[2])
    weights[1] += lr * np.dot(output[2].T,deltaE3)
    
    deltaE1 = np.dot(weights[1].T,deltaE3) * SigmoidDeriv(pre_activate[0])
    weights[0] += lr * np.dot(deltaE1,x_train.T)
    return(weights)
        
        




correct = 0
for epoch in range(epochs): #loops thru for any amount of epochs
    #np.random.shuffle(x_train) #shuffling data each epoch so not always in same order
    start = time.time()
    for pic in range(ndatapoints): #loops thru each picture 
        
        for layer in range(num_layers): #loops thru each layer
            
            if layer == 0: #input layer
                pre_activate[layer] = np.matmul(weights[layer],x_train[pic].reshape(784,1))
                outputs[layer] = Sigmoid(pre_activate[layer]) #outputs are the activations
            else: #hidden and output layer
                inputs[layer] = outputs[layer-1]
                pre_activate[layer] = np.matmul(weights[layer],inputs[layer])
                outputs[layer] = Sigmoid(pre_activate[layer])
                #end of feedforward network
                
        error = y_train[pic] - outputs[layer]
                
        if np.argmax(y_train[pic]) == np.argmax(outputs[layer]): 
            correct += 1
        
        weights = backprop(weights,pre_activate,outputs,error,x_train[pic])
        

    endepoch = time.time() - start
    accuracy = (correct / (ndatapoints*(epoch+1)) )* 100 
    print("Epoch #",epoch,"with",np.round(accuracy,2),"% Accuracy")
    print("This epoch took",endepoch,"Seconds to run")
                    
                    

                    
                
                
        
            
    




