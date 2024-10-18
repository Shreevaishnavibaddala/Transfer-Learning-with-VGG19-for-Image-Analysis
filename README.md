# Vgg19-based-transfer-learning-for-image-classification
Leverage Pytorch API, use VGG19 pretrained on ImageNet. Modify
the classification layer of VGG19 to 5 output nodes. Retrain only the
classification layer.
a. Observed the classification accuracy for training, validation, and test data.
Present the accuracy with confusion matrix and compare with that of the
best performance obtained in Task 2.
b. Considered one image from the training set of each of the classes (same
images as in Task 2 ). Pass each image to CNN. Found out a neuron in the
last convolutional layer (for each image) that is maximally activated. Trace
back to the patch in the image which causes these neurons to fire.
Visualize the patches in each of the images which maximally activate that
neuron.
Played around different hyper-parameters to improve the overall
accuracy of the architecture.
