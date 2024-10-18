# Transfer Learning with VGG19 for Image Analysis
#Leverage Pytorch API, use VGG19 pretrained on ImageNet. Modify
the classification layer of VGG19 to 5 output nodes. Retrain only the
classification layer.
#a. Observed the classification accuracy for training, validation, and test data.
Present the accuracy with confusion matrix and compare with that of the
best performance obtained in Task 2.
#b. Considered one image from the training set of each of the classes (same
images as in Task 2 ). Pass each image to CNN. Found out a neuron in the
last convolutional layer (for each image) that is maximally activated. Trace
back to the patch in the image which causes these neurons to fire.
#Visualize the patches in each of the images which maximally activate that
neuron.
#Played around different hyper-parameters to improve the overall
accuracy of the architecture.
# Image Classification using VGG19 - Caltech-101 Subset


## Dataset

- **Caltech-101 Subset**: Contains images from **5 classes**.
- Images vary in size and are resized to **224 x 224** for consistency.
- The dataset is split into **train**, **validation**, and **test** sets.

## Objectives

### Task 1: Fine-Tune VGG19 on the Given Dataset

1. **Leverage PyTorch API**: Use **VGG19** pretrained on ImageNet.
2. **Modify the Classification Layer**: Adjust the final layer of VGG19 to have 5 output nodes, corresponding to the number of classes in the dataset.
3. **Retrain the Classification Layer**: Freeze the convolutional layers and retrain only the classification layer.

#### Subtasks:

a. **Classification Accuracy**:
   - Measure and report the classification accuracy for **training**, **validation**, and **test sets**.
   - Present the results with a **confusion matrix** to visualize performance.
   - **Compare** these results with the best performance achieved in Task 2.

b. **Neuron Activation Visualization**:
   - Choose one image from the training set of each class.
   - Pass each image through the CNN and identify the **neuron in the last convolutional layer** that is maximally activated.
   - **Trace back** to the image patch that caused this neuron to fire.
   - **Visualize** the patches in each image that activate the neuron.

### Bonus: Hyperparameter Tuning

Explore different hyperparameters such as:
- Learning rate
- Optimizer
- Number of epochs
- Batch size

to improve the overall performance and accuracy of the model.

## Deliverables

1. **Confusion Matrix**: A visual representation of the classification accuracy across the 5 classes.
2. **Classification Accuracy**: Report the accuracy for:
   - Training set
   - Validation set
   - Test set
3. **Neuron Activation Patch Visualization**: Visualize the patches of each image that maximally activate specific neurons in the last convolutional layer.

## Tools & Libraries

- **PyTorch** for model implementation and training.
- **VGG19** pretrained on **ImageNet**.
- **Matplotlib** for visualizing the confusion matrix and neuron activations.
- **Grad-CAM** (or similar technique) for identifying image regions that activate neurons.

## Instructions

1. **Preprocess**: Resize all images to **224 x 224**.
2. **Model Setup**: Load **VGG19** and modify the final classification layer to 5 output nodes.
3. **Training**: Freeze convolutional layers, retrain the classification layer, and monitor performance on training, validation, and test data.
4. **Visualization**: For a selected image from each class, visualize the patches responsible for neuron activations in the last convolutional layer.
5. **Comparison**: Compare the accuracy with the best model from Task 2.

## Bonus Section

Tweak the following hyperparameters:
- **Learning Rate**
- **Optimizer** (SGD, Adam, etc.)
- **Number of Epochs**
- **Batch Size**

to further improve accuracy and model performance.

## How to Run the Project

1. Clone the repository and set up your environment with the necessary dependencies.
2. Prepare the dataset by resizing images to **224 x 224**.
3. Modify the VGG19 classification layer and freeze all other layers.
4. Train the model using the provided train-validation-test splits.
5. Use **Grad-CAM** to visualize important patches that activate neurons.
6. Compare the performance with the previous task.

## Results

- Accuracy for training, validation, and test sets.
- Confusion matrix visualizing performance across all classes.
- Visualizations of patches that maximally activate neurons in the last convolutional layer.

---

