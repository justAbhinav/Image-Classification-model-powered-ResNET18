# DL Assignment Image Classification with ResNet-18

This project aims to build an image classification model leveraging the ResNet-18 architecture. The model is trained and validated using a dataset consisting of images of monkeys, which are categorized into multiple classes. The following steps outline the core components of the project:

## Objectives
- Achieve high classification accuracy using the ResNet-18 model.
- Ensure robust data processing, augmentation, and model evaluation throughout the workflow.

## Steps

1. **Data Preparation**: The dataset is loaded and preprocessed using PyTorch's `torchvision` library, ensuring compatibility with the ResNet-18 model.

2. **Data Augmentation**: Various image transformations, such as rotation, flipping, and resizing, are applied to the training data to enhance the model's ability to generalize to unseen data.

3. **Normalization**: The training dataset's mean and standard deviation are calculated to normalize the images, ensuring the model receives data with consistent statistical properties.

4. **Model Definition**: The ResNet-18 architecture is used as the base model. The final fully connected layer is modified to output the correct number of classes for the dataset.

5. **Training**: The model is trained using mini-batch gradient descent. During training, the loss and accuracy are monitored to evaluate the model's performance over time.

6. **Evaluation**: After training, the model is evaluated on the test dataset to measure its classification accuracy and determine its effectiveness in predicting the correct labels.

7. **Checkpointing**: The best performing model, based on validation accuracy, is saved at each epoch to ensure that the model with the highest accuracy is preserved.

8. **Loading and Saving Models**: The final best model is loaded from the saved checkpoint and stored for future inference or deployment.

## Outcomes
- **Model Performance**: The ResNet-18 model was able to achieve high accuracy on the test dataset, demonstrating its effectiveness in classifying images into their respective categories.
- **Training Process**: The model was trained using mini-batch gradient descent, with careful monitoring of the loss and accuracy to ensure consistent improvements.
- **Checkpointing**: We utilized checkpointing to save the best performing model based on validation accuracy, ensuring that the highest-performing model was used for final evaluation.
- **Final Model**: The best model was saved and is now ready for deployment or further fine-tuning with different datasets or architectures.

## Future Work
- Experiment with deeper models such as ResNet-50 or ResNet-101.
- Apply more advanced data augmentation techniques.
- Fine-tune the learning rate and other hyperparameters to optimize performance further.

Overall, the project demonstrated the power of deep learning for image classification tasks and highlighted the importance of data preprocessing, model selection, and evaluation in achieving high-quality results.
