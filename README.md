# Diseased_Leaf_Classification_Pytorch

# Introduction
This project focused on building and evaluating a Convolutional Neural Network (CNN) for leaf disease classification using PyTorch. The custom CNN, incorporating batch normalization and max-pooling, was designed to handle both binary and multi-class classification tasks by adjusting the number of output neurons and using nn.CrossEntropyLoss(), which is suitable for both scenarios. This flexible approach, combined with training optimizations such as early stopping and model checkpointing, ensured robust performance and adaptability to varying classification needs.

## Dataset and Preprocessing Steps
The dataset consisted of images categorized into two classes: 'diseased leaf' and 'fresh leaf'. It was divided into training, validation, and test sets. Preprocessing involved resizing images to a uniform size and applying data augmentation techniques like random rotations, horizontal flips, and normalization using torchvision.transforms. These steps were crucial in improving model generalization and preventing overfitting by increasing data diversity.

## Model and Techniques Used
The CNN architecture included two convolutional layers with batch normalization and max-pooling to reduce feature map dimensions and control model complexity. The final fully connected layer was configured to output logits for the required number of classes, allowing the model to handle both binary and multi-class classification. nn.CrossEntropyLoss() was chosen over nn.BCELoss() for its capability to process raw logits and integer labels. Early stopping monitored validation loss to halt training when improvements ceased, while model checkpointing saved the best model based on validation performance, ensuring optimal results for test evaluation.

## Training and Evaluation Results
The model was trained over several epochs using the Adam optimizer, achieving high accuracy on the validation set. Early stopping effectively prevented overfitting by stopping training when no improvement in validation loss was observed for a specified number of epochs. The best-performing model was saved and evaluated on the test set, demonstrating strong classification performance with metrics such as accuracy, precision, recall, and confusion matrix, providing comprehensive insights into the model's strengths.

## Challenges Faced and Solutions Implemented
Balancing model complexity to avoid overfitting while ensuring adequate learning capacity was a significant challenge. This was mitigated through the use of batch normalization and dropout layers, which improved generalization. Additionally, fine-tuning the learning rate and implementing early stopping helped avoid over-training. Data scarcity was another challenge, addressed through data augmentation techniques that increased dataset diversity and enhanced the model's performance.

## Potential Applications and Domains
The classification CNN model developed can be adapted for various image classification tasks, such as detecting anomalies in medical imaging (e.g., X-rays, MRIs), quality control in manufacturing to identify defective products, and security systems for object or individual identification. The model's flexible architecture and  training techniques make it a valuable tool for any scenario involving image classification across different domains.
