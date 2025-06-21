🥔 Potato Leaf Disease Classification using CNN
🚀 Final Project – AI Professional Training
Presented by: Sher Alam Khan
Institute: InnoVista, DHA Phase 1, Rawalpindi/Islamabad
Cohort: InnoQuest Cohort-1

🎯 Project Objective
The goal of this project is to design and implement a deep learning-based image classification system using Convolutional Neural Networks (CNNs) that can accurately identify three different categories of potato leaf conditions:

Early Blight

Late Blight

Healthy

This system supports early disease detection in potatoes to help farmers manage crops efficiently and reduce losses.

🗂 Dataset Summary
A publicly available potato leaf disease dataset was used, containing labeled images categorized into three classes.

📊 Dataset Distribution
Class Name	Total Images	Training	Validation	Testing
Early Blight	1,000	700	200	100
Late Blight	1,000	700	200	100
Healthy	152	106	30	16
Total	2,152	1,506	430	216

The dataset was split in a 70:20:10 ratio for training, validation, and testing respectively.

🧠 Model Architecture
A Convolutional Neural Network (CNN) with 6 convolutional layers followed by pooling, a flattening layer, and fully connected dense layers was used to extract and classify image features.

Input Image Size: 256 x 256 pixels

Number of Channels: 3 (RGB)

Output Classes: 3

Total Trainable Parameters: ~184K

Activation Function: ReLU for hidden layers, Softmax for output layer

The model was compiled using the Adam optimizer and trained with Sparse Categorical Crossentropy loss over 26 epochs.

🏋️ Training Progress
The training process showed steady improvements in model accuracy and loss. By the end of the 26th epoch:

Training Accuracy: ~99.07%

Validation Accuracy: ~97.44%

Loss: Steadily decreased across epochs

The model generalized well with minimal overfitting observed.

✅ Validation Evaluation
Evaluation was performed on the 20% validation dataset.

Accuracy: 96.98%

Precision: 97.16%

Recall: 96.98%

F1-Score: 96.98%

Class-wise Performance:
Class	Precision	Recall	F1-Score
Early Blight	1.00	0.94	0.97
Late Blight	0.94	1.00	0.97
Healthy	1.00	0.93	0.97

🧪 Test Evaluation
Final evaluation was conducted on the test set (10% of data).

Accuracy: 99.07%

Precision: 99.10%

Recall: 99.07%

F1-Score: 99.08%

Class-wise Performance:
Class	Precision	Recall	F1-Score
Early Blight	1.00	0.99	0.99
Late Blight	0.99	0.99	0.99
Healthy	0.94	1.00	0.97

📊 Performance Comparison
Metric	Training	Test	Difference
Accuracy	99.07%	99.07%	0.00%
Precision	99.09%	99.10%	0.01%
Recall	99.07%	99.07%	0.00%
F1-Score	99.07%	99.08%	0.01%

✔️ Model is well-balanced with excellent generalization and no overfitting.

🖼️ Predictions on Test Images
The model was tested on individual test images, and predictions were visualized.

Each image shows:

Actual Class Label

Predicted Class Label

Prediction Confidence (in %)

Predictions matched the actual classes with high confidence (>97%), indicating excellent real-world reliability.

💡 Future Improvements
Data Augmentation: Increase healthy class samples through augmentation.

Transfer Learning: Use pre-trained CNNs like EfficientNet or ResNet to improve efficiency.

Deployment: Integrate into a Streamlit or Flask web app for real-time disease detection.

Mobile App: Convert into an Android/iOS app for use in agricultural fields.

📁 Project Structure
bash
Copy
Edit
/split_dataset
  ├── train/
  ├── validation/
  ├── test/
app.py
model.h5
notebook.ipynb
README.md
✅ Summary
This project successfully implemented a CNN-based image classification model that achieved 99.07% accuracy in detecting potato leaf diseases. The solution is robust, scalable, and suitable for real-world deployment to support precision agriculture and smart farming.
