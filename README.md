🧠 Face Recognition using ORL (AT&T) Dataset with SVM


📌 Project Overview

This project implements a Face Recognition system using the ORL (AT&T) Face Dataset, a classic benchmark dataset in Digital Image Processing (DIP) and Machine Learning.

The system:

Loads and preprocesses grayscale face images

Extracts pixel-level features

Trains a Support Vector Machine (SVM) classifier

Evaluates recognition accuracy on unseen test data

This notebook demonstrates a simple yet effective classical ML pipeline for face recognition without using deep learning.

🗂 Dataset Information

ORL (AT&T) Face Dataset:

40 subjects

10 images per subject (400 images total)

Image size: 64 × 64 (grayscale)

Variations in:

Facial expressions

Lighting conditions

Facial details (glasses / no glasses)

📦 Dataset Source: Kaggle (Downloaded using kagglehub)

⚙️ Technologies Used
Component	Description
Python	Programming language
OpenCV (cv2)	Image loading and preprocessing
NumPy	Numerical computations
Scikit-learn	SVM classifier, scaling, train-test split
KaggleHub	Dataset download
Jupyter Notebook	Interactive experimentation
🔍 Methodology
1️⃣ Data Loading

Images are read from subject-wise folders

Each image is converted to a NumPy array

2️⃣ Preprocessing

Normalization of pixel values (0–1 range)

Flattening of images from 64×64 → 4096 features

3️⃣ Train–Test Split

70% training data

30% testing data

Stratified split to preserve class balance

4️⃣ Model Training

Support Vector Machine (SVM) with:

RBF Kernel

Feature scaling using StandardScaler

Implemented using an sklearn pipeline

5️⃣ Evaluation

Training accuracy

Testing accuracy

Model performance analysis

📊 Results

Achieves high classification accuracy on the ORL dataset

Demonstrates that classical ML techniques can perform well on structured face datasets

Suitable for learning feature-based face recognition

(Exact accuracy depends on random state and hyperparameters)

▶️ How to Run the Project
🔧 Prerequisites

Make sure you have Python 3.8+ installed.

Install required libraries:

pip install numpy opencv-python scikit-learn kagglehub

▶️ Execution

Open the Jupyter Notebook

Run all cells sequentially

Dataset will be downloaded automatically

Model will train and evaluate

📁 Project Structure
├── Shravani_P_Deshpande_DIP_8.ipynb
├── README.md

🎯 Learning Outcomes

Understanding face recognition fundamentals

Practical experience with:

Image preprocessing

Feature extraction

SVM classification

Applying Machine Learning to real image datasets

🚀 Future Enhancements

Apply PCA (Eigenfaces) for dimensionality reduction

Compare with KNN and Logistic Regression

Implement CNN-based deep learning model

Add real-time face recognition using webcam

👩‍💻 Author

Shravani P. Deshpande
📚 Digital Image Processing – Academic Project
🎓 Computer Engineering

⭐ Acknowledgements

AT&T Laboratories Cambridge

Kaggle Dataset Contributors

Scikit-learn & OpenCV communities
