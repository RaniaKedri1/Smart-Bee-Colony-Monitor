# 🐝 Smart Bee Colony Monitor

An AI-powered system to monitor beehive health by analyzing sound recordings. The project leverages machine learning to detect stress patterns in bee colonies through audio signal processing and classification.

## 🧠 Project Description

This project uses machine learning to:
- Analyze beehive sound recordings.
- Extract MFCC (Mel-frequency cepstral coefficients) audio features.
- Detect and classify the status of the queen bee and overall hive condition.
- Visualize feature distributions and audio-based stress patterns.
- Train and evaluate a predictive model to monitor colony health.

## 📦 Technologies Used

- **Programming Language**: Python  
- **Machine Learning**: Scikit-learn, TensorFlow, Keras  
- **Audio Processing**: Librosa, PyAudio  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Jupyter Notebook

## 📁 Dataset

Beehive sound dataset:  
[Kaggle - Beehive Sounds Dataset](https://www.kaggle.com/datasets/annajyang/beehive-sounds)

## 🚀 Key Features

- Audio preprocessing and MFCC feature extraction
- Feature correlation heatmaps and outlier detection
- Data cleaning, merging, and encoding
- Feature scaling using `StandardScaler`
- Classification using `RandomForestClassifier`
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation: classification report, confusion matrix, cross-validation
- Feature importance analysis
- Model serialization using `pickle`
- Output predictions and feature importance as CSV files

## 📝 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/Smart-Bee-Colony-Monitor.git
    cd Smart-Bee-Colony-Monitor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook in Jupyter:
    ```bash
    jupyter notebook
    ```

4. Train the model and explore the results in the notebook.

## 📂 Output Files

- `merged_bee_colony_data.csv`: Cleaned and merged dataset
- `encoded_bee_colony_data.csv`: Final encoded dataset used for training
- `train_data.csv`: Preprocessed training data
- `predictions.csv`: Model predictions
- `feature_importances.csv`: Feature importance ranking
- `hive_conditions_model.pkl`: Trained ML model

## 📊 Classification Target

The model predicts **Queen Status** with the following labels:
- `0`: Queen present or original queen  
- `1`: Queen not present  
- `2`: Queen present and rejected  
- `3`: Queen present and newly accepted

## 📌 Notes

- All audio files are processed to extract MFCCs.
- Model complexity was reduced with a controlled max depth to avoid overfitting.
- 10-fold cross-validation was used to evaluate model robustness.

