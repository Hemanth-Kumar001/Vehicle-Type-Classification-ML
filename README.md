# Vehicle Type Classification using Machine Learning

A multi-class classification project that predicts a vehicle type from five numerical vehicle attributes. The workflow compares four scikit-learn classifiers and evaluates them on a stratified 80/20 train-test split.

## Dataset

File: Vehicle_Classification.csv  
Rows: 600  
Target: Vehicle_Type

Input columns:
- Vehicle_Length
- Vehicle_Width
- Vehicle_Weight
- Engine_Capacity
- Mileage

The target dataset contains four vehicle classes. The uploaded CSV has no missing values.

## Algorithms

- Decision Tree Classifier
- K-Nearest Neighbors (k=5)
- Logistic Regression
- Support Vector Classifier (SVC)

Median imputation is applied in each pipeline. Standard scaling is included for k-NN, Logistic Regression, and SVM.

## Evaluation

The script reports accuracy, macro precision, macro recall, and macro F1-score. It also generates a confusion matrix for each model and saves a sorted model comparison table.

## Run locally

Install dependencies with: python -m pip install -r requirements.txt

Run: python vehicle_classification.py

Generated output files are saved beside the script:
- vehicle_model_comparison.csv
- vehicle_confusion_*.png

## Notes

- The script uses random_state=42 for reproducible splitting.
- No performance values are claimed here; run the script to generate results in your environment.
- Dataset source and redistribution terms were not provided with the uploaded file. Confirm the source/license before redistributing the CSV elsewhere.
