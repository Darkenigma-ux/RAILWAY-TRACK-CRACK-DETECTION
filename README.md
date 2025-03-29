## Railway Track Fault Detection using Machine Learning

# Project Overview

This project implements a machine learning-based approach to detect faults and cracks in railway tracks using Decision Trees. It uses real-time sensor data, including vibration levels, strain gauges, ultrasonic readings, and geolocation data (latitude and longitude). The model predicts track faults and visualizes them on an interactive map.

Dataset Information

The dataset consists of 20,000 samples with the following features:

Vibration_Level: Measures track vibration.

Strain_Gauge: Measures strain on the track.

Ultrasonic_Reading: Detects internal cracks.

Crack_Severity: Represents crack intensity.

Latitude & Longitude: Geolocation of track sections.

Crack_Present: Indicates if a crack is present (1) or not (0).

Implementation Steps

1. Load and Validate Dataset

import pandas as pd

df = pd.read_csv("railway_track_fault_data_with_location.csv")

Ensure column names match expected structure
expected_columns = ["Vibration_Level", "Strain_Gauge", "Ultrasonic_Reading", "Crack_Severity", "Latitude", "Longitude", "Crack_Present"]
for col in expected_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in dataset! Check your CSV file.")
        exit()

Explanation:

Reads the dataset using pandas.read_csv().

Checks if all required columns exist. If not, it exits with an error message.

2. Define Features and Target Variable

X = df[["Vibration_Level", "Strain_Gauge", "Ultrasonic_Reading", "Crack_Severity", "Latitude", "Longitude"]].copy()
y = df["Crack_Present"].copy()

Explanation:

X (features) includes sensor data and location.

y (target) represents whether a crack is present (1) or not (0).

3. Split Dataset into Training and Testing Sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Explanation:

Splits 80% of data for training and 20% for testing.

Uses stratify=y to maintain the proportion of faulty vs. non-faulty tracks.

4. Train the Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

Explanation:

Uses DecisionTreeClassifier with a depth of 5.

Trains the model using model.fit().

5. Make Predictions

y_pred = model.predict(X_test)

Explanation:

Predicts faults on test data.

6. Evaluate Model Performance

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

Explanation:

Calculates accuracy.

Generates a classification report with Precision, Recall, and F1-score.

7. Compute Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

Explanation:

Computes confusion matrix to analyze false positives and negatives.

Uses seaborn to plot a heatmap.

8. Visualize Decision Tree

from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=["No Crack", "Crack"], filled=True)
plt.show()

Explanation:

Displays the trained decision tree structure.

9. Fault Detection Map

import folium
from folium.plugins import HeatMap

fault_data = df[df["Crack_Present"] == 1][["Latitude", "Longitude"]]
map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
railway_map = folium.Map(location=map_center, zoom_start=12)
HeatMap(fault_data.values, radius=10, blur=15, max_zoom=1).add_to(railway_map)
railway_map.save("fault_heatmap.html")
print("Heatmap saved as fault_heatmap.html.")

Explanation:

Uses folium to create an interactive railway fault map.

Highlights faulty locations using a HeatMap.

Results

Accuracy: The model achieved approximately X% accuracy (update with actual results).

Confusion Matrix: Shows correct and incorrect predictions.

Decision Tree Visualization: Helps understand the model’s decision-making.

Heatmap: Displays geolocations of detected faults.

Conclusion

This project provides an automated fault detection system for railway tracks. It helps railway authorities identify and fix cracks early, reducing derailment risks and improving passenger safety.

Future Enhancements

Integrate real-time data collection from IoT sensors.

Enhance prediction accuracy using ensemble models.

Develop a mobile app to notify railway maintenance teams.

Installation Guide

Install required libraries:

pip install pandas numpy scikit-learn matplotlib seaborn folium

Run the Python script.

View fault_heatmap.html for detected crack locations.
