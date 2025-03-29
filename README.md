Railway Track Fault Detection using Machine Learning
Project Overview
This project focuses on detecting faults and cracks in railway tracks using Machine Learning (ML) techniques, specifically a Decision Tree Classifier. By analyzing data from railway sensors such as accelerometers, strain gauges, and ultrasonic sensors, the model predicts whether a railway track segment has faults and provides the exact geographical locations of detected faults.

Technologies & Tools Used
Programming Language: Python

Libraries Used:

pandas – Data handling

numpy – Numerical computations

sklearn – ML model training and evaluation

folium – Interactive maps for fault visualization

matplotlib & seaborn – Data visualization (confusion matrix, accuracy graphs)

Dataset Creation
To train the model, a dataset of 20,000 samples was created, considering various track conditions:

Sensor Data: Readings from railway-mounted sensors such as accelerometers, strain gauges, and ultrasonic sensors.

Image Data: Drone and CCTV camera images analyzed for cracks.

Manual Inspection Reports: Data from railway authorities on faulty tracks.

Each data point includes:

Vibration Level

Strain Gauge Readings

Ultrasonic Readings

Crack Severity

Latitude & Longitude (Fault Location)

Crack Presence (Yes/No)

Machine Learning Model: Decision Tree Classifier
A supervised learning approach using a Decision Tree Classifier was implemented for fault detection.

Why Decision Tree?
Easy to interpret and visualize

Works well with both numerical and categorical data

Efficient in handling non-linear relationships in track conditions

Model Training Steps
Data Preprocessing:

Checked for missing values and outliers

Normalized sensor readings

Feature Selection:

Selected "Vibration_Level", "Strain_Gauge", "Ultrasonic_Reading", "Crack_Severity", "Latitude", "Longitude"

"Crack_Present" as the target variable

Train-Test Split:

80% of data used for training, 20% for testing

stratify=y used to maintain class balance

Model Training & Prediction:

Decision Tree trained with max_depth=5 for optimal performance

Predictions made on the test set

Model Evaluation:

Accuracy Score: Measures overall performance

Precision, Recall, F1-Score: Determines how well cracks are classified

Confusion Matrix: Shows correct vs incorrect classifications

Fault Location Mapping & Visualization
To improve railway track monitoring, interactive maps were generated using Folium:

Fault Detection Map

Plots actual fault locations on a railway track map

Uses latitude and longitude to pinpoint high-risk zones

Saves as fault_detection_map.html for easy access

Heatmap for Crack Severity

Highlights most affected areas based on crack severity

Uses Folium’s HeatMap layer for better visualization

Saves as fault_heatmap.html

Graphical Analysis & Insights
To analyze model performance, various plots were generated:

Decision Tree Visualization – Displays model logic for fault prediction

Confusion Matrix – Evaluates how well the model detects faults

Accuracy & Loss Graphs – Shows model training effectiveness

Impact & Future Scope
This ML-powered system helps railway authorities by:

Early Detection: Identifies cracks before accidents occur

Real-time Monitoring: Tracks faults using IoT sensors

Improved Maintenance: Reduces downtime by predicting risky locations

Future Enhancements:
Integration of Deep Learning for better fault detection

Use of CNN models to analyze track images from drones

Deployment as a real-time IoT system for live monitoring

Conclusion
This project successfully implemented Machine Learning for railway track fault detection using Decision Trees. By analyzing sensor data, inspection reports, and images, the model detects cracks with high accuracy. Interactive maps provide real-time insights for railway safety, ensuring safer train travel.
