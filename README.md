# Wildfire-Detection-from-Satellite-Images
This project leverages Convolutional Neural Networks (CNN) to detect wildfires from satellite images. By analyzing these images, the model predicts whether a given image shows a wildfire or not, using a web interface built with Streamlit. This tool is aimed at helping authorities detect wildfires early and monitor their spread, providing crucial information for timely responses.

Project Overview
The Wildfire Detection from Satellite Images project is an image classification model built with TensorFlow/Keras and deployed through a Streamlit UI. The model is trained to differentiate between images containing wildfire and those without, based on visual patterns observed in satellite images. This solution is designed to serve as an early detection and monitoring tool, assisting wildfire management efforts.

Importance of the Project
Wildfires have devastating effects on ecosystems, human communities, air quality, and climate change. With increased incidents of wildfires globally, a proactive approach to detect and monitor these fires is essential. This project highlights how satellite imagery and machine learning can automate and enhance wildfire detection, enabling quicker and more informed responses to fire threats.

Real-World Use Cases
Early Detection of Wildfires: Allows for rapid detection of fire outbreaks, providing first responders with early alerts to prevent fires from spreading uncontrollably.
Monitoring Fire Spread: Real-time predictions can be used to monitor ongoing fires, giving insights into affected areas and helping allocate resources to critical locations.
Risk Assessment: Identifies high-risk areas and trends, supporting planning and risk mitigation efforts.
Climate Research and Environmental Impact Studies: Helps researchers understand the environmental impact of wildfires on carbon emissions and ecosystems.

Data Description
The dataset used for training and testing this model consists of satellite images classified into two categories:

Fire: Images with visible signs of wildfire.
No Fire: Images with no signs of wildfire.

Data can be sourced from platforms like Kaggle or Earth observation datasets, such as those provided by NASA or Sentinel-2. Ensure that the dataset is structured as follows:

dataset/
│
├── train/
│   ├── fire/
│   └── nofire/
│
└── test/
    ├── fire/
    └── nofire/

Installation
Requirements
To set up and run the project, you will need:

Python 3.9
Required libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Streamlit, Pillow


How to Use the Project
Start the Streamlit app. You will see an option to upload an image.
Upload a satellite image to test for fire presence.
The model will preprocess the image and output a prediction along with a probability score indicating the likelihood of fire.


Model Details
Architecture: A Convolutional Neural Network (CNN) optimized for binary classification.
Preprocessing: Images are resized to (256, 256), reshaped, and normalized to improve model performance.
Prediction Interpretation: Uses a threshold of 0.5 to determine fire (probability < 0.5) or no fire (probability >= 0.5).

Acknowledgments
This project was inspired by the growing need for wildfire monitoring and management systems.
Special thanks to Kaggle and various open-source satellite image datasets for providing accessible data.




