in progress........
in progress........
in progress........





## 📌 Project Overview
This project applies **Convolutional Neural Networks (CNNs)** to satellite imagery to classify and assess infrastructural damage caused by natural disasters (hurricanes, wildfires, floods, earthquakes). 

Using a **change detection framework**, the models compare pre- and post-event imagery to evaluate damage levels on a 4-tier scale (no damage, minor, major, destroyed).

### Key Features & Research Questions:
* **Architectures:** Comparison between a custom CNN trained from scratch and a pre-trained backbone optimized via Transfer Learning.
* **Siamese Networks:** Implementation of a two-branch architecture to process co-registered image patches.
* **Cross-Disaster Generalization:** Evaluating model performance on *unseen* disaster types (e.g., training on floods/hurricanes and testing on wildfires) to simulate real-world emergency deployment.
* **Socioeconomic Impact Analysis:** Overlaying predicted damage maps with demographic data to study recovery equity.

## 📊 Dataset
The primary data source is the **xBD / xView2 dataset**, featuring high-resolution imagery pairs with building footprints. Additional satellite sources include Maxar Open Data and Sentinel-2.

## 🛠️ Tech Stack (Expected)
* **Languages:** Python
* **Frameworks:** PyTorch / PyTorch Lightning (or TensorFlow/Keras)
* **Libraries:** OpenCV, Rasterio, Albumentations, NumPy, Pandas, Matplotlib
