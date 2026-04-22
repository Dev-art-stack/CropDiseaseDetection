A comprehensive full-stack solution for agricultural disease diagnosis, featuring **Explainable AI (Grad-CAM)** and **Bayesian Uncertainty Quantification (MC Dropout)**. This project bridges the gap between high-performance deep learning and user-centric mobile deployment.

## 🚀 Key Features

* **Robust Classification**: Diagnoses 9 distinct crop disease classes using a fine-tuned **MobileNetV3-Large** architecture.
* **Reliability Reporting**: Implements **Monte Carlo (MC) Dropout** with 30 stochastic forward passes to calculate predictive variance, allowing the system to signal when it is "unsure".
* **Explainable AI (XAI)**: Integrated **Grad-CAM** module highlights symptomatic regions on the leaf, providing visual justification for the model's prediction.
* **Leaf Verification**: Automated HSV-based masking prevents non-leaf images from being processed, ensuring system integrity.
* **Mobile-First Design**: A responsive Android application built with **Kotlin** and **Retrofit** for real-time field use.

## 📂 Project Structure

```text
.
├── backend/                # Flask API & Deep Learning Logic
│   ├── server.py           # REST API entry point and MC Dropout controller
│   ├── train.py            # Model training and fine-tuning script
│   ├── gradcam.py          # XAI heatmap generation logic
│   └── ...
├── frontend/               # Android Mobile Application (Kotlin)
│   ├── app/src/main/java/  # Kotlin source code (MainActivity, ApiService, etc.)
│   ├── app/src/main/res/   # UI Layouts (XML) and assets
│   └── ...
└── README.md               # Project documentation
