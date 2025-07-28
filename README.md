ECG Denoising and Anomaly Detection Using Signal Processing and Unsupervised Learning
Project Overview
This project implements an end-to-end pipeline for preprocessing electrocardiogram (ECG) signals and detecting anomalies (e.g., irregular heartbeats or artifacts) using advanced signal processing and unsupervised learning. Designed for wearable health monitoring systems, the pipeline emphasizes ECG denoising (baseline wander removal and wavelet denoising) and anomaly detection with an Isolation Forest algorithm. This project showcases advanced ECG preprocessing skills, complementing other portfolio projects (e.g., ECG arrhythmia classification and HRV stress detection) by focusing on unsupervised methods and signal cleaning, strengthening the author’s PhD profile in Biomedical Signal Processing.
Objectives

Generate synthetic ECG data with simulated noise and anomalies to mimic real-world wearable device signals.
Preprocess ECG signals to remove baseline wander and segment into windows.
Apply wavelet-based denoising to eliminate high-frequency noise.
Detect anomalies using unsupervised learning (Isolation Forest).
Visualize results and save outputs for reproducibility.
Organize the codebase in a modular, professional structure for GitHub.

Dataset

Source: Synthetic ECG data generated using the neurokit2 library.
Specifications:
Duration: 10 minutes (600 seconds).
Sampling Rate: 360 Hz (standard for ECG databases like MIT-BIH), producing 216,000 samples.
Heart Rate: 60 beats per minute (BPM) with 5% noise for realism.
Anomalies: 5% of samples are perturbed with Gaussian noise to simulate irregular heartbeats or artifacts.


Labels: Binary anomaly labels (0: normal, 1: anomaly) generated for evaluation (not used in training, as the model is unsupervised).
Rationale: Synthetic data ensures accessibility and reproducibility, allowing the pipeline to be tested without restricted datasets (e.g., MIT-BIH). The pipeline is extensible to real ECG data.

Methodology
The project follows a six-step pipeline, implemented in Python and optimized for Google Colab. Each step is encapsulated in a modular function, stored in a separate .py file within the src/ folder.
Step 1: Install Dependencies

Purpose: Sets up the Python environment with required libraries.
Libraries:
wfdb: For potential real ECG dataset loading (e.g., MIT-BIH).
numpy, pandas: For numerical computations and data manipulation.
scikit-learn: For feature scaling and Isolation Forest algorithm.
neurokit2: For ECG simulation, cleaning, and R-peak detection.
matplotlib: For visualizing anomaly detection results.
pywt: For wavelet-based denoising.


Implementation: Uses pip install to ensure libraries are available in Colab and imports them.
Output: Confirmation of successful installation (e.g., Successfully installed ...).

Step 2: Generate Synthetic Data (load_synthetic_data)

Purpose: Creates a synthetic ECG signal with added noise and anomalies to simulate real-world wearable data.
Function: load_synthetic_data(duration=600, sampling_rate=360, anomaly_rate=0.05)
Generates a 10-minute ECG signal using nk.ecg_simulate with 60 BPM and 5% noise.
Adds synthetic anomalies by perturbing 5% of samples with Gaussian noise (mean=0, std=0.5).
Creates binary labels (0: normal, 1: anomaly) for evaluation (not used in training).
Prints the shapes of the ECG signal and labels, and the anomaly distribution.


Output:ECG shape: (216000,)
Labels shape: (216000,)
Anomaly distribution: {0: ~205200, 1: ~10800}


Rationale: Synthetic anomalies mimic irregular heartbeats or artifacts, enabling testing of the anomaly detection pipeline.

Step 3: Preprocess ECG (preprocess_ecg)

Purpose: Removes baseline wander and segments the ECG signal into overlapping windows.
Function: preprocess_ecg(ecg_signal, sampling_rate=360, window_size=3600)
Removes baseline wander using 5th-degree polynomial fitting to model low-frequency drift.
Segments the ECG into 10-second windows (3600 samples at 360 Hz) with 50% overlap (step size: window_size // 2) to capture temporal dynamics.
Returns an array of segments and the baseline-corrected ECG signal.


Output:Number of segments: ~119
Segment shape: (~119, 3600)


Rationale: Baseline wander removal is critical for ECG preprocessing, and overlapping windows ensure robust feature extraction.

Step 4: Denoise ECG (denoise_ecg)

Purpose: Applies wavelet denoising to remove high-frequency noise from ECG segments.
Function: denoise_ecg(segments, wavelet='db6', level=4)
Uses pywt.wavedec with the Daubechies 6 (db6) wavelet and 4 decomposition levels.
Applies soft thresholding to high-frequency coefficients based on the universal threshold (σ * sqrt(2 * log(N))).
Reconstructs denoised segments using pywt.waverec, ensuring output length matches input.
Returns an array of denoised segments.


Output:Denoised segments shape: (~119, 3600)


Rationale: Wavelet denoising effectively removes noise while preserving ECG morphology, critical for accurate anomaly detection.

Step 5: Detect Anomalies (detect_anomalies)

Purpose: Identifies anomalies in ECG segments using an Isolation Forest algorithm.
Function: detect_anomalies(denoised_segments, contamination=0.05)
Extracts simple statistical features (mean, standard deviation, max, min) from each segment.
Normalizes features using StandardScaler.
Trains an Isolation Forest model with a contamination rate of 5% to detect anomalies.
Converts model predictions (-1: anomaly, 1: normal) to binary labels (1: anomaly, 0: normal).
Prints the anomaly distribution.


Output:Anomaly distribution: [~113 ~6]  # ~5% anomalies


Rationale: Isolation Forest is efficient for unsupervised anomaly detection, suitable for wearable devices with limited labeled data.

Step 6: Visualize and Save Results (visualize_and_save)

Purpose: Visualizes the denoised ECG signal with detected anomalies and saves results.
Function: visualize_and_save(ecg_no_baseline, anomaly_labels, features, segments)
Reconstructs a full anomaly signal by marking windows with detected anomalies.
Plots the denoised ECG with red markers for anomaly regions.
Saves the plot as results/anomaly_plot.png.
Saves anomaly labels to results/anomaly_scores.txt.
Downloads files in Colab using files.download.


Output: A plot showing the ECG signal with red dots indicating anomalies, plus saved files.
Rationale: Visualization aids interpretation, and saved files ensure reproducibility.

Results

Metric: The Isolation Forest detects ~5% of segments as anomalies, aligning with the synthetic anomaly rate.
Output Files:
results/anomaly_plot.png: Plot of the ECG signal with detected anomalies.
results/anomaly_scores.txt: Array of anomaly labels for each segment.


Insights: The pipeline effectively removes noise and identifies anomalies, demonstrating robust preprocessing and unsupervised learning capabilities.

Project Structure
ECG-Anomaly-Detection/
├── notebooks/
│   └── ecg_anomaly_detection_notebook.ipynb
├── src/
│   ├── setup.py
│   ├── load_data.py
│   ├── preprocess_data.py
│   ├── denoise_ecg.py
│   ├── detect_anomalies.py
│   ├── visualize_results.py
├── results/
│   ├── anomaly_scores.txt
│   ├── anomaly_plot.png
└── README.md


notebooks/: Contains the Colab notebook (ecg_anomaly_detection_notebook.ipynb) with the full pipeline.
src/: Contains Python scripts for each step.
results/: Stores output files (plot and anomaly scores).
README.md: This documentation file.

How to Run
Option 1: Google Colab

Open Google Colab (https://colab.research.google.com).
Create a new notebook and copy the consolidated code into a single cell.
Run the cell (Shift+Enter) to execute the pipeline.
Download anomaly_plot.png and anomaly_scores.txt using the provided files.download commands.
Export the notebook as ecg_anomaly_detection_notebook.ipynb (File > Download > .ipynb).

Option 2: Local Environment

Clone the repository:git clone https://github.com/your-username/ECG-Anomaly-Detection
cd ECG-Anomaly-Detection


Install dependencies:pip install wfdb numpy pandas scikit-learn neurokit2 matplotlib pywt


Run the scripts in order:python src/setup.py
python src/load_data.py
python src/preprocess_data.py
python src/denoise_ecg.py
python src/detect_anomalies.py
python src/visualize_results.py


Alternatively, run the notebook:jupyter notebook notebooks/ecg_anomaly_detection_notebook.ipynb



Notes

Requires Python 3.7+ for compatibility with neurokit2 and pywt.
Synthetic data ensures reproducibility; extend to real datasets (e.g., MIT-BIH) for practical use.
Adjust window_size or contamination parameters to fine-tune results.

Setup GitHub Repository

Create a local folder ECG-Anomaly-Detection.
Create subfolders: src/, notebooks/, results/.
Save the .py files in src/ (split the consolidated code into individual functions).
Save the Colab notebook in notebooks/.
Move anomaly_plot.png and anomaly_scores.txt to results/.
Save this README as README.md in the root folder.
Initialize a Git repository:cd ECG-Anomaly-Detection
git init
git add .
git commit -m "Initial commit of ECG Anomaly Detection project"


Create a GitHub repository named ECG-Anomaly-Detection:
Description: "ECG Denoising and Anomaly Detection Using Signal Processing and Unsupervised Learning"
Uncheck "Initialize with a README".


Push to GitHub:git remote add origin https://github.com/your-username/ECG-Anomaly-Detection.git
git branch -M main
git push -u origin main



Troubleshooting

Few anomalies detected: Adjust the contamination parameter in detect_anomalies (e.g., increase to 0.1).
Wavelet denoising errors: Ensure pywt is installed (pip install pywt) and verify wavelet='db6' is supported.
Poor anomaly detection: Check segment quality (print(denoised_segments.shape)). Increase window_size (e.g., to 7200) for more robust features.
Import errors: Verify library installations (pip install neurokit2 pywt).
Other issues: Share error messages for targeted assistance.

Future Work

Real Datasets: Apply the pipeline to real ECG datasets (e.g., MIT-BIH Arrhythmia Database).
Advanced Features: Incorporate additional features (e.g., R-peak intervals, morphological features) for anomaly detection.
Alternative Models: Explore other unsupervised methods (e.g., Autoencoders, DBSCAN).
Real-Time Deployment: Optimize for embedded systems (e.g., Raspberry Pi) for wearable applications.
Validation: Compare detected anomalies with ground-truth labels from real datasets.
