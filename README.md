# ECG Denoising and Anomaly Detection Using Signal Processing and Unsupervised Learning

## Project Overview
This project implements an end-to-end pipeline for preprocessing electrocardiogram (ECG) signals and detecting anomalies (e.g., irregular heartbeats or artifacts) using advanced signal processing and unsupervised learning. Designed for wearable health monitoring systems, the pipeline emphasizes ECG denoising (baseline wander removal and wavelet denoising) and anomaly detection with an Isolation Forest algorithm. This project showcases advanced ECG preprocessing skills, complementing other portfolio projects (e.g., ECG arrhythmia classification and HRV stress detection) by focusing on unsupervised methods and signal cleaning, strengthening the author’s PhD profile in Biomedical Signal Processing.

## Objectives
- Generate synthetic ECG data with simulated noise and anomalies to mimic real-world wearable device signals.
- Preprocess ECG signals to remove baseline wander and segment into windows.
- Apply wavelet-based denoising to eliminate high-frequency noise.
- Detect anomalies using unsupervised learning (Isolation Forest).
- Visualize results and save outputs for reproducibility.
- Organize the codebase in a modular, professional structure for GitHub.

## Dataset
- **Source**: Synthetic ECG data generated using the `neurokit2` library.
- **Specifications**:
  - Duration: 10 minutes (600 seconds).
  - Sampling Rate: 360 Hz (standard for ECG databases like MIT-BIH), producing 216,000 samples.
  - Heart Rate: 60 beats per minute (BPM) with 5% noise for realism.
  - Anomalies: 5% of samples are perturbed with Gaussian noise to simulate irregular heartbeats or artifacts.
- **Labels**: Binary anomaly labels (0: normal, 1: anomaly) generated for evaluation (not used in training, as the model is unsupervised).
- **Rationale**: Synthetic data ensures accessibility and reproducibility, allowing the pipeline to be tested without restricted datasets (e.g., MIT-BIH). The pipeline is extensible to real ECG data.

## Methodology
The project follows a six-step pipeline, implemented in Python and optimized for Google Colab. Each step is encapsulated in a modular function, stored in a separate `.py` file within the `src/` folder.

### Step 1: Install Dependencies
- **Purpose**: Sets up the Python environment with required libraries.
- **Libraries**:
  - `wfdb`: For potential real ECG dataset loading (e.g., MIT-BIH).
  - `numpy`, `pandas`: For numerical computations and data manipulation.
  - `scikit-learn`: For feature scaling and Isolation Forest algorithm.
  - `neurokit2`: For ECG simulation, cleaning, and R-peak detection.
  - `matplotlib`: For visualizing anomaly detection results.
  - `pywt`: For wavelet-based denoising.
- **Implementation**: Uses `pip install` to ensure libraries are available in Colab and imports them.


### Step 2: Generate Synthetic Data (`load_synthetic_data`)
- **Purpose**: Creates a synthetic ECG signal with added noise and anomalies to simulate real-world wearable data.
- **Function**: `load_synthetic_data(duration=600, sampling_rate=360, anomaly_rate=0.05)`
  - Generates a 10-minute ECG signal using `nk.ecg_simulate` with 60 BPM and 5% noise.
  - Adds synthetic anomalies by perturbing 5% of samples with Gaussian noise (mean=0, std=0.5).
  - Creates binary labels (0: normal, 1: anomaly) for evaluation (not used in training).
  - Prints the shapes of the ECG signal and labels, and the anomaly distribution.
