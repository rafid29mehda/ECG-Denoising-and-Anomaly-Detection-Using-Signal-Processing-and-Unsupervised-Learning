# ECG Denoising and Anomaly Detection Using Unsupervised Learning

## Project Overview
This project implements an end-to-end pipeline for preprocessing electrocardiogram (ECG) signals and detecting anomalies (e.g., irregular heartbeats or artifacts) using advanced signal processing and unsupervised learning. Designed for wearable health monitoring systems, the pipeline emphasizes ECG denoising (baseline wander removal and wavelet denoising) and anomaly detection with an Isolation Forest algorithm. 

## Objectives
- Generate synthetic ECG data with simulated noise and anomalies to mimic real-world wearable device signals.
- Preprocess ECG signals to remove baseline wander and segment into windows.
- Apply wavelet-based denoising to eliminate high-frequency noise.
- Detect anomalies using unsupervised learning (Isolation Forest).
- Visualize results and save outputs for reproducibility.


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

- **Rationale**: Synthetic anomalies mimic irregular heartbeats or artifacts, enabling testing of the anomaly detection pipeline.

### Step 3: Preprocess ECG (`preprocess_ecg`)
- **Purpose**: Removes baseline wander and segments the ECG signal into overlapping windows.
- **Function**: `preprocess_ecg(ecg_signal, sampling_rate=360, window_size=3600)`
- Removes baseline wander using 5th-degree polynomial fitting to model low-frequency drift.
- Segments the ECG into 10-second windows (3600 samples at 360 Hz) with 50% overlap (step size: `window_size // 2`) to capture temporal dynamics.
- Returns an array of segments and the baseline-corrected ECG signal.

- **Rationale**: Baseline wander removal is critical for ECG preprocessing, and overlapping windows ensure robust feature extraction.

### Step 4: Denoise ECG (`denoise_ecg`)
- **Purpose**: Applies wavelet denoising to remove high-frequency noise from ECG segments.
- **Function**: `denoise_ecg(segments, wavelet='db6', level=4)`
- Uses `pywt.wavedec` with the Daubechies 6 (`db6`) wavelet and 4 decomposition levels.
- Applies soft thresholding to high-frequency coefficients based on the universal threshold (`σ * sqrt(2 * log(N))`).
- Reconstructs denoised segments using `pywt.waverec`, ensuring output length matches input.
- Returns an array of denoised segments.

- **Rationale**: Wavelet denoising effectively removes noise while preserving ECG morphology, critical for accurate anomaly detection.

### Step 5: Detect Anomalies (`detect_anomalies`)
- **Purpose**: Identifies anomalies in ECG segments using an Isolation Forest algorithm.
- **Function**: `detect_anomalies(denoised_segments, contamination=0.05)`
- Extracts simple statistical features (mean, standard deviation, max, min) from each segment.
- Normalizes features using `StandardScaler`.
- Trains an Isolation Forest model with a contamination rate of 5% to detect anomalies.
- Converts model predictions (-1: anomaly, 1: normal) to binary labels (1: anomaly, 0: normal).
- Prints the anomaly distribution.

- **Rationale**: Isolation Forest is efficient for unsupervised anomaly detection, suitable for wearable devices with limited labeled data.

### Step 6: Visualize and Save Results (`visualize_and_save`)
- **Purpose**: Visualizes the denoised ECG signal with detected anomalies and saves results.
- **Function**: `visualize_and_save(ecg_no_baseline, anomaly_labels, features, segments)`
- Reconstructs a full anomaly signal by marking windows with detected anomalies.
- Plots the denoised ECG with red markers for anomaly regions.
- Saves the plot as `results/anomaly_plot.png`.
- Saves anomaly labels to `results/anomaly_scores.txt`.
- Downloads files in Colab using `files.download`.

- **Rationale**: Visualization aids interpretation, and saved files ensure reproducibility.
