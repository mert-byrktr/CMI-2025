# Gesture Recognition Competition Solution

This repository outlines the two primary modeling approaches used in CMI 2025 Competition: a feature-rich CatBoost model and a custom multi-branch 1D-CNN. The aim was to use sensor data to classify body-focused repetitive behaviors (BFRBs) and other gestures.

I started the competition with the CatBoost algorithm. After a month of testing, I switched to CNN models, and the performance increase was truly surprising. However, CatBoost was the addition that provided the largest increase in the ensemble phase.

---

## Metric
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non-target class
The final score is the average of the binary F1 and the macro F1 scores.


## CatBoost and Pytabkit Model

### CatBoost and Pytabkit Model Features:
- Raw sensor measurements (ac_ and rot_ sensors)
- Euler features obtained from rotation signals
- Angular jerk features
- Rotation matrix features
- Sequence-based aggregations
- Normalized temporal position per sequence id
- Velocity, acceleration, jerk features
- Absolute rate of change compared to the previous and next signal
- Peak detection features, sequence-based peak count
- Sequence-based average velocity
- Average of acc_ and rot_ signals based on normalized temporal position, std, velocity mean, std, absolute mean (these features were calculated per sequence id for each time interval, dividing the measurement time by five).
- Zero-crossing features
- Correlation features (acc_x vs acc_y, etc.)
- Differences between each sensor pair for thm_ attributes
- Sequence-based aggregations
- Average of thm_ and tof_ signals, std, velocity mean, std, max, and absolute mean according to normalized temporal position (these attributes are taken per sequence_id for each time interval, dividing the measurement time by five).
- Sequence-based mean, std, and the difference between each of these for each measurement in the tof matrix.
- Average of corner pixels, average of center pixels, and corner-to-center ratio.
- Asymmetry ratio of right and left pixels.
- Average, std, max, and min values for time intervals greater than 0.6 in the normalized time of the matrix divided into four parts.
- Average difference between each of the four parts for a matrix divided into four parts.

### Mixup:
- 2x data by generating synthetic data from the existing measurements.

### Cross-validation Setup:
- 5-fold MultilabelStratifiedKFold, grouped per subject, stratified on adult_child, handedness, and sex.

---

## Custom Multi-Branch 1D-CNN Architecture

### Preprocessing:
- Identified subjects with wrong-side device placement by checking the average `acc_y` sign and flipped the y-axis and x-axis (`acc_y`, `acc_x`) measurements accordingly.
- For sequences with entirely missing quaternion data, the values were imputed with an identity quaternion (w=0, x=0, y=0, z=0) and a `rot_missing_flag` was added.

### IMU (acc_ & rot_ sensors) Features:
*   **Magnitude Features:** Calculated the vector magnitude for raw acceleration (`acc_mag`) and rotation (`rot_mag`).
*   **Gravity Removal:** Separated linear acceleration from gravity using quaternion rotations, generating `linear_acc_x`, `linear_acc_y`, `linear_acc_z`, and `linear_acc_mag`.
*   **Angular Velocity and Distance:** Derived `angular_vel_x/y/z` and `angular_distance` directly from the raw quaternion time series to capture rotational speed and change.
*   **Euler Features:** Converted quaternions to Euler angles (`roll`, `pitch`, `yaw`) and calculated their combined magnitude (`euler_mag`).
*   **Rotation Matrix Features:** Extracted all nine elements (r11 to r33) from the 3D rotation matrix derived from quaternions to capture detailed spatial orientation.
*   **Interaction Feature:** Created a `acc_angular_sync` feature by multiplying accelerometer and angular velocity magnitudes.

### Temporal and Rolling Window Features:
*   **Normalized Temporal Position:** Created a `normalized_position` feature from 0 to 1 for each sequence, along with cyclical sin/cos transformations (`position_sin`, `position_cos`) for both first and second harmonics.
*   **Velocity Features (First Derivatives):** Calculated the first-order difference (velocity) for all core IMU, Euler, linear acceleration, and rotation matrix features.
*   **Rolling Window Aggregations:** Applied rolling windows of sizes 3, 5, and 10 over key accelerometer and angular velocity features to compute `std`, and `min`.

### Thermopile (thm_) Sensor Features:
*   **Temporal Derivatives:** Calculated velocity (`_vel`), acceleration (`_accel`), and mean-centered (`_relative`) values for each of the five `thm_` sensors.
*   **Ratio Feature:** Computed a `thm_side_ratio` to capture the relationship between the outer four sensors and the central sensor.

### Time-of-Flight (tof_) Sensor Features:
*   **Per-Sensor Aggregations:** Calculated the mean distance, standard deviation of distance, and the count of active pixels for each of the five `tof_` sensors.
*   **Center of Mass (CoM):** Computed the weighted CoM (`_com_x`, `_com_y`) for each sensor's 8x8 grid and the distance of the CoM from the sensor's center.
*   **Regional (Quadrant) Statistics:** Divided each 8x8 sensor grid into four quadrants and calculated the `mean`, `std`, `min`, and `max` for each region.
*   **Texture and Contrast Features:** Calculated local contrast within each quadrant and overall distance uniformity (`_uniformity`) for each sensor.

### Augmentation and Training Strategy:
#### Mixup:
*   Used a `MixupSequence` generator during training to create synthetic data. New samples were formed by taking weighted combinations of random sample pairs from each batch, controlled by an alpha parameter of 0.3.

#### Epoch Averaging:
*   Implemented an epoch averaging callback to average the model's weights during the later stages of training. This was done to improve model generalization by simply taking the mean of several epochs.

### 1D-CNN Architecture:
*   **Multi-Branch Input:** The model uses three parallel branches with different kernel sizes (`[3,3,3]`, `[7,5,3]`, `[15,11,7]`) to capture short, medium, and long-term temporal patterns simultaneously.
*   **Branch Structure:** Each branch consists of a series of 1D Convolution, Batch Normalization, MaxPooling, and Dropout layers.
*   **Fusion and Classification:** The outputs of the three branches are concatenated, passed through a final convolutional block, followed by Global Max Pooling, and then classified using two dense layers with Dropout.
*   **Optimizer and Loss:** The model was trained using the AdamW optimizer and a Categorical Crossentropy loss function with label smoothing of 0.1.

### Cross-validation Setup:
*   **5-fold MultilabelStratifiedKFold:** Grouped per `subject`, and stratified on `adult_child`, `handedness`, and `sex`.

### Ensemble Strategy:
* For single models, I took the majority voting of each fold model.
* For the multi-model ensemble, took the mean of probabilities.
* I used Catboost, my custom CNN architecture and some public models

### Public Models:
* https://www.kaggle.com/code/wasupandceacar/lb-0-841-5fold-single-model-with-split-sensors
* https://www.kaggle.com/code/wasupandceacar/lb-0-82-5fold-single-bert-model
