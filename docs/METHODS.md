# Methods

This file explains the definitions and choices behind the numbers in the README, so that anyone can check them or change them.

## 1. Data and cleaning

- Source: OpenBiomechanics `dataset-v1` baseball pitching. Files used: `poi_metrics.csv` (one row per pitch, 3D summary metrics and event times), `metadata.csv` (pitcher id, mass, height, level), `landmarks.csv` (joint centres at 360 Hz) and `joint_angles.csv` (45 angle channels at 360 Hz).
- Lab frame: +x from second base to home plate, +y toward first base, +z up, metres.
- Left-handed pitchers are mirrored (y → −y), so every trial is a right-hander with the throwing arm on −y, the third-base side.
- Events: peak knee height (`pkh_time`), foot plant (`fp_100_time`, lead-foot force reaches 100 % body weight), ball release (`BR_time`). Foot plant uses the 100 % threshold rather than the 10 % one because full weight-bearing is the moment that is visible on video.
- Plausibility filter, fixed before any analysis: foot plant to release between 80 and 250 ms, and knee lift to foot plant between 0.30 and 1.00 s. It removes 18 of 411 pitches, leaving 393 from 98 pitchers.

## 2. Metrics

Every camera metric is written twice in `pitchlab/metrics.py`: once from the 3D markers (the truth) and once from 2D pixels (what a camera sees). Both use the same joint centres and the same event frame, so the difference is the error that a single camera adds.

| Metric | 3D truth | 2D image |
|---|---|---|
| Elbow flexion | 180° − angle shoulder–elbow–wrist | same, in pixels |
| Lead knee flexion | 180° − angle hip–knee–ankle | same, in pixels |
| Trunk forward tilt | mid-hip → mid-shoulder line projected on the x–z plane, angle from vertical, positive toward home | the same line's angle from image vertical, signed toward home |
| Trunk lateral tilt | the same line projected on the y–z plane, absolute angle from vertical | absolute angle from image vertical |
| Arm slot | elbow → hand projected on the y–z plane, elevation above horizontal | elbow → hand elevation above image horizontal |
| Hip-shoulder separation | acute angle between the hip line and the shoulder line, both projected on the ground plane | acute angle between the two lines in the image |
| Stride length | rear ankle to lead ankle on the ground plane, as % of height | the same in pixels, divided by a pixel height estimate |

The pixel height comes from the set position: rear ankle to throwing shoulder in the first frame, divided by 0.779 (shoulder height 0.818 H minus ankle height 0.039 H, Drillis and Contini 1966).

The trunk is the mid-hip to mid-shoulder line because that is the only trunk line a 2D pose model gives. It is not the same as OBP's thorax segment angle, which is why Study A uses OBP's own trunk metrics and Study B uses this definition for both truth and image.

## 3. Virtual camera (Study B)

- Pinhole camera, 1920 × 1080, focal length 3024 px, 10 m from a target point at (1.0, −0.3, 1.0) m, lens at 1.2 m. This frames a 2.5 m tall volume in about 70 % of the image height, roughly a phone on a tripod down the line.
- Four named views by azimuth: catcher 0°, glove side 90°, center field 180°, arm side 270°. The geometry sweep covers every 15°.
- Frame rate is simulated by keeping every k-th frame of the 360 Hz data (360, 240, 120, 60 fps), starting at a random phase so the camera frames do not line up with the lab events.
- Noise: independent Gaussian, 4 px SD per coordinate per frame, then the same 12 Hz zero-phase Butterworth low-pass that the video pipeline uses.

Two tests:

1. **Geometry sweep**: image metric at the true event frame vs truth, with 0 and 4 px noise. This isolates what the projection alone does.
2. **End-to-end**: events detected from the 2D keypoints, metric read at the detected frame, vs truth at the true frame. This is what a user would actually get.

## 4. Event detection (2D)

In `pitchlab/events.py`:

- **Peak knee height**: the frame where the lead knee is highest in the image (smallest y).
- **Ball release**: the frame of peak throwing-wrist speed after peak knee height, plus a fixed offset per view. The offset is the median gap between that peak and the true release. In the end-to-end test it is learned on four folds of pitchers and applied to the fifth (GroupKFold), so no pitcher's own release is used to time their release. The final offsets saved in `configs/event_offsets.json` use all pitchers.
- **Foot plant**: from peak knee height to 30 ms before the wrist-speed peak, take the frame where the lead ankle moves fastest, then the first frame after it where the ankle speed drops below 15 % of that peak.

An earlier foot-plant rule based on ankle height had outliers near 850 ms and was replaced. The current rule and its 15 % threshold were set on the logic of the stride, not tuned against the error table.

## 5. Tiers

At 120 fps, 4 px noise and detected events, for each view and metric:

- **A**: RMSE ≤ 5 (degrees or % height) and Pearson r ≥ 0.80. Report.
- **B**: RMSE ≤ 10 and r ≥ 0.60. Report with caution.
- **C**: anything else. Withheld, with the RMSE and r shown as the reason.

r matters as much as RMSE: a metric with low error but no correlation cannot tell two pitchers apart. 95 % intervals on RMSE come from 1,000 bootstrap resamples of pitchers.

## 6. Norms

For tier A and B metrics only, `configs/norms.json` stores the 5th to 95th percentiles of the *image* metric (not the marker metric), taken from OBP pitches run through the same camera, noise and event detection. A user's number is therefore compared with numbers produced the same way, which keeps any small bias out of the percentile. Percentiles by playing level are stored when a level has at least 20 pitches. Windmill deliveries get no percentiles.

## 7. Study A: drivers

- Unit: pitcher. Each pitcher's 2–5 pitches are averaged first (n = 100).
- 15 mechanics chosen before running anything, taken directly from OBP's 3D POI metrics. Two outcomes: pitch speed and peak elbow varus moment normalised as 100 × M / (mass × 9.81 × height), in %BW·Ht.
- Model: `outcome ~ mechanic_z + mass_z + height_z`, ordinary least squares. The effect is the coefficient on the mechanic, in outcome units per 1 SD.
- 95 % CI: 2,000 bootstrap resamples of pitchers. p-value: 5,000 permutations of the mechanic across pitchers. Benjamini-Hochberg across all 30 tests; significant means q < 0.05.

## 8. Study C: leakage

- Input: the 45 OBP joint-angle channels from peak knee height to release, resampled to 64 steps and standardised on the training fold.
- BiLSTM: one bidirectional layer, 32 hidden units; the mean over time and the last step are joined, then dropout 0.2 and a linear head. Target standardised on the training fold. Adam, learning rate 3e-3, weight decay 1e-3, 150 epochs, batch size 32. Fixed before any run, no tuning.
- Ridge: alpha = 10 on the 76 numeric POI metrics, standardised on the training fold.
- Pitcher memory: predicts the mean speed of the same pitcher's training pitches, or the training mean for an unseen pitcher.
- Splits: `KFold(5, shuffle)` for random, `GroupKFold(5)` by pitcher for pitcher-out. Three seeds (fold shuffling for the random split, network initialisation for both). For each seed the out-of-fold predictions are pooled and scored with R² and MAE, and the table reports the mean over seeds.

## 9. Video pipeline

- Pose: MediaPipe Pose Landmarker (heavy), 33 landmarks per frame. Landmarks with visibility under 0.3 are dropped, gaps up to 6 frames are filled linearly, and a clip is rejected if any needed joint is missing in more than 20 % of frames.
- Mapping: the throwing hand sets which side is "throwing" and which is "lead". The hand point is the mean of wrist, index and pinky.
- From there the code path is the same as Study B: low-pass, events, metrics, tier gate, percentile.

## 10. What is not validated

- Real phone video against markers on the same pitches. Study B uses projected markers with synthetic noise. Pose models are usually worse than 4 px on a fast, blurred throwing arm, so the tiers are a best case.
- Windmill pitching, non-fastball pitches, and youth pitchers.
- Camera placement errors (height, distance, tilt away from the standard positions) beyond the 15° azimuth sweep.
