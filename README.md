# AI Push-Up Correction – Real-Time Form Feedback

A computer vision project that helps users improve their push-up form by providing instant feedback through a webcam.

## Overview

This system uses AI and pose estimation to analyze body movements during push-ups and determine whether the exercise is being performed correctly. It captures body landmarks in real time, processes the movement data, and instantly classifies each repetition as either **Correct** or **Wrong**.

To make the feedback more stable and user-friendly, prediction smoothing is applied to reduce rapid fluctuations between classifications.

## Features

* Real-time body pose detection using a webcam
* Tracks 33 body landmarks with MediaPipe Pose
* Provides instant feedback on push-up form
* Classifies posture as **Correct** or **Wrong**
* Uses a trained Random Forest model with 96% accuracy
* Smooth prediction output for a better user experience

## Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn (Random Forest)

## Run the Project

```bash
pip install -r requirements.txt
python pushup_realtime.py
```
