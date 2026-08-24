# Drowsiness Detector

A real-time Python-based drowsiness detection system that uses your webcam to monitor your eyes.

If your eyes remain closed for a few seconds, the system detects possible drowsiness and plays a custom voice alarm to wake you up.

## Features

- Real-time webcam monitoring
- Eye landmark detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection
- Custom voice alarm
- Real-time detection
- Works locally/offline

## Requirements

- Windows
- Python 3.x
- Webcam
- Working speakers or headphones

## Project Structure

```text
DrowsinessDetector/
|
|-- main.py
|-- alarm.wav
|-- requirements.txt
|-- README.md
|-- .gitignore
|
`-- models/
    `-- face_landmarker.task