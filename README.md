# AI-Based Smart Traffic System with Emergency Vehicle Detection

## Overview

This project presents an AI-based smart traffic management system that performs real-time vehicle detection and adaptive traffic signal control using computer vision. The system analyzes traffic congestion across four independent traffic lanes and dynamically allocates green signal duration based on congestion levels. The project is designed to improve traffic flow and reduce waiting time while providing a scalable framework for emergency vehicle prioritization.

---

## Features

- Real-time vehicle detection using YOLOv8
- Four-lane traffic monitoring dashboard
- Adaptive traffic signal control
- Lane-wise congestion estimation
- Dynamic green signal allocation
- Road-width-aware congestion analysis (In Progress)
- Emergency vehicle detection and priority signal control (In Progress)

---

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

---

## Project Structure

```
AI-Based-Smart-Traffic-System/
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Smart-Traffic-System.git
```

Move into the project folder:

```bash
cd AI-Based-Smart-Traffic-System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the pretrained YOLOv8 Nano model:

```bash
yolo detect predict model=yolov8n.pt
```

Place the following files in the project directory:

- lane1.mp4
- lane2.mp4
- lane3.mp4
- lane4.mp4
- yolov8n.pt

Run the project:

```bash
python main.py
```

---

## Current Implementation

- Vehicle detection and counting
- Four-lane traffic monitoring
- Adaptive traffic signal allocation
- Congestion-based lane prioritization

---

## Future Enhancements

- Emergency vehicle detection
- Siren sound analysis
- Queue length estimation
- Weighted congestion score
- Real-time CCTV integration
- Smart city deployment

---

## Author

**Dhanvitha K N**

Computer Science and Business Systems  
Canara Engineering College