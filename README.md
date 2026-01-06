# Freak Detector

**Freak Detector** is a real-time computer vision application that uses webcam input to detect facial expressions, specifically tongue protrusion and closed eyes. The application displays corresponding GIFs or images in a separate reaction window while the detected expression is maintained.

The application is powered by [MediaPipe Face Mesh](https://developers.google.com/mediapipe) for facial landmark detection and [OpenCV](https://opencv.org/) for video processing.

---

## Overview

The application opens two windows:

- **Freak Detector:** displays the live webcam feed with facial detection overlay
- **Reaction:** displays GIFs or images based on detected facial expressions

Press **Q** at any time to close both windows and exit the application.

---

## Features

- Real-time webcam tracking and facial landmark detection
- Detection of tongue protrusion gestures
- Detection of closed eyes
- Separate reaction window for displaying visual feedback
- Standalone operation without external AI API dependencies

---

## Requirements

- **Python 3.12.x** (Note: `mediapipe` does not support Python 3.13 yet)
- Webcam or video capture device
- Compatible with **Windows**, **macOS**, and **Linux** operating systems

---

## Folder Structure

```
Freak-Detector/
│
├── assets/
│   ├── tongue.gif
│   ├── closed_eyes.gif
│
├── output/
│
├── main.py
└── README.md
```

---

## Installation and Usage

### Step 1: Clone the Repository

Open Terminal (CMD/Powershell on Windows) and execute:

```bash
git clone https://github.com/PavloKostereva/Freak-Detector.git
cd Freak-Detector
```

### Step 2: Install Python 3.12

Download Python 3.12 from the official website:
[https://www.python.org/downloads/release/python-3126/](https://www.python.org/downloads/release/python-3126/)

During installation:

- Check **"Add Python 3.12 to PATH"**
- Click **Install Now**

Verify the installation:

```bash
python --version
```

---

### Step 3: Create a Virtual Environment

Create a virtual environment using the following command:

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows:**

  ```bash
  .venv\Scripts\activate
  ```

- **Mac/Linux:**

  ```bash
  source .venv/bin/activate
  ```

---

### Step 4: Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install -r requirements.txt
```

Alternatively, install packages individually:

```bash
pip install opencv-python mediapipe imageio numpy
```

---

### Step 5: Run the Application

Execute the main script:

```bash
python main.py
```

Two windows will appear:

- `Freak Detector` → displays the camera feed with facial detection
- `Reaction` → displays the GIF or image corresponding to detected facial expressions

Press **Q** to quit the application.

---

## Customization

### Replacing Reaction Media

To customize the reaction GIFs or images, replace the files in the **`assets/`** folder:

| Expression  | File Name         | Description                     |
| ----------- | ----------------- | ------------------------------- |
| Tongue out  | `tongue.gif`      | GIF or image for tongue gesture |
| Eyes closed | `closed_eyes.gif` | GIF or image for closed eyes    |

Supported file formats include `.gif`, `.jpg`, `.png`, and `.mp4`. If using different file formats, update the corresponding file paths in the source code.

---

## Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

```
opencv-python
mediapipe
imageio
numpy
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Windows Quick Launch

For Windows users, create a **`run.bat`** file in the project root directory:

```bat
@echo off
call .venv\Scripts\activate
python main.py
pause
```

Double-click `run.bat` to launch the application without opening a terminal.

---

## License

See the LICENSE file for details.
