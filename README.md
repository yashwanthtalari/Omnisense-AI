# OmniSense AI v4 🚀

OmniSense AI is a next-generation spatial intelligence platform that combines **MediaPipe Hand Tracking**, **Three.js AR**, and **Ollama Local Intelligence** to create an interactive 3D workspace controlled by hand gestures and voice.

---

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Node.js** (v18 or higher)
2.  **Ollama** (for local AI intelligence)
    -   Download from [ollama.com](https://ollama.com)
3.  **Webcam** (for hand tracking)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yashwanthtalari/Omnisense-AI.git
cd multi
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Setup Ollama Models
OmniSense requires local models to handle reasoning and vision. Run the following commands in your terminal:
```bash
ollama pull llama3.1
ollama pull llama3.2-vision
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add the following:
```env
VITE_OLLAMA_URL=http://localhost:11434/v1/chat/completions
VITE_TEXT_MODEL=llama3.1
VITE_VISION_MODEL=llama3.2-vision
```

### 5. Run the Application
```bash
npm run dev
```
Once started, open `http://localhost:5173` in your browser.

---

## ✋ Hand Gestures Guide

OmniSense uses advanced hand tracking to interact with 3D objects.

| Gesture | Action | Description |
| :--- | :--- | :--- |
| ✋ **Open Palm** | **Grab & Move** | Hover near a shape to grab it. Move your hand to position it. |
| ✊ **Fist** | **Drop / Delete** | Close your hand to drop an object. If no object is held, it deletes the selected one. |
| 🤏 **Pinch** | **Select** | Pinch your thumb and index finger to select the nearest object. |
| 👆 **Point** | **Select / Toggle** | Point to select objects or point at the 3D Mic icon for 1.2s to toggle voice. |
| ✌️ **Peace** | **Spawn** | Show the peace sign to spawn a random 3D shape at your palm. |

---

## 🎙 Voice Commands

Toggle the microphone and try saying:
- *"Spawn a sphere and move it to the left."*
- *"Clear the scene."*
- *"Scan the environment."* (Requires Ollama Vision)
- *"Change mode to infinite space."*

---

## 🏗 Technology Stack

-   **Frontend**: React + Vite
-   **3D Engine**: Three.js (@react-three/fiber & @react-three/drei)
-   **Hand Tracking**: Google MediaPipe Vision
-   **Intelligence**: Ollama (Llama 3.1 & 3.2-Vision)
-   **UI/Animations**: Framer Motion & Lucide React
