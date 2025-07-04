# codealpha_task3
# AI Music Generator 🎵 🎹

This project is a deep learning-based **AI Music Generator** that learns patterns from MIDI files and creates new music compositions. Built using Python, TensorFlow, and the `music21` library.

---

## 🚀 Features

- 🎼 Loads and processes MIDI files
- 🎹 Extracts note sequences for training
- 🧠 Trains an LSTM neural network model
- 🎶 Generates new music based on learned patterns
- 💾 Saves the generated output as a `.mid` file

---

## 📁 Project Structure

codealpha_task3/
├── main.py # Main Python script
├── midi/ # Folder with input MIDI files
│ ├── song1.mid
│ └── song2.mid
├── generated_output.mid # Output generated MIDI
└── README.md # Project documentation


## 🛠️ How to Run

### 1. Clone the Repository

git clone https://github.com/Dhanshreeshende/codealpha_task3.git
cd codealpha_task3

## 2. Add MIDI Files
Add at least 2 MIDI files into the midi/ folder.

Example:

midi/
├── song1.mid
└── song2.mid

## 3. Run the Script

python main.py

## 4. Output
The generated music will be saved as: generated_output.mid

📦 Dependencies
tensorflow
kerasmusic21
numpy
sklearn

✅ All dependencies can be installed using the requirements.txt.

📌 Notes
The model requires enough note data to train. Use longer or more MIDI files for better results.


👩‍💻 Author
Developed by Dhanshree Shende
Internship Project for CodeAlpha – Task 3

📄 License
This project is open-source and available for educational use.
