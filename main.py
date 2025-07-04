import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess dataset
def load_dataset(folder_path="midi"):
    notes = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid"):
            try:
                midi = converter.parse(os.path.join(folder_path, file))
                parts = instrument.partitionByInstrument(midi)
                elements = parts.parts[0].recurse() if parts else midi.flat.notes

                for element in elements:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            except Exception as e:
                print(f"Error loading {file}: {e}")
    print(f" Loaded {len(notes)} notes.")
    return notes

# Prepare sequences for training
def prepare_sequences(notes, sequence_length=32):
    if len(notes) < sequence_length + 1:
        print("⚠️ Not enough notes to prepare sequences.")
        return None, None, None

    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitchnames))
    network_output = to_categorical(network_output, num_classes=len(pitchnames))

    return network_input, network_output, pitchnames

# Build LSTM model
def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate music
def generate_notes(model, network_input, pitchnames, num_notes=100):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]

    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitchnames))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern[1:], [[index]], axis=0)

    return prediction_output

# Save generated notes to MIDI
def create_midi(prediction_output, output_file="generated_output.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"🎶 Generated music saved to '{output_file}'")

# Main execution
if __name__ == "__main__":
    notes = load_dataset("midi")
    if len(notes) < 100:
        print(" Not enough note data. Please add more MIDI files.")
        exit()

    X, y, pitchnames = prepare_sequences(notes)
    if X is None:
        print(" Unable to prepare data. Exiting.")
        exit()

    model = build_model((X.shape[1], X.shape[2]), len(pitchnames))
    model.fit(X, y, epochs=10, batch_size=64)

    output_notes = generate_notes(model, X, pitchnames, num_notes=100)
    create_midi(output_notes)

