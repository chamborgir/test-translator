import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# INSTALL (tensorflow, numpy, pandas)

# Paths
dataset_path = "./data/phrase-table-filtered"

# Parameters
batch_size = 64
epochs = 50
latent_dim = 256

# Storage for the dataset
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Read the dataset
with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("|||")
        if len(parts) != 2:
            # print(f"Skipping malformed line: {line.strip()}")
            continue
        input_text, target_text = parts
        input_text = input_text.strip()
        target_text = target_text.strip()
        target_text = "\t" + target_text + "\n"  # Add start and stop tokens
        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            input_characters.add(char)
        for char in target_text:
            target_characters.add(char)

# Check if the dataset is valid
if not input_texts:
    print("Error: No valid data found in the dataset. Please check the dataset format.")
    exit(1)

print(f"Loaded {len(input_texts)} valid translation pairs.")

# Prepare token sets
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max(len(txt) for txt in input_texts)
max_decoder_seq_length = max(len(txt) for txt in target_texts)

print(f"Number of samples: {len(input_texts)}")
print(f"Number of unique input tokens: {num_encoder_tokens}")
print(f"Number of unique output tokens: {num_decoder_tokens}")
print(f"Max sequence length for inputs: {max_encoder_seq_length}")
print(f"Max sequence length for outputs: {max_decoder_seq_length}")

# Token index mappings
input_token_index = dict((char, i) for i, char in enumerate(input_characters))
target_token_index = dict((char, i) for i, char in enumerate(target_characters))
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Encode the input and target data
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# Build the model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

# Build inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Function to decode sequences
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence

# Real-time translation
while True:
    user_input = input("Enter English text to translate (or 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(user_input):
        if char in input_token_index:
            input_seq[0, t, input_token_index[char]] = 1.0
    translated_text = decode_sequence(input_seq)
    print(f"Translated to Filipino: {translated_text}")
