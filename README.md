# Room Recognition from Ambient Sound

A neural network that identifies **which room you're in from its background noise**, and then starts music in that room through the **Spotify Web API**.

## How it works

1. **Data:** hour-long ambient recordings from four different rooms.
2. **Segmentation:** each recording is cut into fixed-length clips of 1,000,000 samples each, giving **650 clips** in total.
3. **Features:** each clip is turned into **40 MFCCs** (Mel-frequency cepstral coefficients, a compact "fingerprint" of a sound's frequency content) with `librosa`, averaged over time.
4. **Model:** a fully connected neural network in **Keras/TensorFlow**:
   `Dense(100) → Dense(200) → Dense(100) → Softmax(4)` with ReLU activations and 50% dropout
5. **Result:** **~97.7% accuracy** on a held-out 20% test set.
6. **Action:** the predicted room ID is mapped to that room's Spotify device, and a request to the Spotify Web API starts playback there.

## Built with

Python · TensorFlow / Keras · librosa (audio processing, MFCC) · scikit-learn · NumPy · pandas · Spotify Web API · Jupyter

## Running

1. Record ambient audio in each room you want to recognise and update the file paths in `Ambient.ipynb`.
2. Run the cells in order to extract features and train the model.
3. For the Spotify step, you need your own **Spotify access token** (from the [Spotify developer dashboard](https://developer.spotify.com/)) and your devices' IDs from the `/v1/me/player/devices` endpoint.
