# Speech-Emotion-Recognition-System-using-CNN-and-Bi-LSTM-
# Speech Emotion Recognition (SER) Using CNN-BiLSTM

## Overview
This project implements a **Speech Emotion Recognition (SER) system** using a **CNN-BiLSTM** architecture. The system analyzes audio signals, extracting both spatial and temporal features to classify emotions accurately. The combination of **Convolutional Neural Networks (CNNs)** for feature extraction and **Bidirectional Long Short-Term Memory (BiLSTM)** for sequential modeling enhances performance in recognizing emotions from speech data.

## Features
- **Preprocessing Audio Data**: Converts raw speech signals into Mel spectrograms for efficient analysis.
- **Feature Extraction with CNN**: Captures spatial patterns in spectrograms.
- **Sequential Modeling with BiLSTM**: Learns temporal dependencies in speech for improved classification.
- **Multi-Class Emotion Classification**: Recognizes emotions such as happy, sad, angry, neutral, etc.
- **Performance Metrics**: Evaluates accuracy, precision, recall, and confusion matrices for validation.

## Dataset
This model is trained on publicly available speech emotion datasets such as:
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- **TESS (Toronto Emotional Speech Set)**

## Model Architecture
### **CNN-BiLSTM Pipeline**
1. **Preprocessing**:
   - Convert audio to Mel spectrogram.
   - Normalize and augment data.
2. **CNN Layers**:
   - Extract spatial features from spectrograms.
   - Apply ReLU activation and max-pooling.
3. **BiLSTM Layers**:
   - Capture temporal dependencies in speech sequences.
   - Improve long-range feature extraction.
4. **Fully Connected Layers**:
   - Classify extracted features into emotion categories using Softmax.

## Workflow
1. **Load and preprocess audio data.**
2. **Extract Mel spectrograms from audio signals.**
3. **Train CNN-BiLSTM model on labeled emotion datasets.**
4. **Evaluate model performance using test data.**
5. **Deploy for real-time speech emotion recognition.**

## Experimentation
### **Hyperparameter Tuning**
- Learning rate adjustment to **0.001** for optimal training.
- Batch normalization for stable convergence.
- Data augmentation techniques to improve generalization.

### **Performance Comparison**
| Model           | Accuracy (%) |
|----------------|-------------|
| CNN Only       | 75.4        |
| LSTM Only      | 78.1        |
| CNN-BiLSTM    | **85.7**    |

## Libraries & Dependencies
- **Python 3**
- **TensorFlow / Keras**
- **Librosa (for audio processing)**
- **Matplotlib & Seaborn (for visualization)**
- **NumPy & Pandas**

## Future Improvements
- Implementing **transformers** for improved sequence modeling.
- Extending to **multilingual emotion recognition**.
- Enhancing real-time inference speed.

## References
- [TensorFlow Speech Processing Guide](https://www.tensorflow.org/tutorials/audio)
- [Librosa Documentation](https://librosa.org/doc/main/)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)

## Conclusion
This project demonstrates the power of **CNN-BiLSTM** in recognizing emotions from speech signals. By integrating **deep learning** techniques, it improves upon traditional methods and provides a robust solution for emotion classification in speech.


