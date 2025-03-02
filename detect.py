from deep_emotion_recognition import DeepEmotionRecognizer
import json
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()
# get the accuracy
print(deeprec.test_score())
# predict angry audio sample

# I have 2 options, either run it every time I need
# OR periodically check for files


sharedPath = r'C:\Users\R1nge\Documents\TELEGRAM\SHARED\1.oga'
prediction = deeprec.predict(sharedPath)
print(f"Prediction: {prediction}")

with open(f'{sharedPath}/emotions.json', 'w', encoding='utf-8') as f:
    json.dump(prediction, f, ensure_ascii=False, indent=4)