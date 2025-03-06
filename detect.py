from deep_emotion_recognition import DeepEmotionRecognizer
import json
import os
import time
import subprocess

sharedPath = r'C:\Users\R1nge\Documents\TELEGRAM\SHARED'
statePath = os.path.join(sharedPath, "STATE")
statePathJson = os.path.join(statePath, "STATE.json")
faceEmotions = os.path.join(sharedPath, "Emotions_Face.bat")

# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
deeprec.train()
print(deeprec.test_score())

while True:
    with open(statePathJson, 'r', encoding='utf-8') as f:
        contents = f.read()
        state = json.loads(contents)
        print(state)

    if state['state'] == "splitting_finished":        
        with open(statePathJson,'w', encoding='utf-8') as f:
            json.dump(state, f ,ensure_ascii=False, indent=4)

        i = 0
        for file in os.listdir(sharedPath):
            if file.endswith(".oga"):
                # Analyze each file and save to a JSON
                prediction = deeprec.predict(os.path.join(sharedPath, file))
                print(f"Prediction for {file}: {prediction}")
                data = {
                    "emotion": prediction
                }
                with open(os.path.join(sharedPath, f"{i}.json"), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                i += 1

            if file.endswith(".mp4"):
                prediction = deeprec.predict(os.path.join(sharedPath, file))
                print(f"Prediction for {file}: {prediction}")
                data = {
                    "emotion": prediction
                }
                with open(os.path.join(sharedPath, f"{i}.json"), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    print(data)
                i += 1


        time.sleep(2)
        result = subprocess.run([faceEmotions], shell=True)
        print(result)
    else:
        print("waiting")
        time.sleep(1)