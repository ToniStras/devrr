from flask import Flask, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convertir l'image en RGB pour Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_frame)

        # Dessiner les contours des mains détectées
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Entourer la main d'un cadre rouge
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Encode le frame en JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        # Yield le frame encodé
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Page avec caméra</title>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f0f0f0;
                }
                #camera-container {
                    width: 80%;
                    max-width: 800px;
                    border: 2px solid #333;
                    border-radius: 10px;
                    overflow: hidden;
                }
                img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
            </style>
        </head>
        <body>
            <div id="camera-container">
                <img src="/video_feed" />
            </div>
        </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
