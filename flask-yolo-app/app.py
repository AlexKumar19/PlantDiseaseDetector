from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os

# Load your YOLOv8 model from the runs directory
model = YOLO('/home/vishnuaa77/vscode/PlantDiseaseDetector/flask-yolo-app/best.pt')

#static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
#template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

#app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)
app = Flask(__name__)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for web camera

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model.predict(source=frame)
            annotated_frame = results[0].plot()  # Annotate the frame with detections
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
