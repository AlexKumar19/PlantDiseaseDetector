from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the YOLOv8 model
yolo_model = YOLO('best.pt')

# Load the Keras model
keras_model = tf.keras.models.load_model('leaf_classification_model.keras')

# Define the class names
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for web camera

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = yolo_model.predict(source=frame)
            annotated_frame = results[0].plot()  # Annotate the frame with detections
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return render_template('video.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved to {filepath}")  # Debug statement

            # Perform object detection
            image = Image.open(filepath)
            results = yolo_model.predict(source=image)
            annotated_image = results[0].plot()

            # Convert BGR (OpenCV default) to RGB
            # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Save the annotated image
            annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
            Image.fromarray(annotated_image).save(annotated_image_path)
            
            print(f"Annotated image saved to {annotated_image_path}")  # Debug statement

            # Extract and preprocess all detected leaves
            leaves_images = []
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                leaf_image = np.array(image)[y1:y2, x1:x2]
                leaf_image = Image.fromarray(leaf_image).convert('RGB')  # Ensure the image is in RGB format
                leaf_image = leaf_image.resize((128, 128))
                leaf_array = np.array(leaf_image) / 255.0  # Normalize the image
                leaf_array = np.expand_dims(leaf_array, axis=0)  # Add batch dimension
                leaves_images.append(leaf_array)
            
            # Run the Keras model on each leaf and average the predictions
            all_predictions = []
            for leaf_array in leaves_images:
                predictions = keras_model.predict(leaf_array)
                print(f"Predictions: {predictions}")  # Debug statement
                all_predictions.append(predictions)
            
            # Average the predictions
            avg_predictions = np.mean(all_predictions, axis=0)
            predicted_class = class_names[np.argmax(avg_predictions)]
            print(f"Average Predictions: {avg_predictions}, Predicted Class: {predicted_class}")  # Debug statement
            
            return redirect(url_for('display_image', filename='annotated_' + filename, prediction=predicted_class))
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def display_image(filename):
    prediction = request.args.get('prediction', '')
    file_url = os.path.join('uploads', filename).replace("\\", "/")
    print(f"Displaying image from {file_url}")  # Debug statement
    return render_template('display.html', filename=file_url, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
