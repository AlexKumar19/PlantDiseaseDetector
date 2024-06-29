from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Load the YOLOv8 model
yolo_model = YOLO('best.pt')

# Load the Keras model
keras_model = tf.keras.models.load_model('leaf_classification_model.keras')

# Define the class names
class_names = [
    "Pepper bell Bacterial spot", "Healthy bell pepper", "Potato Early blight",
    "Potato Late blight", "Healthy potato", "Tomato Bacterial spot", "Tomato Early blight",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Septoria leaf spot",
    "Tomato Spider mites (Two spotted spider mite)", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato mosaic virus", "Healthy tomato"
]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

disease_prediction = None  # Variable to hold the disease prediction

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
    global disease_prediction  # Use the global variable to store the prediction
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
            
            # Annotate the frame without labels
            annotated_image = np.array(image)
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
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
            disease_prediction = class_names[np.argmax(avg_predictions)]
            print(f"Average Predictions: {avg_predictions}, Predicted Class: {disease_prediction}")  # Debug statement
            
            return redirect(url_for('display_image', filename='annotated_' + filename, prediction=disease_prediction))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def display_image(filename):
    prediction = request.args.get('prediction', '')
    file_url = os.path.join('uploads', filename).replace("\\", "/")
    print(f"Displaying image from {file_url}")  # Debug statement
    return render_template('display.html', filename=file_url, prediction=prediction)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on plants and you are going to answer questions specific to diseases on potatoes and tomatoes."},
            {"role": "system", "content": f"The predicted disease is: {disease_prediction}"},
            {"role": "user", "content": user_message}
        ]
    )
    return jsonify({"response": response.choices[0].message.content})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Video file saved to {filepath}")  # Debug statement
        
        return redirect(url_for('process_video', filename=filename))
    return redirect(url_for('video'))

@app.route('/process_video/<filename>')
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    def generate_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform object detection
            results = yolo_model.predict(source=frame)
            annotated_frame = results[0].plot()
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    
    return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
