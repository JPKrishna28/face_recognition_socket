import os
import cv2
import numpy as np
import face_recognition
import base64
from flask import Flask, request, render_template, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create directory for uploaded images if it doesn't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store face encodings
known_face_encodings = []
known_face_names = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    if 'file' not in request.files:
        return render_template('register.html', error="No file selected")

    file = request.files['file']
    name = request.form['name']

    if file.filename == '':
        return render_template('register.html', error="No file selected")

    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, f"{name}_{file.filename}")
    file.save(file_path)

    # Load the uploaded image and find face encodings
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return render_template('register.html', error="No face found in the image", image_path=file_path)

    if len(face_locations) > 1:
        return render_template('register.html', error="Multiple faces found. Please upload an image with only one face",
                               image_path=file_path)

    # Get face encodings and store them
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Store the encoding and name
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

    return render_template('register.html', success=True, name=name, image_path=file_path)


@app.route('/check', methods=['POST'])
def check():
    if len(known_face_encodings) == 0:
        return render_template('check.html', error="No faces registered yet")

    if 'file' not in request.files:
        return render_template('check.html', error="No file selected")

    file = request.files['file']

    if file.filename == '':
        return render_template('check.html', error="No file selected")

    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, f"check_{file.filename}")
    file.save(file_path)

    # Load the uploaded image and find face encodings
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return render_template('check.html', error="No face found in the image", image_path=file_path)

    # Get face encodings for the faces in the check image
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Create an image to draw faces on
    image_cv = cv2.imread(file_path)

    matches = []

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with all known faces
        matches_list = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches_list[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw rectangle around the face and add name
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_cv, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        matches.append({
            "name": name,
            "is_match": name != "Unknown"
        })

    # Save the result image with rectangles
    result_path = os.path.join(UPLOAD_FOLDER, f"result_{file.filename}")
    cv2.imwrite(result_path, image_cv)

    return render_template('check.html', matches=matches, image_path=file_path, result_path=result_path)


@app.route('/live')
def live():
    return render_template('live.html')


# WebSocket for handling live camera feed
@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('image')
def handle_image(data):
    if len(known_face_encodings) == 0:
        emit('response_back', {'message': 'No faces registered yet'})
        return

    # Get the image data from the client
    image_data = data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Convert to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check for matches
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw a rectangle and name for the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        results.append({
            'name': name,
            'is_match': name != "Unknown",
            'location': [top, right, bottom, left]
        })

    # Convert the processed image back to base64 to send to client
    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')

    # Emit the processed image and results back to the client
    emit('response_back', {
        'image': f"data:image/jpeg;base64,{processed_image}",
        'results': results
    })


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)