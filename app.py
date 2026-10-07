from flask import Flask, render_template, request, redirect, session, url_for
import os
import numpy as np
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 256
CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

model = load_model('bloodgroup_alexnet_final.keras')

USERS = {}


# ------------------- ROUTES -------------------

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS:
            return "Username already exists!"

        USERS[username] = password
        return redirect('/login')

    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect('/upload')

        return "Invalid username or password!"

    return render_template("login.html")


@app.route('/upload')
def upload():
    if 'user' not in session:
        return redirect('/login')

    return render_template("upload.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return "Image cannot be processed!"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted = CLASSES[np.argmax(pred)]

    return render_template("result.html", prediction=predicted, filename=filename)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
