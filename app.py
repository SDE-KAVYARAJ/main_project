from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime
from flask_migrate import Migrate

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

app = Flask(__name__)
# ---------------------------- Sqlite setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
# login_manager.login_view = 'login'
login_manager.init_app(app)

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    prediction_output = db.Column(db.String(100), nullable=True) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def track_activity(user_id, action_type, file_path, prediction_output=None):
    activity = Activity(user_id=user_id, action_type=action_type, file_path=file_path, prediction_output=prediction_output)
    db.session.add(activity)
    db.session.commit()


# Define User Loader Function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# --------------------------

dependencies = {
    'auc_roc': AUC
}

class_names = {
0: 'Tampered (Fake)',
1: 'Authentic (Real)',

}
# Fake logo classess
# classes = ["Fake", "Genuine"]
# Fake logo Model
model_fake = tf.saved_model.load('C:\\Users\\vishn\\OneDrive\\Documents\\c c++\\Kavya_projects\\JPPY2309-Digital Image\\SOURCE CODE\\logo_pred\\ResNet50')
# graph = tf.Graph()
# with graph.as_default():
#     model = tf.keras.models.load_model('C:\\Users\\vishn\\OneDrive\\Documents\\c c++\\Kavya_projects\\JPPY2309-Digital Image\\SOURCE CODE\\logo_pred\\ResNet50')
# Image forgery model
model = load_model('casia.h5')

# Load your video forgery detection model
video_model = load_model("C:\\Users\\vishn\\OneDrive\\Documents\\c c++\\Kavya_projects\\JPPY2309-Digital Image\\SOURCE CODE\\Resnet50\\ResNet50forgery_model.hdf5")

# Registration 

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('registration.html')
# ===================================================
# Function to perform fake logo detection
def detect_fake_logo(image_path):
    classes = ["Fake", "Genuine"]
    img = Image.open(image_path).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.LANCZOS)
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    class_scores = model_fake(inp)[0].numpy()
    predicted_class = classes[class_scores.argmax()]
    return predicted_class

# ====================================================
def detect_video_forgery(video_path):
    vid = []
    sum_frames = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        sum_frames += 1
        vid.append(frame)
    cap.release()
    
    Xtest = np.array(vid)
    output = video_model.predict(Xtest)
    results = (output > 0.5).astype(int)
    no_of_forged = np.sum(results)
    
    forged_percentage = (no_of_forged / sum_frames) * 100
    
    if forged_percentage < 70:  # If less than 50% of frames are forged, consider it authentic
        return "The video is not forged"
    else:
        return f"The video is forged. Percentage of forged frames: {forged_percentage:.2f}%"

    
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

image_size = (200, 200)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
# @app.route("/login")
# def login():
# 	return render_template('login.html')   
#new login code
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and password == user.password:
            login_user(user)
            session['username'] = username
            # Track user login activity
            # track_activity(user.id, 'Login')
            return redirect('/index')
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login1.html')
    
@app.route("/index", methods=['GET', 'POST'])
@login_required
def index():
    username = session.get('username')
    user_activities = Activity.query.filter_by(user_id=current_user.id).order_by(Activity.timestamp.desc()).all()
    # return render_template("index.html", user_activities=user_activities)
    return render_template("index.html", username=username,user_activities=user_activities)

# Code for previous activities
@app.route("/activity_history")
@login_required
def activity_history():
    # Fetch user's activity history from the database
    user_activities = Activity.query.filter_by(user_id=current_user.id).order_by(Activity.timestamp.desc()).all()
    return render_template("activity_history.html", user_activities=user_activities)


@app.route("/image", methods=['GET', 'POST'])
@login_required
def image():
     return render_template('image.html')

@app.route('/logout')
@login_required
def logout():
    # Track user logout activity
    # track_activity(current_user.id, 'Logout')
    logout_user()
    return redirect(url_for('login'))

# Route to render video.html
@app.route("/video")
@login_required
def video():
    return render_template("video1.html")

# Route to render logo.html
@app.route("/logo")
@login_required
def logo():
    return render_template("logo.html")

@app.route("/submit", methods = ['GET', 'POST'])
@login_required
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        # model = request.form['model']
        # print(model)
    
        img_path = "static/tests/" + img.filename    
        img.save(img_path)
        #plt.imshow(img)
        
        image = prepare_image(img_path)
        image = image.reshape(-1, 200, 200, 3)
        y_pred = model.predict(image) 
        y_pred_class = np.argmax(y_pred, axis=1)[0]
        predict_result = class_names[y_pred_class]   
        if current_user.is_authenticated:
            track_activity(current_user.id, 'Fake Image Detection', img_path, predict_result)

        #  print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    return render_template("prediction.html", prediction=predict_result, img_path=img_path)


# Route to handle video forgery detection
@app.route("/video", methods=['POST'])
def detect_video():
    if request.method == 'POST':
        video_file = request.files['my_video']
        video_path = "static/videos/" + video_file.filename
        video_file.save(video_path)
        forgery_result = detect_video_forgery(video_path)
        if current_user.is_authenticated:
            track_activity(current_user.id,'Video Forgery Detection', video_path, forgery_result)
        return render_template("video_prediction.html", result=forgery_result, video_path=video_path)

# Route to handle Fake logo detection    
@app.route("/logo", methods=['POST'])
@login_required
def detect_logo():
    if request.method == 'POST':
        img = request.files['my_logo']
        img_path = "static/logos/" + img.filename
        img.save(img_path)

        # Perform fake logo detection
        predicted_class = detect_fake_logo(img_path)
        if current_user.is_authenticated:
            track_activity(current_user.id,'Fake Logo Detection', img_path, predicted_class)
        return render_template("logo_result.html", prediction=predicted_class, img_path=img_path)

# ---------------------------------------------
@app.route("/performance")
@login_required
def performance():
	return render_template('performance.html')   

@app.route("/chart")
@login_required
def chart():
	return render_template('chart.html')   
	
if __name__ =='__main__':
	app.run(debug = True)


	

	


