import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import string
import json

# =========================
# NLTK (SAFE FOR RENDER)
# =========================
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize

NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

# NOTE: Download locally ONCE (not at runtime on Render)
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# =========================
# TensorFlow / Keras
# =========================
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(BASE_DIR, "instagram.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "static/images")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

db = SQLAlchemy(app)

# =========================
# Cyberbullying Model Init
# =========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(list(string.punctuation))

try:
    model = load_model("nagesh.h5")
    with open("word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    max_len = 30
    print("✅ Cyberbullying model loaded")

except Exception as e:
    print("❌ Model load failed:", e)
    model = None
    word_to_index = {}

# =========================
# Text Processing
# =========================
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text(text):
    words = word_tokenize(text)
    cleaned = []

    for word in words:
        if word.lower() not in stop_words:
            pos = pos_tag([word])[0][1]
            lemma = lemmatizer.lemmatize(word, get_simple_pos(pos))
            cleaned.append(lemma.lower())

    return " ".join(cleaned)

def sentences_to_indices(sentences, max_len):
    X = np.zeros((len(sentences), max_len))
    for i, sentence in enumerate(sentences):
        for j, w in enumerate(sentence.split()[:max_len]):
            X[i, j] = word_to_index.get(w, 0)
    return X

def detect_cyberbullying(text):
    if model is None:
        return 0.0

    cleaned = clean_text(text)
    indices = sentences_to_indices([cleaned], max_len)
    score = model.predict(indices, verbose=0)[0][0]
    return float(score)

# =========================
# DATABASE MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text, default="")
    profile_pic = db.Column(db.String(200), default="default_profile.png")
    reputation_score = db.Column(db.Float, default=10.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    bullying_score = db.Column(db.Float)
    user_id = db.Column(db.Integer)
    post_id = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Follow(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer)
    followed_id = db.Column(db.Integer)

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    post_id = db.Column(db.Integer)

# =========================
# HELPERS
# =========================
def get_current_user():
    if 'user_id' in session:
        return db.session.get(User, session['user_id'])
    return None

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    if not get_current_user():
        return redirect(url_for("login"))
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template("index.html", posts=posts)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session["user_id"] = user.id
            return redirect(url_for("index"))
        flash("Invalid login", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            username=request.form["username"],
            email=request.form["email"],
            full_name=request.form["full_name"],
            password=generate_password_hash(request.form["password"])
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/comment/<int:post_id>", methods=["POST"])
def comment(post_id):
    if not get_current_user():
        return jsonify({"error": "login required"}), 401

    text = request.json.get("comment", "")
    score = detect_cyberbullying(text)

    comment = Comment(
        text=text,
        bullying_score=score,
        post_id=post_id,
        user_id=session["user_id"]
    )
    db.session.add(comment)
    db.session.commit()

    return jsonify({"bullying_score": score})

# =========================
# START APP (RENDER SAFE)
# =========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000)
