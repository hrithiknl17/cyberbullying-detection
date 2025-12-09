import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import string
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# --- 1. CONFIGURATION & SETUP ---

# Initialize NLTK data (Safe for cloud deployment)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Security: Use environment variable for secret key in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key-change-this')

# Database: Auto-switch between Cloud (Postgres) and Local (SQLite)
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///instagram.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File Uploads
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# --- 2. AI MODEL LOADING (PRESERVED LOGIC) ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(list(string.punctuation))

model = None
word_to_index = {}
max_len = 30

try:
    # Check if files exist before loading to prevent crashing on cloud if missing
    if os.path.exists('nagesh.h5'):
        model = load_model('nagesh.h5')
        print("✅ Cyberbullying detection model loaded successfully")
    else:
        print("⚠️ Warning: nagesh.h5 not found. AI features disabled.")

    if os.path.exists('word_to_index.pkl'):
        with open('word_to_index.pkl', 'rb') as f:
            word_to_index = pickle.load(f)
    else:
        print("⚠️ Warning: word_to_index.pkl not found.")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")

def get_simple_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_text(text):
    words = word_tokenize(text)
    output_words = []
    for word in words:
        if word.lower() not in stop_words:
            pos = pos_tag([word])
            clean_word = lemmatizer.lemmatize(word, pos=get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return " ".join(output_words)

def sentences_to_indices(X, max_len):
    X_indices = np.zeros((len(X), max_len))
    for i, sentence in enumerate(X):
        sentence_words = [w.lower() for w in sentence.split()]
        j = 0
        for word in sentence_words:
            if word in word_to_index and j < max_len:
                X_indices[i, j] = word_to_index[word]
                j += 1
    return X_indices

def detect_cyberbullying(text):
    if model is None:
        return 0.0  # Default to "safe" if model is missing
    
    try:
        cleaned_text = clean_text(text)
        text_indices = sentences_to_indices([cleaned_text], max_len)
        prediction = model.predict(text_indices)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0

# --- 3. DATABASE MODELS ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text, default='')
    profile_pic = db.Column(db.String(200), default='default_profile.png')
    reputation_score = db.Column(db.Float, default=10.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    posts = db.relationship('Post', backref='author', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    followers = db.relationship('Follow', foreign_keys='Follow.followed_id', backref='followed', lazy='dynamic')
    following = db.relationship('Follow', foreign_keys='Follow.follower_id', backref='follower', lazy='dynamic')
    likes = db.relationship('Like', backref='user', lazy=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    comments = db.relationship('Comment', backref='post', lazy=True)
    likes = db.relationship('Like', backref='post', lazy=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    bullying_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Follow(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    followed_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- 4. ROUTES ---

@app.before_request
def before_request():
    if request.endpoint in ['static', 'login', 'register', 'logout']:
        return
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user is None:
            session.pop('user_id', None)
            return redirect(url_for('login'))

def get_current_user():
    if 'user_id' not in session:
        return None
    return db.session.get(User, session['user_id'])

# --- NEW: The "Nudge" Feature Route ---
@app.route('/check_content', methods=['POST'])
def check_content():
    """Real-time check for frontend to warn users BEFORE they post"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'score': 0})
        
    score = detect_cyberbullying(text)
    
    # Return friendly warning if score is high
    warning = None
    if score > 0.4:
        warning = "⚠️ Wait! This content looks aggressive. It could hurt your reputation."
        
    return jsonify({
        'score': score,
        'is_bullying': score > 0.4,
        'warning': warning
    })
# --------------------------------------

@app.route('/')
def index():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    following_ids = [f.followed_id for f in user.following.all()]
    following_ids.append(user.id)
    
    posts = Post.query.filter(Post.user_id.in_(following_ids)).order_by(Post.created_at.desc()).all()
    
    # Suggestions: Users not followed
    suggestions = User.query.filter(User.id.notin_(following_ids)).limit(5).all()
    
    return render_template('index.html', user=user, posts=posts, suggestions=suggestions)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        
        if User.query.filter((User.username==username) | (User.email==email)).first():
            flash('Username or Email already exists', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password, full_name=full_name)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_post():
    user = get_current_user()
    if not user: return redirect(url_for('login'))
    
    file = request.files.get('image')
    caption = request.form.get('caption', '')
    
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        # Unique filename to prevent overwrites
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        new_post = Post(image=filename, caption=caption, user_id=user.id)
        db.session.add(new_post)
        db.session.commit()
        flash('Post uploaded!', 'success')
    else:
        flash('No file selected', 'error')
    
    return redirect(url_for('index'))

@app.route('/comment/<int:post_id>', methods=['POST'])
def add_comment(post_id):
    user = get_current_user()
    if not user: return jsonify({'error': 'Login required'}), 401
    
    data = request.get_json()
    comment_text = data.get('comment', '').strip()
    
    if not comment_text:
        return jsonify({'error': 'Empty comment'}), 400
    
    # Check restrictions
    if user.reputation_score < 5.0:
        return jsonify({'error': 'Account restricted due to low reputation.'}), 403
    
    # AI Detection
    bullying_score = detect_cyberbullying(comment_text)
    
    new_comment = Comment(
        text=comment_text, user_id=user.id, post_id=post_id, bullying_score=bullying_score
    )
    db.session.add(new_comment)
    
    # Update Reputation
    reputation_loss = 0
    if bullying_score > 0.4:
        reputation_loss = bullying_score * 2  # Penalize heavily for confirmed bullying
        user.reputation_score = max(0.0, user.reputation_score - reputation_loss)
        
        flash(f'🚨 Cyberbullying detected! Reputation dropped by {reputation_loss:.2f}.', 'warning')
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'author': user.username,
        'text': new_comment.text,
        'reputation_loss': reputation_loss,
        'current_score': user.reputation_score
    })

@app.route('/like/<int:post_id>', methods=['POST'])
def like_post(post_id):
    user = get_current_user()
    if not user: return jsonify({'error': 'Login required'}), 401
    
    existing = Like.query.filter_by(user_id=user.id, post_id=post_id).first()
    
    if existing:
        db.session.delete(existing)
        liked = False
    else:
        new_like = Like(user_id=user.id, post_id=post_id)
        db.session.add(new_like)
        liked = True
        
    db.session.commit()
    return jsonify({'liked': liked})

@app.route('/profile/<username>')
def profile(username):
    user = get_current_user()
    profile_user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(user_id=profile_user.id).order_by(Post.created_at.desc()).all()
    
    is_following = False
    if user and user.id != profile_user.id:
        is_following = Follow.query.filter_by(follower_id=user.id, followed_id=profile_user.id).first() is not None
        
    return render_template('profile.html', user=user, profile_user=profile_user, posts=posts, is_following=is_following)

@app.route('/follow/<int:user_id>', methods=['POST'])
def follow_user(user_id):
    user = get_current_user()
    if not user: return jsonify({'error': 'Login required'}), 401
    
    if user.id == user_id:
        return jsonify({'error': 'Cannot follow self'}), 400
        
    existing = Follow.query.filter_by(follower_id=user.id, followed_id=user_id).first()
    
    if existing:
        db.session.delete(existing)
        following = False
    else:
        if user.reputation_score < 5.0:
            return jsonify({'error': 'Restricted account cannot follow.'}), 403
        
        new_follow = Follow(follower_id=user.id, followed_id=user_id)
        db.session.add(new_follow)
        following = True
        
    db.session.commit()
    return jsonify({'following': following})

@app.route('/explore')
def explore():
    user = get_current_user()
    if not user: return redirect(url_for('login'))
    
    # Random posts from people you don't follow
    following_ids = [f.followed_id for f in user.following.all()]
    following_ids.append(user.id)
    
    posts = Post.query.filter(Post.user_id.notin_(following_ids)).order_by(db.func.random()).limit(20).all()
    return render_template('explore.html', user=user, posts=posts)

@app.route('/search')
def search():
    user = get_current_user()
    query = request.args.get('q', '')
    users = []
    if query:
        users = User.query.filter(User.username.contains(query) | User.full_name.contains(query)).all()
    return render_template('search.html', user=user, users=users, query=query)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    user = get_current_user()
    if not user: return redirect(url_for('login'))
    
    if request.method == 'POST':
        user.full_name = request.form.get('full_name', user.full_name)
        user.bio = request.form.get('bio', user.bio)
        
        file = request.files.get('profile_pic')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user.profile_pic = filename
            
        db.session.commit()
        flash('Profile updated.', 'success')
        
    return render_template('settings.html', user=user)

# --- 5. INITIALIZATION ---

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)