
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from cv2 import dnn_superres
import os
import sys
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter
import datetime

# === CONFIGURATION ===
app = Flask(__name__)
app.config['SECRET_KEY'] = 'terminator-ai-secret-key-999' # In prod, use env var
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///antigravity.db')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

MODEL_NAME = "EDSR_x4.pb"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_NAME)
MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
SMALL_SIZE_THRESHOLD = 500

# === DATABASE MODELS ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    files = db.relationship('FileLog', backref='owner', lazy=True)

class FileLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300))
    processed_name = db.Column(db.String(300))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(50))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === CORE AI ENGINE (Unchanged but wrapped) ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(MODEL_URL, stream=True, timeout=60)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=4096):
                    f.write(chunk)
            return True
        except: return False
    return True

sr_model = None
def get_model():
    global sr_model
    if sr_model is None:
        if download_model():
            try:
                sr_model = dnn_superres.DnnSuperResImpl_create()
                sr_model.readModel(MODEL_PATH)
                sr_model.setModel("edsr", 4)
            except: pass
    return sr_model

def analyze_scene(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    b, c = gray.mean(), gray.std()
    if b < 90 and c < 45: return "Night"
    elif b < 90 and c >= 45: return "Dark_UI"
    return "Day"

def process_file_logic(filepath, filename):
    try:
        stream = np.fromfile(filepath, dtype=np.uint8)
        img_cv = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_cv is None: return None, ["Error: Corrupt File"]
        
        original_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        w, h = original_pil.size
        scene = analyze_scene(img_cv)
        sharpness = cv2.Laplacian(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        logs = [f"Metrics: {w}x{h}, Sharpness: {sharpness:.1f}, Scene: {scene}"]
        final_img = None
        suffix = ""

        if w < SMALL_SIZE_THRESHOLD or h < SMALL_SIZE_THRESHOLD:
            mod = get_model()
            if mod:
                res = mod.upsample(img_cv)
                final_img = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                suffix = "_AI"
                logs.append("Action: AI Upscaling (x4)")
            else:
                final_img = original_pil.resize((w*4, h*4), Image.Resampling.LANCZOS)
                suffix = "_Fallback"
        elif sharpness > 150:
             final_img = original_pil.resize((w*2, h*2), Image.Resampling.LANCZOS)
             suffix = "_HD"
             logs.append("Action: Lanczos x2")
        else:
             final_img = original_pil
             suffix = "_Enh"
             logs.append("Action: Enhancement Only")

        final_img = final_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
        
        if scene == "Night":
            final_img = ImageEnhance.Color(final_img).enhance(1.15)
            final_img = ImageEnhance.Brightness(final_img).enhance(1.10)
            suffix += "_Night"
        
        save_name = os.path.splitext(filename)[0] + suffix + ".png"
        final_img.save(os.path.join(app.config['PROCESSED_FOLDER'], save_name))
        return save_name, logs
    except Exception as e:
        return None, [str(e)]

# === ROUTES ===

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check details.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username exists.')
            return redirect(url_for('register'))
        
        hashed = generate_password_hash(password, method='scrypt')
        
        # First user is Admin automatically
        is_admin = False
        if User.query.count() == 0:
            is_admin = True
            
        new_user = User(username=username, password=hashed, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        return "Access Denied", 403
    users = User.query.all()
    logs = FileLog.query.order_by(FileLog.timestamp.desc()).limit(100).all()
    return render_template('admin.html', users=users, logs=logs)

@app.route('/admin/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if not current_user.is_admin:
        return "Access Denied", 403
        
    target = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        new_username = request.form.get('username')
        password = request.form.get('password')
        is_admin = 'is_admin' in request.form
        
        # Check username uniqueness if changed
        if new_username != target.username:
            if User.query.filter_by(username=new_username).first():
                flash('Username already exists.')
                return redirect(url_for('edit_user', user_id=user_id))
            target.username = new_username
            
        if password:
            target.password = generate_password_hash(password, method='scrypt')
            
        target.is_admin = is_admin
        db.session.commit()
        flash('User updated successfully.')
        return redirect(url_for('admin_panel'))
        
    return render_template('edit_user.html', target_user=target)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return "Access Denied", 403
        
    if user_id == current_user.id:
        flash("Cannot delete yourself.")
        return redirect(url_for('admin_panel'))
        
    target = User.query.get_or_404(user_id)
    
    # Optional: Delete related logs if cascading is not set up, but let's assume it's fine or we leave logs orphan for record.
    # Actually, let's delete their logs to be clean.
    FileLog.query.filter_by(user_id=target.id).delete()
    
    db.session.delete(target)
    db.session.commit()
    flash(f"User {target.username} deleted.")
    return redirect(url_for('admin_panel'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    files = request.files.getlist('file')
    results = []
    
    for file in files:
        if file.filename == '': continue
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processed_name, logs = process_file_logic(filepath, filename)
        
        status = 'success' if processed_name else 'error'
        
        # Log to DB
        new_log = FileLog(
            filename=filename,
            processed_name=processed_name if processed_name else "FAILED",
            owner=current_user,
            status=status
        )
        db.session.add(new_log)
        db.session.commit()

        if processed_name:
            results.append({
                'original': filename,
                'proxied_url': f"/processed/{processed_name}",
                'logs': logs,
                'status': 'success'
            })
        else:
            results.append({'original': filename, 'error': logs, 'status': 'error'})
            
    return jsonify({'results': results})

@app.route('/processed/<filename>')
@login_required
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Auto-download model on start
    download_model()
    app.run(debug=True, port=5000)
