from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import predict as pc

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) 

# Create DB
with app.app_context():
    db.create_all()

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists. Choose another.', 'danger')
            return redirect(url_for('signup'))
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# Login Route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

# Home Route (Protected)
@app.route('/home')
@login_required
def home():
    title = 'Plant-Growth Prediction- COLLEGE_NAME'
    return render_template('index.html', title=title)
import os
# Growth Prediction Route
@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = 'Plant-Growth Prediction- COLLEGE_NAME'

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
    

        if not file:
            flash('No file selected', 'danger')
            return render_template('disease.html', title=title)
        
        try:
            # Create 'input' directory if it doesn't exist
            input_folder = 'static/input'
            os.makedirs(input_folder, exist_ok=True)
            file_path = os.path.join(input_folder, file.filename)
            file.save(file_path)
            try:
                closest_stage,stages_resutl = pc.main(input_folder+"\\"+file.filename)
                
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
           
            return render_template('disease-result.html', disease={"Predicted Growth Stage":closest_stage}, image_filename=file.filename,image_path = stages_resutl, title=title)
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(request.url)
    print("here")
    return render_template('disease.html', title=title)

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
