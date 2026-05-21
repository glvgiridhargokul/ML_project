from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

RESOURCES = {
    'Data Scientist': ['Python', 'Pandas', 'NumPy', 'Statistics'],
    'Machine Learning Engineer': ['Scikit-learn', 'TensorFlow', 'MLOps'],
    'Full Stack Web Developer': ['HTML/CSS', 'JavaScript', 'React', 'Flask'],
    'Cybersecurity Analyst': ['Networking', 'Kali Linux', 'Ethical Hacking'],
    'Mobile App Developer': ['Kotlin', 'Android Studio'],
    'UI/UX Designer': ['Figma', 'Wireframing']
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Student')
    skills = request.form.get('skills', '')
    career = model.predict(vectorizer.transform([skills]))[0]
    resources = RESOURCES.get(career, ['Problem Solving'])
    return render_template('result.html', name=name, skills=skills, career=career, resources=resources)

if __name__ == '__main__':
    app.run(debug=True)
