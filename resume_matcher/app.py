import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

UPLOAD_FOLDER = 'uploads'
JOB_DESCRIPTION_FOLDER = os.path.join(UPLOAD_FOLDER, 'job_description')
RESUMES_FOLDER = os.path.join(UPLOAD_FOLDER, 'resumes')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JOB_DESCRIPTION_FOLDER'] = JOB_DESCRIPTION_FOLDER
app.config['RESUMES_FOLDER'] = RESUMES_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def calculate_similarity(resume_text, job_description_text):
    documents = [resume_text, job_description_text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return cosine_sim[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'job_description' not in request.files:
            return redirect(request.url)
        job_description_file = request.files['job_description']
        if job_description_file.filename == '':
            return redirect(request.url)
        job_description_path = os.path.join(app.config['JOB_DESCRIPTION_FOLDER'], secure_filename(job_description_file.filename))
        job_description_file.save(job_description_path)
        job_description_text = extract_text_from_pdf(job_description_path)

        if 'resumes' not in request.files:
            return redirect(request.url)
        resumes_files = request.files.getlist('resumes')
        results = []

        for resume_file in resumes_files:
            resume_path = os.path.join(app.config['RESUMES_FOLDER'], secure_filename(resume_file.filename))
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)
            similarity_score = calculate_similarity(resume_text, job_description_text)
            status = "Selected" if similarity_score >= 0.6 else "Rejected"
            results.append((resume_file.filename, status))

        results.sort(key=lambda x: x[1], reverse=True)
        return render_template('index.html', results=results, job_description_text=job_description_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
