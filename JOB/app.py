import os
import re
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

def extract_skills_experience(text):
    skills_pattern = re.compile(r"(?i)skills?:(.*?)(\n\n|\nExperience|Education|Projects|$)", re.DOTALL)
    experience_pattern = re.compile(r"(?i)experience:(.*?)(\n\n|\nSkills|Education|Projects|$)", re.DOTALL)
    projects_pattern = re.compile(r"(?i)projects?:(.*?)(\n\n|\nSkills|Experience|Education|$)", re.DOTALL)
    
    skills = skills_pattern.findall(text)
    experience = experience_pattern.findall(text)
    projects = projects_pattern.findall(text)
    
    skills = skills[0][0].strip().split(',') if skills else []
    experience = experience[0][0].strip() if experience else ''
    projects = projects[0][0].strip() if projects else ''
    
    return skills, experience, projects

def calculate_similarity(resume_text, job_description_text):
    documents = [resume_text, job_description_text]
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(documents)
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
        job_skills, job_experience, job_projects = extract_skills_experience(job_description_text)

        if 'resumes' not in request.files:
            return redirect(request.url)
        resumes_files = request.files.getlist('resumes')
        results = []

        for resume_file in resumes_files:
            resume_path = os.path.join(app.config['RESUMES_FOLDER'], secure_filename(resume_file.filename))
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)
            resume_skills, resume_experience, resume_projects = extract_skills_experience(resume_text)

            # Print debug statements
            print(f"Resume Skills: {resume_skills}")
            print(f"Job Skills: {job_skills}")

            # Calculate similarity based on skills
            skills_similarity = calculate_similarity(" ".join(resume_skills), " ".join(job_skills))
            
            # Check if skills are mentioned in projects
            skills_in_projects = all(skill.lower() in resume_projects.lower() for skill in resume_skills)

            # Final decision based on both criteria
            if skills_similarity >= 0.6 and skills_in_projects:
                status = "Selected"
            else:
                status = "Rejected"

            results.append((resume_file.filename, status))

        return render_template('index.html', results=results, job_skills=job_skills, job_experience=job_experience)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
