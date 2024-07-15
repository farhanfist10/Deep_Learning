import spacy

nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    return skills

job_description = "Looking for a data scientist with experience in Python, machine learning, and NLP."
skills = extract_skills(job_description)
print(skills)
