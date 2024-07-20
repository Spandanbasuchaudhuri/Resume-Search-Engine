import pandas as pd
import re
import os
import multiprocessing as mp
from whoosh import index
from whoosh.fields import Schema, TEXT, KEYWORD
from whoosh.qparser import MultifieldParser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

skills_list = [
    'Python', 'R', 'SQL', 'Java', 'C++', 'Tableau', 'Power BI', 'Excel',
    'Machine Learning', 'Deep Learning', 'NLP', 'Pandas', 'NumPy', 'SciPy',
    'TensorFlow', 'Keras', 'PyTorch', 'AWS', 'Azure', 'Git', 'Docker', 'Kubernetes'
]

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

def preprocess_resumes(resumes):
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(preprocess_text, resumes)

def extract_skills(text):
    skills_found = []
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills_found.append(skill)
    return skills_found

def extract_skills_parallel(resumes):
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(extract_skills, resumes)

def batch_write(index, docs, batch_size=1000):
    writer = index.writer()
    for i, doc in enumerate(docs):
        writer.add_document(**doc)
        if i > 0 and i % batch_size == 0:
            writer.commit()
            writer = index.writer()
    writer.commit()

def search_resumes(ix, skills):
    result_ids = []
    query_str = " AND ".join(skills)
    with ix.searcher() as searcher:
        query = MultifieldParser(["resume", "skills"], ix.schema).parse(query_str)
        results = searcher.search(query, limit=None)
        for result in results:
            result_ids.append(result['id'])
    return result_ids

if __name__ == '__main__':
    download_nltk_data()

    file_path = r'F:\Data\UpdatedResumeDataSet.csv'
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded.")

    print("Preprocessing resumes...")
    df['Cleaned_Resume'] = preprocess_resumes(df['Resume'])
    print("Preprocessing complete.")

    print("Extracting skills...")
    df['Skills'] = extract_skills_parallel(df['Cleaned_Resume'])
    print("Skill extraction complete.")

    schema = Schema(id=TEXT(stored=True), category=TEXT(stored=True), resume=TEXT(stored=True), skills=KEYWORD(stored=True))

    index_dir = "indexdir"
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    print("Creating index...")
    ix = index.create_in(index_dir, schema)

    documents = [
        {
            'id': str(i),
            'category': row['Category'],
            'resume': row['Cleaned_Resume'],
            'skills': ",".join(row['Skills'])
        }
        for i, row in df.iterrows()
    ]

    batch_write(ix, documents)
    print("Index creation complete.")

    while True:
        query = input("Enter the skills to search for (comma-separated, or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        skills = [skill.strip() for skill in query.split(',')]
        resume_ids = search_resumes(ix, skills)
        print("Resumes found:", resume_ids)
