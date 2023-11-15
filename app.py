import pickle
import streamlit as st
import altair as alt
import spacy

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

nlp = spacy.load('en_core_web_lg')

def preprocess(text):
  doc = nlp(text)
  filtered_tokens = []
  for token in doc:
    if token.is_stop or token.is_punct or token.like_email or token.like_url:
      continue
    filtered_tokens.append(token.lemma_)
  return ' '.join(filtered_tokens).strip().lower()

def main():
    st.title('Resume Screener')
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_text = preprocess(resume_text)
        input_features = tfidf.transform([cleaned_text])
        prediction_id = clf.predict(input_features)[0]

        category_mapping = {
                    15: "Java Developer",
                    23: "Testing",
                    8: "DevOps Engineer",
                    20: "Python Developer",
                    24: "Web Designing",
                    12: "HR",
                    13: "Hadoop",
                    3: "Blockchain",
                    10: "ETL Developer",
                    18: "Operations Manager",
                    6: "Data Science",
                    22: "Sales",
                    16: "Mechanical Engineer",
                    1: "Arts",
                    7: "Database",
                    11: "Electrical Engineering",
                    14: "Health and fitness",
                    19: "PMO",
                    4: "Business Analyst",
                    9: "DotNet Developer",
                    2: "Automation Testing",
                    17: "Network Security Engineer",
                    21: "SAP Developer",
                    5: "Civil Engineer",
                    0: "Advocate",
                }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)


if __name__ == "__main__":
    main()
