# NLP-Resume-Screener

This repository contains code to run an NLP based Streamlit application using a trained KNN model
for resume screening and to assess the alignment between application skills and the job description.

Bag of Words(n-gram CountVectorization), TF-IDF and Word Embedding techniques were all tried to make
sure the best representation of vectorized data was used to train the KNN model. The final model was
trained using TF-IDF Vectorization.

Spacy was used to map words to their base token lemmas and remove stop sign words and punctuation
symbols from the text, essentially removing unnecessary data.
