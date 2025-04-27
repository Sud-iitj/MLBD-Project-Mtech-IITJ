
import streamlit as st
import requests
import re
from urllib.parse import quote
import numpy as np
import joblib
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.sql import Row
from pyspark.sql.functions import length

# Start Spark session
spark = SparkSession.builder.appName("WikipediaVandalismPredictionApp").getOrCreate()

# Load saved coefficients and intercept
model_coefficients, model_intercept = joblib.load("models/lr_model.joblib")

# Initialize TF and Tokenizer
tokenizer = Tokenizer(inputCol="editcomment", outputCol="words")
stopword_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)

# Streamlit App
st.title("Wikipedia Vandalism Detector (with PySpark and Manual Model)")

wiki_url = st.text_input("Enter Wikipedia Page Link (e.g., https://en.wikipedia.org/wiki/Air_Jordan)")

if wiki_url:
    def extract_page_title(url):
        return url.split("/")[-1].replace("_", " ")

    def get_latest_edit_comment(title):
        history_url = f"https://en.wikipedia.org/w/index.php?title={quote(title)}&action=history"
        response = requests.get(history_url)
        comment_match = re.search(r'class="comment">\((.*?)\)</span>', response.text)
        if comment_match:
            return comment_match.group(1)
        else:
            return "No comment found"

    # Get page title and latest comment
    page_title = extract_page_title(wiki_url)
    latest_comment = get_latest_edit_comment(page_title)

    st.markdown(f"**Latest Edit Comment:** {latest_comment}")

    # Prepare single input DataFrame
    df_input = spark.createDataFrame([Row(editcomment=latest_comment)])

    # Preprocessing
    df_input = df_input.withColumn('comment_length', length('editcomment'))
    df_words = tokenizer.transform(df_input)
    df_filtered = stopword_remover.transform(df_words)
    df_tf = hashingTF.transform(df_filtered)

    # Manually build feature vector
    raw_features = df_tf.select('rawFeatures').first()['rawFeatures'].toArray()
    comment_length_value = df_input.select('comment_length').first()['comment_length']
    feature_vector = np.append(raw_features, comment_length_value)

    # Manual prediction
    score = np.dot(feature_vector, model_coefficients) + model_intercept
    probability = 1 / (1 + np.exp(-score))

    prediction = 1 if probability >= 0.5 else 0

    st.write("### Prediction:", "🛡️ Regular" if prediction == 0 else "⚠️ Vandalism")
    st.metric("Confidence", f"{float(probability)*100:.2f}%")

    if prediction == 1:
        st.markdown("## 🔍 Vandalism detected. Please review this page carefully.")
