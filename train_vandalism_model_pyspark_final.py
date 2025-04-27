
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
import os
import joblib

# Start Spark Session
spark = SparkSession.builder.appName("WikipediaVandalismDetection").getOrCreate()

# Load the dataset
df_spark = spark.read.csv("edits-en.csv", header=True, inferSchema=True)

# Fill missing edit comments with empty string
df_spark = df_spark.fillna({'editcomment': ''})

# Add a feature: comment length
df_spark = df_spark.withColumn('comment_length', length(col('editcomment')))

# Group and show original label distribution
print("✅ Original Label Distribution:")
df_spark.groupBy('class').count().show()

# Map text labels to numeric
df_spark = df_spark.withColumn(
    "numeric_label",
    when(col("class") == "regular", 0.0)
    .when(col("class") == "vandalism", 1.0)
    .otherwise(None)
)

# Drop rows where numeric_label is NULL
df_spark = df_spark.filter(col("numeric_label").isNotNull())

# Show new label distribution
print("✅ Label Distribution after Mapping:")
df_spark.groupBy('numeric_label').count().show()

# Tokenization
tokenizer = Tokenizer(inputCol="editcomment", outputCol="words")
words_data = tokenizer.transform(df_spark)

# Stopwords removal
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
words_data = remover.transform(words_data)

# TF-IDF feature extraction
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
featurized_data = hashingTF.transform(words_data)

idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
idfModel = idf.fit(featurized_data)
rescaled_data = idfModel.transform(featurized_data)

# Feature assembly
assembler = VectorAssembler(
    inputCols=["tfidf_features", "comment_length"],
    outputCol="features"
)
final_data = assembler.transform(rescaled_data)

# Selecting final dataset
model_data = final_data.select("features", col("numeric_label").alias("label"))

# Checking total rows
after_count = model_data.count()
print(f"✅ Total rows available for training: {after_count}")

if after_count == 0:
    print("❌ No valid labeled data available for training. Exiting.")
    exit(1)

# Train-Test Split
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

# Model training
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=1000)
lr_model = lr.fit(train_data)

# Save model coefficients manually (bypass Hadoop issue)
os.makedirs("models", exist_ok=True)
model_coefficients = lr_model.coefficients.toArray()
model_intercept = lr_model.intercept

# Save with joblib
joblib.dump((model_coefficients, model_intercept), "models/lr_model.joblib")

print("✅ Model coefficients and intercept saved manually using joblib under 'models/' folder!")
