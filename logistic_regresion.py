import os
import sys
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import Bucketizer
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator


import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)



rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed

if __name__ == "__main__":
    spark = SparkSession.builder.master("yarn").appName("Logistic-Regression-Amazon-Reviews").getOrCreate()

    #conf = SparkConf()
    #conf.setMaster('yarn')
    #conf.setAppName('spark-test')
    sc = spark.sparkContext# SparkContext(conf=conf)

    sqlContext = SQLContext(sc)

    AMAZON_DATA = sys.argv[1]

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    
    amazon_df = spark.read.options(delimiter="\t", header=True).csv(AMAZON_DATA)


    
    cols = ("marketplace","customer_id","review_id","product_id","product_parent","product_title","product_category","helpful_votes","total_votes","vine","verified_purchase","review_date","review_headline")
    sentiments = amazon_df.drop(*cols)
    sentiments = sentiments.withColumn("star_rating",col("star_rating").cast(IntegerType()))
    bucketizer = Bucketizer()

    bucketizer.setSplits([-float("inf"), 3 , float("inf")])
    
    bucketizer.setInputCol("star_rating")
    bucketizer.setOutputCol("target")

    labeled = bucketizer.setHandleInvalid("keep").transform(sentiments)
    labeled = labeled.drop("star_rating")
    labeled = labeled.dropna()
    labeled.printSchema()
    labeled.show()


    (train_set, val_set, test_set) = labeled.randomSplit([0.70, 0.01, 0.29], seed = 2000)
    
    


    tokenizer = RegexTokenizer(inputCol="review_body", outputCol="words", pattern="\\s+|[,.()\"]")
    

    cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
    idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5)
    label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
    lr = LogisticRegression(maxIter=100)
    pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])
    

    pipelineFit = pipeline.fit(train_set)
    predictions = pipelineFit.transform(val_set)
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
    roc_auc = evaluator.evaluate(predictions)

    print("Accuracy Score: {0:.4f}".format(accuracy))
    print("ROC-AUC: {0:.4f}".format(roc_auc))

