import sys


from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.ml.feature import IndexToString, StringIndexer

if __name__ == "__main__":
    spark = SparkSession.builder.master("yarn").appName("Colaborative-Filtering-Amazon-Reviews").getOrCreate()

    sc = spark.sparkContext

    sqlContext = SQLContext(sc)

    AMAZON_DATA = sys.argv[1]

    amazon_df = spark.read.options(delimiter="\t", header=True).csv(AMAZON_DATA)



    cols = ("marketplace","review_id","product_id","product_parent","product_title","helpful_votes","total_votes","vine","verified_purchase","review_date","review_headline","review_body")
    amazon_filtered = amazon_df.drop(*cols)
    amazon_filtered.show(5)

    amazon_filtered = amazon_filtered.withColumn("star_rating",col("star_rating").cast(IntegerType()))
    amazon_filtered = amazon_filtered.withColumn("customer_id",col("customer_id").cast(IntegerType()))
    
    amazon_filtered = amazon_filtered.dropna()
    
    indexer = StringIndexer(inputCol="product_category", outputCol="categoryIndex")
    #indexer.setHandleInvalid("error")
    model = indexer.fit(amazon_filtered)
    

    td = model.transform(amazon_filtered)

    td.show(5)

    (training, test) = td.randomSplit([0.8, 0.2])
    
    als = ALS(maxIter=5, regParam=0.01, userCol="customer_id", itemCol="categoryIndex", ratingCol="star_rating",
          coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="star_rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
