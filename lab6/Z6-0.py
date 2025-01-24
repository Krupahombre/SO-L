from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("LogisticRegressionExperiment").getOrCreate()

df = spark.read.csv("/content/sample_data/appendicitis.csv", inferSchema=True)

df.show()

df = df.withColumnRenamed("_c0", "feature1") \
       .withColumnRenamed("_c1", "feature2") \
       .withColumnRenamed("_c2", "feature3") \
       .withColumnRenamed("_c3", "feature4") \
       .withColumnRenamed("_c4", "feature5") \
       .withColumnRenamed("_c5", "feature6") \
       .withColumnRenamed("_c6", "feature7") \
       .withColumnRenamed("_c7", "label")

assembler = VectorAssembler(inputCols=[
    "feature1", "feature2", "feature3", 
    "feature4", "feature5", "feature6", 
    "feature7"
    ], outputCol="features")
df = assembler.transform(df)

df.show()

train_data, test_data = df.randomSplit([0.5, 0.5], seed=1234)

log_reg = LogisticRegression(labelCol="label", featuresCol="features")

paramGrid = (ParamGridBuilder()
             .addGrid(log_reg.regParam, [0.01, 0.1])
             .build())

evaluator = BinaryClassificationEvaluator(labelCol="label", 
                                          rawPredictionCol="prediction")

tvs = TrainValidationSplit(estimator=log_reg,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8)

model = tvs.fit(train_data)

predictions = model.transform(test_data)

predictions.select("features", "probability", "prediction", "label").show(20)