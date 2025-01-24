from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("LogisticRegressionAndDecisionTreeExperiment").getOrCreate()

df = spark.read.csv("/content/sample_data/appendicitis.csv", inferSchema=True, header=False)

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

train_data, test_data = df.randomSplit([0.5, 0.5], seed=1234)

log_reg = LogisticRegression(labelCol="label", featuresCol="features")
decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")

evaluator = BinaryClassificationEvaluator(labelCol="label", 
                                          rawPredictionCol="prediction")

models = {
    "LogReg": log_reg,
    "DecTree": decision_tree
}

param_grids = {
    "LogReg": ParamGridBuilder()
                   .addGrid(log_reg.regParam, [0.01, 0.1, 1.0])
                   .addGrid(log_reg.elasticNetParam, [0.0, 0.5, 1.0])
                   .build(),
    "DecTree": ParamGridBuilder()
                 .addGrid(decision_tree.maxDepth, [3, 5, 10])
                 .addGrid(decision_tree.minInstancesPerNode, [1, 5, 10])
                 .build()
}

predictions = {}
for name, model in models.items():
  cv = CrossValidator(estimator=model,
                          estimatorParamMaps=param_grids[name],
                          evaluator=evaluator,
                          numFolds=5)

  cv_model = cv.fit(train_data)
  predictions[name] = cv_model.transform(test_data)

  print(f"{name} Average Metrics: ", max(cv_model.avgMetrics))