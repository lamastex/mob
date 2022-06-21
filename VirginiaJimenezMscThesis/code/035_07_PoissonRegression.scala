// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Poisson Linear Regression on the number of accidents.
// MAGIC 
// MAGIC Virginia Jimenez Mohedano ([LinkedIn](https://www.linkedin.com/in/virginiajimenezmohedano/)) and Raazesh Sainudiin ([LinkedIn](https://www.linkedin.com/in/raazesh-sainudiin-45955845/)).
// MAGIC 
// MAGIC ```
// MAGIC This project was supported by UAB SENSMETRY through a Data Science Thesis Internship 
// MAGIC between 2022-01-17 and 2022-06-05 to Virginia J.M. and 
// MAGIC Databricks University Alliance with infrastructure credits from AWS to 
// MAGIC Raazesh Sainudiin, Department of Mathematics, Uppsala University, Sweden.
// MAGIC ```
// MAGIC 
// MAGIC 2022, Uppsala, Sweden

// COMMAND ----------

import org.apache.spark.ml.regression._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions._


// COMMAND ----------

// Accidents dataset
val dataset0 = spark.read.parquet("dbfs:/datasets/lithuania/dataset_poissonReg.parquet")
val distances = spark.read.parquet("dbfs:/datasets/lithuania/acc_inters_distances.parquet")
val pageRank = spark.read.parquet("dbfs:/datasets/lithuania/acc_inters_pagerank_2")

// Dataset with pagerank info
val dataset1 = dataset0.join(distances, dataset0("id") === distances("acc_id")).drop("acc_id","number_of_lanes").join(pageRank, $"id" === $"acc_id").drop("id","acc_id","inters_id")


// COMMAND ----------

dataset1.count()

// COMMAND ----------

//dataset1.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with date anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +----------+----+-------+-----+-----+-----------------+-------------------+------------------+
// MAGIC |      date|time|weather|light|urban|surface_condition|distance_acc_inters|          pageRank|
// MAGIC +----------+----+-------+-----+-----+-----------------+-------------------+------------------+
// MAGIC |20xx-mm-dd|1545|     01|   01|   01|               01|   71.4226418463424|0.8232438315921039|
// MAGIC +----------+----+-------+-----+-----+-----------------+-------------------+------------------+
// MAGIC only showing top 1 row.
// MAGIC ```

// COMMAND ----------

// MAGIC %md
// MAGIC # Individual feature analysis
// MAGIC ## Dataset grouped in general

// COMMAND ----------

// Rounding and grouping by features
val dataset_rounded = dataset1.withColumn("distance_rounded",round($"distance_acc_inters",-1)).withColumn("pagerank_rounded",round($"pageRank",1))
val dataset_grouped_general = dataset_rounded.groupBy("weather","light","distance_rounded","urban","surface_condition").count()

// COMMAND ----------

dataset_grouped_general.count()

// COMMAND ----------

dataset_grouped_general.show()

// COMMAND ----------

// Transforming string into indexes (e.g. (01, 05, 99) -> (0, 1, 2))
// original: val strIndexer = new StringIndexer().setInputCols(Array("weather","light")).setOutputCols(Array("weather_index","light_index"))
// Multiple StringIndexers because of Spark 2
val strIndexer1 = new StringIndexer().setInputCol("weather").setOutputCol("weather_index").setStringOrderType("alphabetAsc") 
val strIndexer2 = new StringIndexer().setInputCol("light").setOutputCol("light_index").setStringOrderType("alphabetAsc") 
val strIndexer3 = new StringIndexer().setInputCol("urban").setOutputCol("urban_index").setStringOrderType("alphabetAsc") 
val strIndexer4 = new StringIndexer().setInputCol("surface_condition").setOutputCol("surface_index").setStringOrderType("alphabetAsc") 
val indexed1 = strIndexer1.fit(dataset_grouped_general).transform(dataset_grouped_general)
val indexed2 = strIndexer2.fit(indexed1).transform(indexed1)
val indexed3 = strIndexer3.fit(indexed2).transform(indexed2)
val indexed4 = strIndexer4.fit(indexed3).transform(indexed3)

// COMMAND ----------

// Transforming indexes into one hot encoding (e.g. (0, 1, 2) -> ([1,0,0], [0,1,0], [0,0,1]))
// called OneHotEncoder in Spark 3
val encoder = new OneHotEncoderEstimator().setInputCols(Array("weather_index","light_index","urban_index","surface_index")).setOutputCols(Array("weather_onehot","light_onehot","urban_onehot","surface_onehot")).setDropLast(false)
val encoded = encoder.fit(indexed4).transform(indexed4).select("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded","count")

// COMMAND ----------

indexed4.orderBy($"count".desc).select("weather", "light","urban", "surface_condition", "distance_rounded", "count").show()

// COMMAND ----------

val encodedAll = encoder.fit(indexed4).transform(indexed4)

// COMMAND ----------

encodedAll.printSchema

// COMMAND ----------

encoded.orderBy($"count".desc).show() 

// COMMAND ----------

encodedAll.select("light","light_index","light_onehot","count").orderBy($"count".desc).show()

// COMMAND ----------

// Transforming grouping all features into a feature column
val assembler = new VectorAssembler().setInputCols(Array("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded")).setOutputCol("features")
val assembled = assembler.transform(encoded).select($"count", $"features")

// COMMAND ----------

//CrossValidation to find the best parameter
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val pr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setLabelCol("count")

val paramGrid = new ParamGridBuilder()
     //.addGrid(pr.regParam, Array(0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001))
     //.addGrid(pr.regParam, Array(5, 4, 3, 2, 1, 0.5, 0.3, 0.1, 0.05, 0.01))
     //.addGrid(pr.regParam, Array(3.25, 3.2, 3.15, 3.1, 3.05, 3))
     //.addGrid(pr.regParam, Array(0.01, 0.1, 1.0, 10.0))
     //.addGrid(pr.regParam, Array(0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5))
     .addGrid(pr.regParam, Array(0.005, 0.01, 0.02, 0.03, 0.1, 0.5))
     .build()

val cv = new CrossValidator()
     .setEstimator(pr)
     .setEvaluator(new RegressionEvaluator().setLabelCol("count"))
     .setEstimatorParamMaps(paramGrid)
     .setNumFolds(5)

val model = cv.fit(assembled)

// COMMAND ----------

model.bestModel.explainParams

// COMMAND ----------

model.bestModel.extractParamMap()

// COMMAND ----------

// Doing regression
val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.03).setLabelCol("count")
val glrModel = glr.fit(assembled)
// Analyzing coefficients
glrModel.summary

// COMMAND ----------

// Doing regression
val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.02).setLabelCol("count")
val glrModel = glr.fit(assembled)
// Analyzing coefficients
glrModel.summary

// COMMAND ----------

// Doing regression
val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.01).setLabelCol("count")
val glrModel = glr.fit(assembled)
// Analyzing coefficients
glrModel.summary

// COMMAND ----------

// Doing regression // may be use this...
val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(100).setRegParam(0.005).setLabelCol("count")
val glrModel = glr.fit(assembled)
// Analyzing coefficients
glrModel.summary

// COMMAND ----------

// Doing regression // may be use this...
val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.005).setLabelCol("count")
val glrModel = glr.fit(assembled)
// Analyzing coefficients
glrModel.summary

// COMMAND ----------

// Sorting results and adding explanation
val coefficient_names = Array("Weather: Clear", "Weather: Rain", "Weather: Snow", "Weather: Fog", "Weather: Hail", "Weather: Severe winds", "Light: Daylight", "Light: Twilight", "Light: Darkness street lights lit", "Light: Darkness street lights unlit", "Light: Darkness no street lights", "Urban area: Yes", "Urban Area: No", "Surface:  Dry", "Surface conditions: Snow", "Surface conditions: Slippery", "Surface conditions: Wet", "Distance to intersection");
val coefficients = glrModel.coefficients.toArray;
val results = (coefficient_names zip coefficients).sortBy(- _._2);

// COMMAND ----------

// Printing results
for((name,value) <- results)
{
  println(name + ": " + value)
}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Dataset grouped by month

// COMMAND ----------

// Grouping by month
val dataset_grouped_month = dataset_rounded.withColumn("month",substring(col("date"),1,7)).groupBy("weather","light","distance_rounded","urban","surface_condition","month").count()

// COMMAND ----------

// Doing all the steps
val indexed1_month = strIndexer1.fit(dataset_grouped_month).transform(dataset_grouped_month)
val indexed2_month = strIndexer2.fit(indexed1_month).transform(indexed1_month)
val indexed3_month = strIndexer3.fit(indexed2_month).transform(indexed2_month)
val indexed4_month = strIndexer4.fit(indexed3_month).transform(indexed3_month)
val encoded_month = encoder.fit(indexed4_month).transform(indexed4_month).select("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded","count")
val assembled_month = assembler.transform(encoded_month).select($"count", $"features")
val glr_month = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.005).setLabelCol("count")
val glrModel_month = glr_month.fit(assembled_month)
val coefficients_month = glrModel_month.coefficients.toArray;
val results_month = (coefficient_names zip coefficients_month).sortBy(- _._2);

// COMMAND ----------

// Printing results
for((name,value) <- results_month)
{
  println(name + ": " + value)
}

// COMMAND ----------

glrModel_month.summary

// COMMAND ----------

// MAGIC %md
// MAGIC ## Dataset grouped by day

// COMMAND ----------

// Grouping by day
val dataset_grouped_day = dataset_rounded.groupBy("weather","light","distance_rounded","urban","surface_condition","date").count()

// COMMAND ----------

// Doing all the steps
val indexed1_day = strIndexer1.fit(dataset_grouped_day).transform(dataset_grouped_day)
val indexed2_day = strIndexer2.fit(indexed1_day).transform(indexed1_day)
val indexed3_day = strIndexer3.fit(indexed2_day).transform(indexed2_day)
val indexed4_day = strIndexer4.fit(indexed3_day).transform(indexed3_day)
val encoded_day = encoder.fit(indexed4_day).transform(indexed4_day).select("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded","count")
val assembled_day = assembler.transform(encoded_day).select($"count", $"features")
val glr_day = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(10).setRegParam(0.3).setLabelCol("count")
val glrModel_day = glr_day.fit(assembled_day)
val coefficients_day = glrModel_day.coefficients.toArray;
val results_day = (coefficient_names zip coefficients_day).sortBy(- _._2);

// COMMAND ----------

// Printing results
for((name,value) <- results_day)
{
  println(name + ": " + value)
}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Just light

// COMMAND ----------

// Doing all the steps
val dataset_grouped_light = dataset_rounded.groupBy("light").count()
val indexed_light = strIndexer2.fit(dataset_grouped_light).transform(dataset_grouped_light)
val encoder_light = new OneHotEncoderEstimator().setInputCols(Array("light_index")).setOutputCols(Array("light_onehot")).setDropLast(false)
val encoded_light = encoder_light.fit(indexed_light).transform(indexed_light).select("light_onehot","count")
val assembler_light = new VectorAssembler().setInputCols(Array("light_onehot")).setOutputCol("features")
val assembled_light = assembler_light.transform(encoded_light).select($"count", $"features")
val glr_light = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(10).setRegParam(0.3).setLabelCol("count")
val glrModel_light = glr_light.fit(assembled_light)
glrModel_light.summary

// COMMAND ----------

// MAGIC %md
// MAGIC ## Just distance

// COMMAND ----------

// Doing all the steps
val dataset_grouped_distance = dataset_rounded.groupBy("distance_rounded").count()
val assembler_distance = new VectorAssembler().setInputCols(Array("distance_rounded")).setOutputCol("features")
val assembled_distance = assembler_distance.transform(dataset_grouped_distance).select($"count", $"features")
val glr_distance = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(10).setRegParam(0.3).setLabelCol("count")
val glrModel_distance = glr_distance.fit(assembled_distance)
glrModel_distance.summary

// COMMAND ----------

dataset_grouped_distance.show(false)

// COMMAND ----------

// MAGIC %md
// MAGIC #Including PageRank

// COMMAND ----------

// Doing all the steps
val dataset_grouped_general_pagerank = dataset_rounded.groupBy("weather","light","distance_rounded","pagerank_rounded","urban","surface_condition").count()
val indexed1_pagerank = strIndexer1.fit(dataset_grouped_general_pagerank).transform(dataset_grouped_general_pagerank)
val indexed2_pagerank = strIndexer2.fit(indexed1_pagerank).transform(indexed1_pagerank)
val indexed3_pagerank = strIndexer3.fit(indexed2_pagerank).transform(indexed2_pagerank)
val indexed4_pagerank = strIndexer4.fit(indexed3_pagerank).transform(indexed3_pagerank)

val encoded_pagerank = encoder.fit(indexed4_pagerank).transform(indexed4_pagerank).select("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded","pagerank_rounded","count")

val assembler_pagerank = new VectorAssembler().setInputCols(Array("weather_onehot","light_onehot","urban_onehot","surface_onehot","distance_rounded","pagerank_rounded")).setOutputCol("features")
val assembled_pagerank = assembler_pagerank.transform(encoded_pagerank).select($"count", $"features")

val glr_pagerank = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.005).setLabelCol("count")
val glrModel_pagerank = glr_pagerank.fit(assembled_pagerank)
// Analyzing coefficients
glrModel_pagerank.summary


// COMMAND ----------

// MAGIC %md
// MAGIC ## Just PageRank

// COMMAND ----------

// Doing all the steps
val dataset_grouped_pagerank = dataset1.withColumn("pagerank_rounded",round($"pageRank",2)).groupBy("pagerank_rounded").count()
val assembler_pagerank = new VectorAssembler().setInputCols(Array("pagerank_rounded")).setOutputCol("features")
val assembled_pagerank = assembler_pagerank.transform(dataset_grouped_pagerank).select($"count", $"features")
val glr_pagerank = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.005).setLabelCol("count")
val glrModel_pagerank = glr_pagerank.fit(assembled_pagerank)
glrModel_pagerank.summary

// COMMAND ----------

// MAGIC %md
// MAGIC ### PageRank by month

// COMMAND ----------

// Doing all the steps
val dataset_grouped_pagerank_month = dataset1.withColumn("pagerank_rounded",round($"pageRank",2)).withColumn("month",substring(col("date"),1,7)).groupBy("pagerank_rounded","month").count()
val assembler_pagerank_month = new VectorAssembler().setInputCols(Array("pagerank_rounded")).setOutputCol("features")
val assembled_pagerank_month = assembler_pagerank_month.transform(dataset_grouped_pagerank_month).select($"count", $"features")
val glr_pagerank_month = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(10).setRegParam(0.3).setLabelCol("count")
val glrModel_pagerank_month = glr_pagerank_month.fit(assembled_pagerank_month)
glrModel_pagerank_month.summary

// COMMAND ----------

// MAGIC %md
// MAGIC ## PageRank and distance

// COMMAND ----------

// Doing all the steps
val dataset_grouped_general_pagerank_distance = dataset_rounded.groupBy("distance_rounded","pagerank_rounded").count()
val assembler_pagerank_distance = new VectorAssembler().setInputCols(Array("distance_rounded","pagerank_rounded")).setOutputCol("features")
val assembled_pagerank_distance = assembler_pagerank_distance.transform(dataset_grouped_general_pagerank_distance).select($"count", $"features")

val glr_pagerank_distance = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(50).setRegParam(0.005).setLabelCol("count")
val glrModel_pagerank_distance = glr_pagerank_distance.fit(assembled_pagerank_distance)
// Analyzing coefficients
glrModel_pagerank_distance.summary


// COMMAND ----------

dataset_grouped_general_pagerank_distance.count()

// COMMAND ----------

// MAGIC %md
// MAGIC # Grouped feature analysis

// COMMAND ----------

// Column values explained
val dataset2 = dataset1.withColumn("weather_explained", when($"weather" === "01", "Weather: Clear").when($"weather" === "02", "Weather: Rain").when($"weather" === "03", "Weather: Snow").when($"weather" === "04", "Weather: Fog").when($"weather" === "05", "Weather: Hail").when($"weather" === "06", "Weather: Severe winds")).withColumn("light_explained", when($"light" === "01", "Light: Daylight").when($"light" === "02", "Light: Twilight").when($"light" === "03", "Light: Darkness street lights lit").when($"light" === "04", "Light: Darkness street lights unlit").when($"light" === "05", "Light: Darkness no street lights")).withColumn("urban_explained", when($"urban" === "01", "Urban Area: No").when($"urban" === "02", "Urban Area: No")).withColumn("surface_explained", when($"surface_condition" === "01", "Surface: Dry").when($"surface_condition" === "02", "Surface: Snow").when($"surface_condition" === "03", "Surface: Slippery").when($"surface_condition" === "04", "Surface: Wet"))
dataset2.show()

// COMMAND ----------

val grouped = dataset2.withColumn("grouped",concat($"weather",$"light",$"urban",$"surface_condition")).withColumn("grouped_explained",concat($"weather_explained",$"light_explained",$"urban_explained",$"surface_explained")).withColumn("distance_rounded",round($"distance_acc_inters",-1)).select("grouped","grouped_explained","distance_rounded").groupBy("grouped","grouped_explained","distance_rounded").count()
grouped.show()

// COMMAND ----------

grouped.count()

// COMMAND ----------

// Doing all the steps
val strIndexer_grouped = new StringIndexer().setInputCol("grouped").setOutputCol("grouped_index").setStringOrderType("alphabetAsc") 
val indexed_grouped = strIndexer_grouped.fit(grouped).transform(grouped)
val encoder_grouped = new OneHotEncoderEstimator().setInputCols(Array("grouped_index")).setOutputCols(Array("onehot")).setDropLast(false)
val encoded_grouped = encoder_grouped.fit(indexed_grouped).transform(indexed_grouped).select("onehot","distance_rounded","count")
val assembler_grouped = new VectorAssembler().setInputCols(Array("onehot","distance_rounded")).setOutputCol("features")
val assembled_grouped = assembler_grouped.transform(encoded_grouped).select($"count", $"features")
val glr_grouped = new GeneralizedLinearRegression().setFamily("poisson").setLink("log").setMaxIter(10).setRegParam(0.3).setLabelCol("count")
val glrModel_grouped = glr_grouped.fit(assembled_grouped)
glrModel_grouped.summary
