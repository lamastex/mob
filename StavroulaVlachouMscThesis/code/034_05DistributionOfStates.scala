// Databricks notebook source
// MAGIC %md
// MAGIC ## Posterior - Conditional Distribution of State Counts for a Given Time Unit  
// MAGIC 
// MAGIC Stavroula Rafailia Vlachou ([LinkedIn](https://www.linkedin.com/in/stavroula-rafailia-vlachou/)) and Raazesh Sainudiin ([LinkedIn](https://www.linkedin.com/in/raazesh-sainudiin-45955845/)).
// MAGIC 
// MAGIC ```
// MAGIC This project was supported by SENSMETRY through a Data Science Project Internship 
// MAGIC between 2022-01-17 and 2022-06-05 to Stavroula R. Vlachou and 
// MAGIC Databricks University Alliance with infrastructure credits from AWS to 
// MAGIC Raazesh Sainudiin, Department of Mathematics, Uppsala University, Sweden.
// MAGIC ```
// MAGIC 
// MAGIC 2022, Uppsala, Sweden

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import spark.implicits._

// COMMAND ----------

val different_dates = spark.read.parquet("/FileStore/tables/LTaccidents_id_date.parquet").toDF("id", "date").orderBy($"date".asc).select("date").rdd.map(element => element(0)).collect.toSet;
val distinct_dates  = different_dates.toList;

// COMMAND ----------

//the conditional distribution for each state given a time unit
def conditional_distribution(sample_date: String): org.apache.spark.sql.DataFrame = {
  import spark.implicits._
  
  val id_date = spark.read.parquet("/FileStore/tables/LTaccidents_id_date.parquet").toDF("id", "date")
  val matched_events = spark.read.parquet("dbfs:/_checkpoint/GeoMatch_G0").toDF("point", "state")
  val state_counts = matched_events.join(id_date, matched_events("point") === id_date("id"), "inner").drop("id").where($"date" === sample_date).groupBy("state").count()
  val global_count = state_counts.count.toFloat
  val state_space = spark.read.parquet("dbfs:/_checkpoint/StateSpaceInitialG0").toDF("initial_state","count").drop("count")
  val per_state_conditional_counts = state_space.join(state_counts, state_space("initial_state") === state_counts("state"), "left_outer").na.fill(0, Seq("count"))
  val number_of_states = state_space.count.toFloat
  val all_state_counts = per_state_conditional_counts.select("initial_state", "count").withColumn("prior", lit(1f/number_of_states)).orderBy($"count".asc)
  val df = all_state_counts.select(col("initial_state"), col("count").cast(FloatType), col("prior")).withColumn("global_count", lit(global_count))

  val posteriors = df.selectExpr("initial_state", "count + prior as posterior", "global_count")

  val posterior_means = posteriors.selectExpr("initial_state","posterior/(global_count + 1) as posterior_mean").orderBy($"posterior_mean".asc)
  posterior_means.createOrReplaceTempView("posterior_means")
  val df_1 = spark.sql("select initial_state, posterior_mean,"+" SUM(posterior_mean) over ( order by initial_state rows between unbounded preceding and current row ) cumulative_Sum " + " from posterior_means").toDF("initial_state", "posterior_mean", "cumulative_Sum")
  val df_2 = df_1.withColumn("prob_interval", lag($"cumulative_Sum", 1,0).over(Window.orderBy($"cumulative_Sum".asc))).select("initial_state", "prob_interval", "cumulative_Sum")

  val probability_intervals = df_2.selectExpr("initial_state", "(prob_interval, cumulative_Sum) as prob_interval")
  return probability_intervals
}

// COMMAND ----------

//run only once per cluster 
var date = ""

for (date <- distinct_dates){
    val a = date.toString 
    var directory = "dbfs:/roadSafety"
    val probabilities = conditional_distribution(sample_date=a)
    directory += "_" + a 
    dbutils.fs.mkdirs(directory)
    probabilities.write.mode(SaveMode.Overwrite).parquet(directory + "_CD")
    probabilities.unpersist
    display(dbutils.fs.ls(directory))  
}

// COMMAND ----------

//The distribution of states independent of time 
def unconditional_distribution(): org.apache.spark.sql.DataFrame = {
  import spark.implicits._
  val state_space = spark.read.parquet("dbfs:/_checkpoint/StateSpaceInitialG0").toDF("initial_state","count").drop("count")
  val number_of_states = state_space.count.toFloat
  val priors = state_space.select("initial_state").withColumn("prior", lit(1f/number_of_states))
  priors.createOrReplaceTempView("priors")
  val df_1 = spark.sql("select initial_state, prior,"+" SUM(prior) over ( order by initial_state rows between unbounded preceding and current row ) cumulative_Sum " + " from priors").toDF("initial_state", "prior", "cumulative_Sum")
  val df_2 = df_1.withColumn("prob_interval", lag($"cumulative_Sum", 1,0).over(Window.orderBy($"cumulative_Sum".asc))).select("initial_state", "prob_interval", "cumulative_Sum")

  val probability_intervals = df_2.selectExpr("initial_state", "(prob_interval, cumulative_Sum) as prob_interval")
  return probability_intervals
}

// COMMAND ----------

unconditional_distribution.write.mode("overwrite").parquet("dbfs:/roadSafety_no_date_CD")
