// Databricks notebook source
// MAGIC %md
// MAGIC ## Simulating the Arrival Times of a NHPP by Inversion 
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

import org.apache.spark.mllib.random._
import math.{log, floor, ceil}
import org.apache.spark.sql.functions._
import scala.util.{Try,Success,Failure}
import scala.util.control.Exception
import org.apache.spark.mllib.random.RandomRDDs._
import scala.collection.mutable.ArrayBuffer

// COMMAND ----------

// MAGIC %md 
// MAGIC - Load the arrival times of the events from 1 realization of the process 

// COMMAND ----------

val df = spark.read.parquet("FileStore/tables/LT_time_intervals").select("prev_date")

// COMMAND ----------

val ordered_T = df.collect().toArray :+ 1461

// COMMAND ----------

val generator = new UniformGenerator()
generator.setSeed(1234L) //set the seed for reproducability of results 

// COMMAND ----------

//initialization 
var i = 1
var u = generator.nextValue
var E = -math.log(1-u)
var T = 0.0
var m = 0.0
var width = 0.0
var samples = Array[Double]()
val n = 11720 //number of total observations 
val k = 1  //number of realisations

// COMMAND ----------

while (E < n/k){
    m = math.floor(((n+1)*k/n)*E)
    width = ordered_T(m.toInt+1).toString.replaceAll("\\[", "").replaceAll("\\]", "").toDouble - ordered_T(m.toInt).toString.replaceAll("\\[", "").replaceAll("\\]", "").toDouble
    T = ordered_T(m.toInt).toString.replaceAll("\\[", "").replaceAll("\\]", "").toDouble + width * (((n+1)*k/n)*E - m).toDouble
    samples = samples :+ T
    i += 1 
    u = generator.nextValue
    E -= math.log(1-u)
}

// COMMAND ----------

val arrival_samples = sc.parallelize(samples)
val rounded_arrivals = arrival_samples.map(item => math.ceil(item))
val sample_df = rounded_arrivals.toDF("day").groupBy("day").count.orderBy($"day".asc)

// COMMAND ----------

sample_df.count() //number of simulated days 
sample_df.select(sum("count")).show() //number of simulated events

// COMMAND ----------

val times = sample_df 
val initialisation = sc.parallelize(Seq((" ", 0.0))).toDF("initial_state", "time_unit")

// COMMAND ----------

val time_day_map = spark.sql("SELECT sequence(to_date('2017-01-01'), to_date('2020-12-31'), interval 1 day) as dates").select(explode($"dates").alias("day_of_year"), (monotonically_increasing_id + 1).alias("time_unit"))
val initial = ArrayBuffer[(String, Double)]()
val times_list = times.collect()
for (time <-  times_list){
    val day = time_day_map.filter(col("time_unit") === time(0)).select("day_of_year").collect()(0)(0).toString
    val count = time(1).asInstanceOf[Long]
    try {val conditional_distribution = spark.read.parquet("dbfs:/roadSafety_" + day + "_CD").select($"initial_state", $"prob_interval._1".alias("start"), $"prob_interval._2".alias("end"))
         val uniform_samples = uniformRDD(sc,count).toDF()
         val cross_samples_intervals = uniform_samples.crossJoin(conditional_distribution)
         val samples = cross_samples_intervals.filter("start < value").filter("end >= value").select("initial_state").cache()
         val location_time = samples.rdd.map(item => (item(0).toString, time(0).asInstanceOf[Double])).collect()
         initial ++= location_time
         samples.unpersist
         println(time(0).toString)
        }
    catch {
      case u: org.apache.spark.sql.AnalysisException => {
        println("Path does not exist " + day + ". Sampling independent of time")
        val conditional_distribution = spark.read.parquet("dbfs:/roadSafety_no_date_CD").select($"initial_state", $"prob_interval.prob_interval".alias("start"), $"prob_interval.cumulative_Sum".alias("end"))
        val uniform_samples = uniformRDD(sc,count).toDF()
        val cross_samples_intervals = uniform_samples.crossJoin(conditional_distribution)
        val samples = cross_samples_intervals.filter("start < value").filter("end >= value").select("initial_state").cache()
        val location_time = samples.rdd.map(item => (item(0).toString, time(0).asInstanceOf[Double])).collect()
        initial ++= location_time
        samples.unpersist
        println(time(0).toString)
        
      }
    }  
}

// COMMAND ----------

val location_simulation = initial.toDF("simulated_location", "arrival_time")

// COMMAND ----------

location_simulation.show(3)

// COMMAND ----------


