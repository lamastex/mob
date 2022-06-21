// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Segmentation of Lithuania by municipalities using Magellan
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

// MAGIC %md
// MAGIC ## Instructions ##
// MAGIC 1. Clone the Magellan repository from [https://github.com/rahulbsw/magellan.git](https://github.com/rahulbsw/magellan.git).
// MAGIC 1. Build the jar and get it into your local machine. 
// MAGIC 3. In Databricks choose *Create -> Library* and upload the packaged jar.
// MAGIC 4. Create a spark 2.4.5 Scala 2.11 cluster with the uploaded Magellan library installed or if you are already running a cluster and installed the uploaded library to it you have to detach and re-attach any notebook currently using that cluster.

// COMMAND ----------

import magellan.Point 
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.magellan.dsl.expressions._
val toPointUDF = udf{(x:Double,y:Double) => Point(x,y) }

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Car Accidents in Lithuania

// COMMAND ----------

// MAGIC %md 
// MAGIC After downloading the data (see lasts cells of the notebook), we expect to have the following files in distributed file system (dbfs):
// MAGIC 
// MAGIC * ```LTcar_reprojected.csv``` is the file with the data crashes from LT.
// MAGIC * ```municipalities.geojson``` is the geojson file containing LT municipalities.

// COMMAND ----------

// MAGIC %md 
// MAGIC First five lines or rows of the crash data containing: ID, Lon, Lat, timestamp

// COMMAND ----------

//sc.textFile("dbfs:/datasets/magellan/LTcar_reprojected.csv").take(1).foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC id,latitude,longitude,timestamp
// MAGIC LT20xyABCDEF,55.xxxxxx,21.yyyyyy,20xy-mm-dd hh:20:00.000+01:00
// MAGIC ```

// COMMAND ----------

case class CrashRecord(id: String, timestamp: String, point: Point)

// COMMAND ----------

// MAGIC %md
// MAGIC Load accident data and transform latitude and longitude to Magellan's Point

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
val crashes = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/datasets/magellan/LTcar_reprojected.csv").toDF()
val crashes_with_points = crashes.select(col("id"), col("timestamp"), col("longitude").cast(DoubleType), col("latitude").cast(DoubleType)).withColumn("point", toPointUDF($"longitude", $"latitude")).drop("latitude", "longitude").filter(col("timeStamp").isNotNull.as[CrashRecord])

// COMMAND ----------

//crashes.show(1)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+---------+---------+-------------------+
// MAGIC |          id| latitude|longitude|          timestamp|
// MAGIC +------------+---------+---------+-------------------+
// MAGIC |LT20xyABCDEF|55.xxxxxx|21.yyyyyy|20xy-mm-dd hh:20:00|
// MAGIC ```

// COMMAND ----------

//crashes_with_points.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+-------------------+---------------------------+
// MAGIC |id          |timestamp          |point                      |
// MAGIC +------------+-------------------+---------------------------+
// MAGIC |LT20xyABCDEF|20xy-mm-dd hh:20:00|Point(21.yyyyyy, 55.xxxxxx)|
// MAGIC ```

// COMMAND ----------

val crashRecordCount = crashes_with_points.count() // how many crash records?

// COMMAND ----------

// MAGIC %md
// MAGIC The geojson format can spatially describe vector features: `points`, `lines`, and `polygons`, representing, for example, water wells, rivers, and lakes. Each item usually has `attributes` that describe it, such as name or temperature.

// COMMAND ----------

def frameIt( u:String, h:Int ) : String = {
      """<iframe 
 src=""""+ u+""""
 width="95%" height="""" + h + """"
 sandbox>
  <p>
    <a href="http://spark.apache.org/docs/latest/index.html">
      Fallback link for browsers that, unlikely, don't support frames
    </a>
  </p>
</iframe>"""
   }
displayHTML(frameIt("https://en.wikipedia.org/wiki/GeoJSON", 400))

// COMMAND ----------

// MAGIC %md
// MAGIC The name of the municipality in the metadata is "name" so let's keep only that one.

// COMMAND ----------

val municipalities = sqlContext.read.format("magellan")
                                   .option("type", "geojson")
                                   .load("dbfs:/datasets/magellan/municipalities.geojson")
                                   .filter($"polygon".isNotNull)
                                   .select($"polygon", $"metadata"("name") as "municipality")

// COMMAND ----------

municipalities.count()

// COMMAND ----------

municipalities.show(100)

// COMMAND ----------

//If we have the same coordinates system, next cell should not be empty
//The geojson file are presented in the WGS84 coordinate system

// COMMAND ----------

// MAGIC %md
// MAGIC Join the accidents with the municipalities.

// COMMAND ----------

val joined = municipalities
            .join(crashes_with_points)
            .where($"point" within $"polygon")
            .select($"id", $"timestamp", $"municipality", $"point")


// COMMAND ----------

//joined.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+-------------------+--------------------+---------------------------+
// MAGIC |id          |timestamp          |municipality        |point                      |
// MAGIC +------------+-------------------+--------------------+---------------------------+
// MAGIC |LT20xyABCDEF|2019-09-08 20:10:00|Visagino savivaldybė|Point(26.xxxxxx, 55.yyyyy) |
// MAGIC 
// MAGIC ```

// COMMAND ----------

val crashes_in_municipalities = joined.count() 

// COMMAND ----------

crashRecordCount - crashes_in_municipalities // records not in the neighbourhood geojson file

// COMMAND ----------

val municipality_count = joined
  .groupBy($"municipality")
  .agg(countDistinct("id").as("acc_count"))
  .orderBy(col("acc_count").desc)

municipality_count.show(5,false)

// COMMAND ----------

val municipality_count_freq = municipality_count.withColumn("frequency", col("acc_count")/crashes_in_municipalities)
municipality_count_freq.show(10,false)

// COMMAND ----------

// MAGIC %md
// MAGIC ####Save the frequency of accidents in each municipality

// COMMAND ----------

municipality_count_freq.select("municipality","frequency").write.format("csv").option("header", true).save("dbfs:/datasets/lithuania/municipalities_freq.csv")

// COMMAND ----------

// MAGIC %md
// MAGIC ###Use the municipalities' population to normalize the accidents.

// COMMAND ----------

// MAGIC %md
// MAGIC Download most updated population data from https://www.registrucentras.lt/p/853 and upload it

// COMMAND ----------

val municipality_pop = spark.read.format("csv").option("delimiter",";").option("header", "true").option("inferSchema", "true").load("dbfs:/datasets/lithuania/population.csv").toDF()

// COMMAND ----------

municipality_pop.show()

// COMMAND ----------

val municipality_count_pop = municipality_count.join(municipality_pop, municipality_count.col("municipality") === municipality_pop.col("municipality")).withColumn("acc_by_pop", col("acc_count")/col("population")).drop(municipality_pop.col("municipality"))

// COMMAND ----------

municipality_count_pop.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ####Save the accidents normalized by population

// COMMAND ----------

municipality_count_pop.select("municipality","acc_by_pop").write.format("csv").option("header", true).save("dbfs:/datasets/lithuania/municipalities_pop.csv")

// COMMAND ----------

// MAGIC %md 
// MAGIC # Step 0: Downloading datasets and load into dbfs
// MAGIC 
// MAGIC * get the accident data
// MAGIC * get the Lithuanian municipality data

// COMMAND ----------

// MAGIC %md
// MAGIC ###Getting crash data 
// MAGIC ####(This only needs to be done once per shard!)

// COMMAND ----------

dbutils.fs.cp("dbfs:/FileStore/tables/ltcar_reprojected.csv", "dbfs:/datasets/magellan/LTcar_reprojected.csv")

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/datasets/magellan/"))

// COMMAND ----------

// MAGIC %md 
// MAGIC ###Getting Lithuanian Administrative Divisions Data
// MAGIC 
// MAGIC Second-level Administrative Divisions, Lithuania, 2015
// MAGIC 
// MAGIC Data from https://github.com/seporaitis/lt-geojson

// COMMAND ----------

// MAGIC %sh
// MAGIC wget https://raw.githubusercontent.com/seporaitis/lt-geojson/master/geojson/municipalities.geojson

// COMMAND ----------

// MAGIC %python
// MAGIC # Reading and processing geojson. Removing @relations (fails for some reason and not needed)
// MAGIC 
// MAGIC import json
// MAGIC 
// MAGIC # municipalities / Savivaldybės
// MAGIC municipalities = json.load(open("municipalities.geojson", 'r'))
// MAGIC 
// MAGIC list_to_remove = []
// MAGIC i = 0
// MAGIC for feature in municipalities['features']:
// MAGIC   municipalities['features'][i]["properties"].pop("relations", None)
// MAGIC   municipalities['features'][i]["properties"].pop("@relations", None)
// MAGIC 
// MAGIC   i+=1
// MAGIC   
// MAGIC for feature in municipalities['features']:
// MAGIC   for property in feature["properties"]:
// MAGIC     print(property)
// MAGIC     
// MAGIC with open("municipalities.geojson", 'w') as outfile:
// MAGIC     json.dump(municipalities, outfile)

// COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/municipalities.geojson", "dbfs:/datasets/magellan/")

// COMMAND ----------

// MAGIC %md 
// MAGIC ### End of Step 0: downloading and putting data in dbfs
