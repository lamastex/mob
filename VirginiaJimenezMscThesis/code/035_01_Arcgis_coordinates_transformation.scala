// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Transformation of coordinates using Arcgis Runtime library
// MAGIC 
// MAGIC Virginia Jimenez Mohedano ([LinkedIn](https://www.linkedin.com/in/virginiajimenezmohedano/)), Stavroula Rafailia Vlachou ([LinkedIn](https://www.linkedin.com/in/stavroula-rafailia-vlachou/)) and Raazesh Sainudiin ([LinkedIn](https://www.linkedin.com/in/raazesh-sainudiin-45955845/)).
// MAGIC 
// MAGIC ```
// MAGIC This project was supported by UAB SENSMETRY through a Data Science Thesis Internship 
// MAGIC between 2022-01-17 and 2022-06-05 to Stavroula R. Vlachou and Virginia J. Mohedano 
// MAGIC and Databricks University Alliance with infrastructure credits from AWS to 
// MAGIC Raazesh Sainudiin, Department of Mathematics, Uppsala University, Sweden.
// MAGIC ```
// MAGIC 
// MAGIC 2022, Uppsala, Sweden

// COMMAND ----------

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql._ 
import scala.util.matching.Regex

import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

// COMMAND ----------

// MAGIC %md
// MAGIC ####Run the next cells just once per cluster to install the library

// COMMAND ----------

// MAGIC %md 
// MAGIC Arcgis runtime library allows for coordinates transformations.

// COMMAND ----------

// MAGIC %md
// MAGIC - Download arcgis runtime from https://developers.arcgis.com/downloads/#java (.tgz)
// MAGIC 
// MAGIC - Install jar (from the "libs" folder) in the cluster.
// MAGIC 
// MAGIC The version downloaded was 100.4.0

// COMMAND ----------

dbutils.fs.mkdirs("dbfs:/arcGISRuntime/")

// COMMAND ----------

// MAGIC %sh 
// MAGIC tar zxvf /dbfs/arcGISRuntime/arcgis_runtime_sdk_java_100_4_0.tgz -C /dbfs/arcGISRuntime

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/"))

// COMMAND ----------

// MAGIC %md
// MAGIC ####Now the library is installed, the transformation can be performed

// COMMAND ----------

// MAGIC %md
// MAGIC The library needs to be initialized running the following cell

// COMMAND ----------

if(!ArcGISRuntimeEnvironment.isInitialized())
    {
      ArcGISRuntimeEnvironment.setInstallDirectory("/dbfs/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/")
      ArcGISRuntimeEnvironment.initialize() 
    }

// COMMAND ----------

// MAGIC %md
// MAGIC Read the data that needs to be transformed: in this case, osm location data is transformed.

// COMMAND ----------

spark.conf.set("spark.sql.parquet.binaryAsString", true)

val nodes_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.node.parquet")

// COMMAND ----------

nodes_df.count()

// COMMAND ----------

nodes_df.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC In this case, the coordinates are expressed in the WGS84 system and they will be projected into meters to be used with GeoMatch. To do this, one just need to change the code for each of the reference systems in the next function.

// COMMAND ----------

def project_to_meters(lon: Double, lat: Double): String = { 
    
    if(!ArcGISRuntimeEnvironment.isInitialized())
    {
      ArcGISRuntimeEnvironment.setInstallDirectory("/dbfs/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/")
      ArcGISRuntimeEnvironment.initialize() 
    }
  
    val initial_point = new Point(lon, lat, SpatialReference.create(4326))
    val reprojection = GeometryEngine.project(initial_point, SpatialReference.create(3035))
    reprojection.toString
}
spark.udf.register("project_to_meters", project_to_meters(_:Double, _:Double):String)

// COMMAND ----------

val nodes_converted = nodes_df.selectExpr("id","latitude", "longitude", "project_to_meters(longitude, latitude) as new_coord")
nodes_converted.show(5,false)

// COMMAND ----------

// MAGIC %md
// MAGIC Once the transformation is done, it is necessary to unpack the coordinates as follow

// COMMAND ----------

def unpack_lat(str: String): String = {
        val lat = str.replaceAll(",","").replaceAll("\\[","").split(" ")(2)
        return lat
}
spark.udf.register("unpack_lat", unpack_lat(_:String): String)

def unpack_lon(str: String): String = {
        val lon = str.replaceAll(",","").replaceAll("\\[","").split(" ")(1)
        return lon
}
spark.udf.register("unpack_lon", unpack_lon(_:String): String)

// COMMAND ----------

val new_coordinates = nodes_converted.selectExpr("id as node_id", "unpack_lat(new_coord) as reprojected_lat", "unpack_lon(new_coord) as reprojected_lon")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, the new coordinates are expressed in meters.

// COMMAND ----------

new_coordinates.show(5,false)

// COMMAND ----------

val nodes_new_coordinates = nodes_df.join(new_croordinates, nodes_df.col("id") === new_coordinates.col("node_id")).selectExpr("id", "version", "timestamp", "changeset", "uid", "user_sid", "tags", "reprojected_lat as latitude", "reprojected_lon as longitude")

// COMMAND ----------

nodes_new_coordinates.write.parquet("dbfs:/datasets/osm/lithuania/lithuania_nodes_converted.parquet")
