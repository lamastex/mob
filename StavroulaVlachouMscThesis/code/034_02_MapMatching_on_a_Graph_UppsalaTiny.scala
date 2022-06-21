// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC # Map-matching OpenStreetMap Nodes to Road Graph elements
// MAGIC 
// MAGIC Stavroula Rafailia Vlachou ([LinkedIn](https://www.linkedin.com/in/stavroula-rafailia-vlachou/)) and Raazesh Sainudiin ([LinkedIn](https://www.linkedin.com/in/raazesh-sainudiin-45955845/)).
// MAGIC 
// MAGIC ```
// MAGIC This project was supported by SENSMETRY through a Data Science Project Internship 
// MAGIC between 2022-01-17 and 2022-06-05 to Stavroula R. Vlachou
// MAGIC Databricks University Alliance with infrastructure credits from AWS to 
// MAGIC Raazesh Sainudiin, Department of Mathematics, Uppsala University, Sweden.
// MAGIC ```
// MAGIC 
// MAGIC 2022, Uppsala, Sweden

// COMMAND ----------

import org.apache.spark.graphx._
import sqlContext.implicits._
import scala.collection.JavaConversions._
import org.apache.spark.sql.functions.{concat, lit}


import org.cusp.bdi.gm.GeoMatch
import org.cusp.bdi.gm.geom.GMPoint
import org.cusp.bdi.gm.geom.GMLineString
import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

// COMMAND ----------

val edges = spark.read.parquet("dbfs:/graphs/uppsala/edges")
val vertices = spark.read.parquet("dbfs:/graphs/uppsala/vertices").toDF("vertexId", "latitude", "longitude")

// COMMAND ----------

val src_coordinates = edges.join(vertices,vertices("vertexId") === edges("src"), "left_outer").drop("vertexId").withColumnRenamed("latitude", "src_latitude").withColumnRenamed("longitude","src_longitude")
val edge_coordinates = src_coordinates.join(vertices,vertices("vertexId") === src_coordinates("dst")).drop("vertexId").withColumnRenamed("latitude", "dst_latitude").withColumnRenamed("longitude", "dst_longitude")

// COMMAND ----------

val concat_coordinates = edge_coordinates.select($"src",concat($"src_latitude",lit(" "),$"src_longitude").alias("src_coordinates"), $"dst",concat($"dst_latitude",lit(" "),$"dst_longitude").alias("dst_coordinates"))

// COMMAND ----------

val linestring_coordinates = concat_coordinates.select($"src", $"dst",concat($"src_coordinates", lit(","), $"dst_coordinates").alias("list_of_coordinates"))

// COMMAND ----------

val first = linestring_coordinates.select(concat(lit("LineString:"),$"src",lit("+"), $"dst").alias("LineString"),$"list_of_coordinates")

// COMMAND ----------

val first_rdd = first.rdd

// COMMAND ----------

if(!ArcGISRuntimeEnvironment.isInitialized())
    {
      ArcGISRuntimeEnvironment.setInstallDirectory("/dbfs/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/")
      ArcGISRuntimeEnvironment.initialize() 
    }

// COMMAND ----------

def project_to_meters(lon: String, lat: String): String = { 
    
    if(!ArcGISRuntimeEnvironment.isInitialized())
    {
      ArcGISRuntimeEnvironment.setInstallDirectory("/dbfs/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/")
      ArcGISRuntimeEnvironment.initialize() 
    }
  
    val initial_point = new Point(lon.toDouble, lat.toDouble, SpatialReference.create(4326))
    val reprojection = GeometryEngine.project(initial_point, SpatialReference.create(3035))
    reprojection.toString
}
spark.udf.register("project_to_meters", project_to_meters(_:String, _:String):String)

// COMMAND ----------

val ways_reprojected = first_rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(line => {val parts = line.replaceAll("\"","").split(",");val arrCoords = parts.slice(1,parts.length).map(xyStr => {val xy = xyStr.split(' ');val reprojection = project_to_meters(xy(1).toString, xy(0).toString);val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});(parts(0).toString, arrCoords)})

// COMMAND ----------

val ways_unpacked = ways_reprojected.map(item => item._1.toString + "," + item._2(0).toString + "," + item._2(1).toString)

// COMMAND ----------

val rdd_first_set = ways_unpacked.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMLineString(parts(0), arrCoords)}))

// COMMAND ----------

rdd_first_set.take(1)

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

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
val initial_points = vertices.toDF().select(col("vertexId").cast(StringType), col("latitude").cast(StringType), col("longitude").cast(StringType)).withColumn("Point", lit("Point "))
val reprojected_points = initial_points.selectExpr("concat(Point,vertexId) as PointId","project_to_meters(longitude, latitude) as reprojection")
val unpacked_reprojection = reprojected_points.selectExpr("PointId","unpack_lat(reprojection) as new_lat", "unpack_lon(reprojection) as new_lon").rdd

// COMMAND ----------

unpacked_reprojection.take(1)

// COMMAND ----------

val f = unpacked_reprojection.map(line => {val id = line(0).toString; val lat = line(1).toString; val lon = line(2).toString;id+"," + lat +","+ lon})

// COMMAND ----------

val rddSecondSet = f.mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").split(',');new GMPoint(parts(0), (parts(2).toDouble.toInt, parts(1).toDouble.toInt))}))

// COMMAND ----------

rddSecondSet.take(1)

// COMMAND ----------

val geoMatch = new GeoMatch(false, 16, 150, (-1, -1, -1, -1)) //n(=dimension of the Hilber curve) should be a power of 2. 

// COMMAND ----------

val resultRDD = geoMatch.spatialJoinKNN(rdd_first_set, rddSecondSet, 1, false)

// COMMAND ----------

resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty)).take(10)

// COMMAND ----------

resultRDD.toDF("k", "line").show(10, false)

// COMMAND ----------


