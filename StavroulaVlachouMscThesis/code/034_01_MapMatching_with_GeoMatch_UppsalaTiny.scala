// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Map-matching OpenStreetMap Nodes to OpenStreetMap Ways 
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

// MAGIC %md 
// MAGIC ## What is map-matching?
// MAGIC Map matching is the problem of how to match recorded geographic coordinates to a logical model of the real world, typically using some form of Geographic Information System.
// MAGIC 
// MAGIC See [https://en.wikipedia.org/wiki/Map_matching](https://en.wikipedia.org/wiki/Map_matching).

// COMMAND ----------

// MAGIC %md
// MAGIC ## Map-Matching with GeoMatch
// MAGIC [GeoMatch](https://ieeexplore.ieee.org/document/8622488) is a novel, scalable, and efficient big-data pipeline for large-scale map-matching on Apache Spark. It improves existing spatial big data solutions by utilizing a novel spatial partitioning scheme inspired by Hilbert space-filling curves.
// MAGIC 
// MAGIC The library can be found in the following git repository [GeoMatch](https://github.com/bdilab/GeoMatch).
// MAGIC 
// MAGIC The necessary files to generate the jar for this work can be found in the following fork [https://github.com/StavroulaVlachou/GeoMatch](https://github.com/StavroulaVlachou/GeoMatch). 
// MAGIC 
// MAGIC Read [GeoMatch: Efficient Large-Scale Map Matching on Apache Spark](https://eprints.lancs.ac.uk/id/eprint/129165/1/GeoMatch_IEEE_BigData_preprint.pdf)

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Instructions 
// MAGIC `git clone git@github.com:StavroulaVlachou/GeoMatch.git`
// MAGIC 
// MAGIC `cd Common`
// MAGIC 
// MAGIC `mvn compile install`
// MAGIC 
// MAGIC `cd ../GeoMatch`
// MAGIC 
// MAGIC `mvn compile install`
// MAGIC 
// MAGIC The generated `jar` files can be found within the `target` directories. Then, 
// MAGIC 1. In Databricks choose Create -> Library and upload the packaged jars.
// MAGIC 2. Create a Spark 2.4.0 - Scala 2.11 cluster with the uploaded GeoMatch library installed or if you are alreadt running a cluster and installed the uploaded library to it you have to detach and re-attache any notebook currently using that cluster. 

// COMMAND ----------

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.serializer.KryoSerializer
import org.cusp.bdi.gm.GeoMatch
import org.cusp.bdi.gm.geom.GMPoint
import org.cusp.bdi.gm.geom.GMLineString

// COMMAND ----------

import crosby.binary.osmosis.OsmosisReader

import org.apache.hadoop.mapreduce.{TaskAttemptContext, JobContext}
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

import org.openstreetmap.osmosis.core.container.v0_6.EntityContainer
import org.openstreetmap.osmosis.core.domain.v0_6._
import org.openstreetmap.osmosis.core.task.v0_6.Sink

import sqlContext.implicits._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import scala.collection.JavaConversions._
import org.apache.spark.graphx._
import magellan.Point



// COMMAND ----------

// MAGIC %fs ls /datasets/osm/uppsala

// COMMAND ----------

// MAGIC %md 
// MAGIC - Run the following command only once per cluster

// COMMAND ----------

// MAGIC %sh 
// MAGIC java -jar /dbfs/FileStore/jars/2706d711_3963_4d88_92e7_a8870d0164d1-osm_parquetizer_1_0_1_SNAPSHOT-80d25.jar /dbfs/datasets/osm/uppsala/uppsalaTinyR.pbf

// COMMAND ----------

// MAGIC %sh
// MAGIC ls /dbfs/datasets/osm/uppsala/

// COMMAND ----------

spark.conf.set("spark.sql.parquet.binaryAsString", true)

val nodes_df = spark.read.parquet("dbfs:/datasets/osm/uppsala/uppsalaTinyR.pbf.node.parquet")
val ways_df = spark.read.parquet("dbfs:/datasets/osm/uppsala/uppsalaTinyR.pbf.way.parquet")

// COMMAND ----------

val allowableWays = Seq(
  "motorway",
  "motorway_link",
  "trunk",
  "trunk_link",
  "primary",
  "primary_link",
  "secondary",
  "secondary_link",
  "tertiary",
  "tertiary_link",
  "living_street",
  "residential",
  "road",
  "construction",
  "motorway_junction"
)

// COMMAND ----------

//convert the nodes to Dataset containing the fields of interest

case class NodeEntry(nodeId: Long, latitude: Double, longitude: Double, tags: Seq[String])

val nodeDS = nodes_df.map(node => 
  NodeEntry(node.getAs[Long]("id"),
       node.getAs[Double]("latitude"),
       node.getAs[Double]("longitude"),
       node.getAs[Seq[Row]]("tags").map{case Row(key:String, value:String) => value}
)).cache()

// COMMAND ----------

//convert the ways to Dataset containing the fields of interest

case class WayEntry(wayId: Long, tags: Array[String], nodes: Array[Long])

val wayDS = ways_df.flatMap(way => {
        val tagSet = way.getAs[Seq[Row]]("tags").map{case Row(key:String, value:String) =>  value}.toArray
        if (tagSet.intersect(allowableWays).nonEmpty ){
            Array(WayEntry(way.getAs[Long]("id"),
            tagSet,
            way.getAs[Seq[Row]]("nodes").map{case Row(index:Integer, nodeId:Long) =>  nodeId}.toArray
            ))
        }
        else { Array[WayEntry]()}
}
).cache()

// COMMAND ----------

val distinctNodesWays = wayDS.flatMap(_.nodes).distinct //the distinct nodes within the ways 

// COMMAND ----------

val wayNodes = nodeDS.as("nodes") //nodes that are in a way + nodes info from nodeDS
  .joinWith(distinctNodesWays.as("ways"), $"ways.value" === $"nodes.nodeId")
  .map(_._1).cache

// COMMAND ----------

import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.functions.concat_ws
import org.apache.spark.sql.functions._

val nodes = wayDS.
  select($"wayId", $"nodes").
  withColumn("node", explode($"nodes")).
  drop("nodes")
val wayNodesLocated = nodes.join(wayNodes, wayNodes.col("nodeId") === nodes.col("node")).select($"wayId", $"node", $"latitude", $"longitude").groupBy("wayId").agg(collect_list(concat($"latitude",lit(" "), $"longitude")).alias("list_of_coordinates")).withColumn("coordinates_str", concat_ws("," ,col("list_of_coordinates"))).drop("list_of_coordinates")
wayNodesLocated.show(1, false)

// COMMAND ----------

import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

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
  
    val initial_point = new Point(lon.toDouble, lat.toDouble, SpatialReference.create(4326)) //WGS84
    val reprojection = GeometryEngine.project(initial_point, SpatialReference.create(3035))  //European Grid
    reprojection.toString
}
spark.udf.register("project_to_meters", project_to_meters(_:String, _:String):String)

// COMMAND ----------

val ways_reprojected = wayNodesLocated.rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(line => {val parts = line.replaceAll("\"","").split(",");val arrCoords = parts.slice(1,parts.length).map(xyStr => {val xy = xyStr.split(' ');val reprojection = project_to_meters(xy(1).toString, xy(0).toString);val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});("LineString"+" " +parts(0).toString, arrCoords)})
val waysDF = ways_reprojected.toDF("LineStringId","coords")
val ways_unpacked = waysDF.select(col("LineStringId"),concat_ws(",",col("coords"))).rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]",""))
ways_unpacked.take(1)

// COMMAND ----------

val rddFirst = ways_unpacked.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMLineString(parts(0), arrCoords)}))

// COMMAND ----------

rddFirst.take(1)

// COMMAND ----------

val rddFirstSet = sc.textFile("FileStore/tables/UUways.csv").mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMLineString(parts(0), arrCoords)}))

// COMMAND ----------

rddFirstSet.take(1)

// COMMAND ----------

rddFirstSet.count() //9 ways 

// COMMAND ----------

val rddSecondSet = sc.textFile("FileStore/tables/UUnodes.csv").mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").split(',');new GMPoint(parts(0), (parts(1).toDouble.toInt, parts(2).toDouble.toInt))}))

// COMMAND ----------

rddSecondSet.take(1)

// COMMAND ----------

rddSecondSet.count() //626 nodes to be map-matched 

// COMMAND ----------

val geoMatch = new GeoMatch(false, 16, 150, (-1, -1, -1, -1)) //n(=dimension of the Hilber curve) should be a power of 2. 


// COMMAND ----------

val resultRDD = geoMatch.spatialJoinKNN(rddFirst, rddSecondSet, 1, false)


// COMMAND ----------

resultRDD.filter(element => (element._2.isEmpty)).count()  //number of nodes that are not matched successfully

// COMMAND ----------

resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty)).toDF("pointId", "matchId").show(5, false)

// COMMAND ----------


