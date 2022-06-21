// Databricks notebook source
// MAGIC %md
// MAGIC ## Map-Matching Events on a State Space / Coarsened Road Graph with GeoMatch 
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

import org.apache.spark.graphx._
import sqlContext.implicits._
import scala.collection.JavaConversions._
import org.cusp.bdi.gm.GeoMatch
import org.cusp.bdi.gm.geom.GMPoint
import org.cusp.bdi.gm.geom.GMLineString
import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

// COMMAND ----------

// MAGIC %md
// MAGIC ### Road Network

// COMMAND ----------

spark.conf.set("spark.sql.parquet.binaryAsString", true)

val nodes_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.node.parquet")
val ways_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.way.parquet")

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

// MAGIC %md
// MAGIC The first step is to obtain the state space. The State Space consists of road segments and intersection points. The road segments correspond to the edges of the graph while the intersection points can be retrieved from the ways and the nodes dataset as those nodes that lie in at least one way. All coordinates should be in the spatial reference system 3035. To implement the map matching it is better to keep all intermediate points from each edge. 

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/LT"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Obtaining the State Space

// COMMAND ----------

// MAGIC %md
// MAGIC #### Intersection points

// COMMAND ----------

val intersections = spark.read.parquet("dbfs:/LT/intersections")
intersections.show(1)

// COMMAND ----------

intersections.count

// COMMAND ----------

// MAGIC %md
// MAGIC The next step is to obtain the coordinates of the intersection points and convert them into decimal degrees. 

// COMMAND ----------

val intersection_points = nodeDS.join(intersections, intersections("intersectionNode") === nodeDS("nodeId")).drop("tags", "nodeId").select("intersectionNode", "latitude", "longitude")
intersection_points.show(1)

// COMMAND ----------

intersection_points.count()

// COMMAND ----------

import org.apache.spark.sql.functions.{concat, lit}
val concat_coordinates = intersection_points.select($"intersectionNode",concat($"latitude",lit(" "),$"longitude").alias("coordinates"))
concat_coordinates.show(1, false)

// COMMAND ----------

val firstIntersectionStates = concat_coordinates.select(concat(lit("LineString:"),$"intersectionNode").alias("LineString"),$"coordinates")
firstIntersectionStates.show(1, false)
val firstIntersectionStates_rdd = firstIntersectionStates.rdd
firstIntersectionStates_rdd.take(1)

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

val intersections_reprojected = firstIntersectionStates_rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(line => {val parts = line.replaceAll("\"","").split(",");val arrCoords = parts.slice(1,parts.length).map(xyStr => {val xy = xyStr.split(" ");val reprojection = project_to_meters(xy(1).toString, xy(0).toString);val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});(parts(0).toString, arrCoords)})

// COMMAND ----------

intersections_reprojected.take(1)

// COMMAND ----------

val intersections_unpacked = intersections_reprojected.map(item => item._1.toString + "," + item._2(0).toString)
intersections_unpacked.take(1)

// COMMAND ----------

val rdd_first_set_intersections = intersections_unpacked.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMPoint(parts(0), arrCoords(0))}))


// COMMAND ----------

rdd_first_set_intersections.take(1)

// COMMAND ----------

// MAGIC %md
// MAGIC Next, we need to obtain the set of points that are to be map matched. In this case the set of points corresponds to the accident events occuring in LT.

// COMMAND ----------

val events = spark.read.format("csv").load("/FileStore/tables/LTnodes.csv").rdd.map(line => line.toString)

// COMMAND ----------

// MAGIC %md 
// MAGIC ```events.take(1)```
// MAGIC 
// MAGIC Output:
// MAGIC 
// MAGIC ```Array([Point LT2019XXX,52aaa.18bbb,36ccc.21ddd])```

// COMMAND ----------

val all_accidents = spark.read.format("csv").load("FileStore/tables/LTnodes.csv").toDF("PointId", "longitude", "latitude")

// COMMAND ----------

val rddSecondSet = events.mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").replaceAll("\\[","").replaceAll("\\]","").split(',');new GMPoint(parts(0), (parts(1).toDouble.toInt, parts(2).toDouble.toInt))}))

// COMMAND ----------

// MAGIC %md
// MAGIC Implement Map Matching

// COMMAND ----------

val geoMatch = new GeoMatch(false, 256, 20, (-1, -1, -1, -1)) //n(=dimension of the Hilber curve) should be a power of 2. 

// COMMAND ----------

val resultRDD = geoMatch.spatialJoinKNN(rdd_first_set_intersections, rddSecondSet, 1, false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +----------------------------------------+---------------------------------------------+
// MAGIC |k                                       |line                                         |
// MAGIC +----------------------------------------+---------------------------------------------+
// MAGIC |[Point LT20xyABCDEF, [521xxxx, 362yyyy]]|[[LineString:1254578sss, [521zzzz, 362zzzz]]]|
// MAGIC +----------------------------------------+---------------------------------------------+
// MAGIC only showing top 1 rows
// MAGIC ```

// COMMAND ----------

resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => (element._2.isEmpty)).count()

// COMMAND ----------

resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty)).count()

// COMMAND ----------

val unmatched_events = resultRDD.filter(element => (element._2.isEmpty)).map(element => element._1.payload).toDF("id")

val second_set_second_round = unmatched_events.join(all_accidents, unmatched_events("id") === all_accidents("PointId")).drop("id").rdd.map(line => line.toString)

val rddSecondSetSecondRound = second_set_second_round.mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").replaceAll("\\[","").replaceAll("\\]","").split(',');new GMPoint(parts(0), (parts(1).toDouble.toInt, parts(2).toDouble.toInt))}))

// COMMAND ----------

val edges = spark.read.parquet("dbfs:/_checkpoint/edges_LT_100")
val vertices = spark.read.parquet("dbfs:/_checkpoint/vertices_LT_100").toDF("vertexId", "latitude", "longitude")

// COMMAND ----------

edges.show(1)

// COMMAND ----------

val src_coordinates = edges.join(vertices,vertices("vertexId") === edges("src"), "left_outer").drop("vertexId").withColumnRenamed("latitude", "src_latitude").withColumnRenamed("longitude","src_longitude")
val edge_coordinates = src_coordinates.join(vertices,vertices("vertexId") === src_coordinates("dst")).drop("vertexId").withColumnRenamed("latitude", "dst_latitude").withColumnRenamed("longitude", "dst_longitude")

// COMMAND ----------

import org.apache.spark.sql.functions.{concat, lit}
val concat_coordinates = edge_coordinates.select($"src",concat($"src_latitude",lit(" "),$"src_longitude").alias("src_coordinates"), $"dst",concat($"dst_latitude",lit(" "),$"dst_longitude").alias("dst_coordinates"))

// COMMAND ----------

concat_coordinates.show(1, false)

// COMMAND ----------

val linestring_coordinates = concat_coordinates.select($"src", $"dst",concat($"src_coordinates", lit(","), $"dst_coordinates").alias("list_of_coordinates"))

// COMMAND ----------

linestring_coordinates.show(1, false)

// COMMAND ----------

val first = linestring_coordinates.select(concat(lit("LineString:"),$"src",lit("+"), $"dst").alias("LineString"),$"list_of_coordinates")

// COMMAND ----------

val first_rdd = first.rdd

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
  
    val initial_point = new Point(lon.toDouble, lat.toDouble, SpatialReference.create(4326))
    val reprojection = GeometryEngine.project(initial_point, SpatialReference.create(3035))
    reprojection.toString
}
spark.udf.register("project_to_meters", project_to_meters(_:String, _:String):String)

// COMMAND ----------

first_rdd.take(1)

// COMMAND ----------

val ways_reprojected = first_rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(line => {val parts = line.replaceAll("\"","").split(",");val arrCoords = parts.slice(1,parts.length).map(xyStr => {val xy = xyStr.split(" ");val reprojection = project_to_meters(xy(1).toString, xy(0).toString);val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});(parts(0).toString,arrCoords)})

// COMMAND ----------

ways_reprojected.take(1)

// COMMAND ----------

ways_reprojected.map(item => item._2(1)).take(1)

// COMMAND ----------

val ways_unpacked = ways_reprojected.map(item => item._1.toString + "," + item._2(0).toString + "," + item._2(1).toString)

// COMMAND ----------

val rdd_first_set = ways_unpacked.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMLineString(parts(0), arrCoords)}))

// COMMAND ----------

rdd_first_set.count()

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

val geoMatchSecond = new GeoMatch(false, 256, 200, (-1, -1, -1, -1)) //n(=dimension of the Hilber curve) should be a power of 2. 

// COMMAND ----------

val resultRDDsecond = geoMatchSecond.spatialJoinKNN(rdd_first_set, rddSecondSetSecondRound, 1, false)

// COMMAND ----------

resultRDDsecond.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => (element._2.isEmpty)).count()

// COMMAND ----------

// MAGIC %md
// MAGIC The next step is for each state to obtain the count 

// COMMAND ----------

val res = resultRDDsecond.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty))

// COMMAND ----------

val res_df = res.map(element => (element._1, element._2(0))).toDF("PointId", "State")

// COMMAND ----------

val edge_counts = res_df.groupBy("State").count

// COMMAND ----------

// MAGIC %md
// MAGIC ```edge_counts.show(2, false)```
// MAGIC 
// MAGIC Output:
// MAGIC ```
// MAGIC +--------------------------------+-----+
// MAGIC |State                           |count|
// MAGIC +--------------------------------+-----+
// MAGIC |LineString:469327286+3637433937 |a    |
// MAGIC |LineString:2488853231+272553182 |b    |
// MAGIC |LineString:5074963276+2221962222|c    |
// MAGIC +--------------------------------+-----+
// MAGIC ```

// COMMAND ----------

val res1 = resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty))
val res1_df = res1.map(element => (element._1, element._2(0))).toDF("PointId", "State")
val intersection_counts = res1_df.groupBy("State").count

// COMMAND ----------

import org.apache.spark.sql.functions._
val state_counts = edge_counts.union(intersection_counts)
state_counts.agg(sum("count")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Find the states with no matched events, assign count value equal to 0 and union them with the rest of the states_counts 

// COMMAND ----------

val all_intersection_states = rdd_first_set_intersections.toDF("stateId", "coords").drop("coords")
val all_edge_states = rdd_first_set.toDF("stateId", "coords").drop("coords")
val all_states = all_intersection_states.union(all_edge_states)
all_states.count

// COMMAND ----------

val s1 = all_states.join(state_counts, all_states("stateId") === state_counts("State"), "left_outer").drop("State")
val s_final = s1.na.fill(0)

// COMMAND ----------


