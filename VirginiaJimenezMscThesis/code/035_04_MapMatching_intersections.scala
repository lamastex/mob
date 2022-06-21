// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Map-matching the accident with their closest intersection and measuring the distance between them.
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

import org.apache.spark.graphx._
import sqlContext.implicits._
import scala.collection.JavaConversions._
import org.cusp.bdi.gm.GeoMatch
import org.cusp.bdi.gm.geom.GMPoint
import org.cusp.bdi.gm.geom.GMLineString
import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

import com.esri.core.geometry.GeometryEngine.geodesicDistanceOnWGS84
import com.esri.core.geometry.{Point => Points}

// COMMAND ----------

// MAGIC %md
// MAGIC In this notebook, we will find the intersections of the road network and extract their coordinates. Then, we load the accident data and use GeoMatch to match the accidents with their closest intersections. Later, when having the coordinates of both the intersections and the accidents, the distance between them is measured.

// COMMAND ----------

// MAGIC %md
// MAGIC ## What is GeoMatch
// MAGIC 
// MAGIC GeoMatch is a novel, scalable, and efficient big-data pipeline for large-scale map matching on Apache Spark.
// MAGIC 
// MAGIC Read [GeoMatch: Efficient Large-Scale Map Matching on Apache Spark](https://eprints.lancs.ac.uk/id/eprint/129165/1/GeoMatch_IEEE_BigData_preprint.pdf)
// MAGIC 
// MAGIC The project is open source and all the relevant code can be found on the [GeoMatch git repository](https://github.com/bdilab/GeoMatch).
// MAGIC 
// MAGIC The jar library needs to be build from the git repository and uploaded to the cluster. 

// COMMAND ----------

// MAGIC %md
// MAGIC ### Road Network

// COMMAND ----------

// MAGIC %md
// MAGIC First, we want to obtain the intersections.

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

case class NodeEntry(nodeId: Long, latitude: Double, longitude: Double, tags: Seq[String])

val nodeDS = nodes_df.map(node => 
  NodeEntry(node.getAs[Long]("id"),
       node.getAs[Double]("latitude"),
       node.getAs[Double]("longitude"),
       node.getAs[Seq[Row]]("tags").map{case Row(key:String, value:String) => value}
)).cache()

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

import org.apache.spark.sql.functions.explode

val nodeCounts = wayDS
                    .select(explode('nodes).as("node"))
                    .groupBy('node).count

val intersectionNodes = nodeCounts.filter('count >= 2).select('node.alias("intersectionNode"))
val intersections = intersectionNodes

// COMMAND ----------

intersections.show(10)
intersections.count()

// COMMAND ----------

// MAGIC %md
// MAGIC The next step is to obtain the coordinates of the intersection points and convert them into decimal degrees (needed for GeoMatch) 

// COMMAND ----------

val intersection_points = nodeDS.join(intersections, intersections("intersectionNode") === nodeDS("nodeId")).drop("tags", "nodeId").select("intersectionNode", "latitude", "longitude")

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

val intersections_reprojected = firstIntersectionStates_rdd.map(
                                line => 
                                line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(
                                      line => 
                                      {val parts = line.replaceAll("\"","").split(",");
                                       val arrCoords = parts.slice(1,parts.length).map(
                                          xyStr => 
                                          {val xy = xyStr.split(" ");
                                          val reprojection = project_to_meters(xy(1).toString, xy(0).toString);
                                          val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);
                                          val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});
                                       (parts(0).toString, arrCoords)})


// COMMAND ----------

intersections_reprojected.take(1)

// COMMAND ----------

val intersections_unpacked = intersections_reprojected.map(item => item._1.toString + "," + item._2(0).toString)
intersections_unpacked.take(1)

// COMMAND ----------

// MAGIC %md
// MAGIC The first rdd to be used in GeoMatch is formed with the intersections.

// COMMAND ----------

val rdd_first_set_intersections = intersections_unpacked.mapPartitions(_.map(line =>
                                                                             {val parts = line.replaceAll("\"","").split(',');
                                                                              val arrCoords = parts.slice(1, parts.length).map(xyStr => 
                                                                                                                               {val xy = xyStr.split(' ');
                                                                                                                                (xy(0).toDouble.toInt, xy(1).toDouble.toInt)});
                                                                              new GMPoint(parts(0), arrCoords(0))}))


// COMMAND ----------

rdd_first_set_intersections.take(1)

// COMMAND ----------

// MAGIC %md
// MAGIC Next, we need to obtain the set of points that are to be map matched. In this case the set of points corresponds to the accident events occuring in LT.

// COMMAND ----------

// MAGIC %md
// MAGIC The three events filtered are the ones corresponding to incorrect coordinates values so we dropped them. 

// COMMAND ----------

val events = spark.read.format("parquet").load("dbfs:/FileStore/tables/LT_accV.parquet").rdd//.filter(line => (line(0) != "LT20xyABCDEF") && (line(0) != "LT20xyABCDEF") && (line(0) != "LT20xyABCDEF")).map(line => line.toString)
events.take(1)
events.count()

// COMMAND ----------

val all_accidents = spark.read.format("parquet").load("dbfs:/FileStore/tables/LT_accV.parquet").toDF("PointId", "longitude", "latitude")

// COMMAND ----------

// MAGIC %md 
// MAGIC The second rdd is formed with the accidents.

// COMMAND ----------

val rddSecondSet = events.mapPartitions(_.map(line => 
                                              {new GMPoint(line.getString(0), (line.getString(1).toDouble.toInt, line.getString(1).toDouble.toInt))}))

// COMMAND ----------

// MAGIC %md
// MAGIC Implement Map-Matching using GeoMatch

// COMMAND ----------

//n(=dimension of the Hilber curve) should be a power of 2. , distance =  meters from an intersection
//distance too big to cover all the points far away from intersections.
val geoMatch = new GeoMatch(false, 256, 4000, (-1, -1, -1, -1))

// COMMAND ----------

val resultRDD = geoMatch.spatialJoinKNN(rdd_first_set_intersections, rddSecondSet, 1, false)
//k is set to one to get the closest intersection

// COMMAND ----------

//resultRDD.toDF("k", "line").show(1, false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +----------------------------------+---------------------------------------------+
// MAGIC |k                                 |line                                         |
// MAGIC +----------------------------------+---------------------------------------------+
// MAGIC |[LT20xyABCDEF, [520xxxx, 361yyyy]]|[[LineString:924144ssss, [520zzzz, 361zzzz]]]|
// MAGIC +----------------------------------+---------------------------------------------+
// MAGIC only showing top 1 rows 
// MAGIC ```

// COMMAND ----------

val unmatched_events = resultRDD.filter(element => (element._2.isEmpty)).toDF("id", "intersection")

// COMMAND ----------

//no accidents unmatched
unmatched_events.count()

// COMMAND ----------

import org.apache.spark.sql.functions._
val intersections_df = intersections_unpacked.toDF("intersection")
val intersection_df2 = intersections_df.withColumn("all", split(col("intersection"), ",")).withColumn("intersection_id", $"all"(0)).withColumn("coordinates", $"all"(1)).drop("intersection", "all")
val intersection_df3 = intersection_df2.withColumn("all", split(col("coordinates"), " ")).withColumn("longitude", $"all"(0)).withColumn("latitude", $"all"(1)).drop("coordinates", "all")

// COMMAND ----------

intersection_df3.show(1,false) //intersections together with their coordinates

// COMMAND ----------

// MAGIC %md
// MAGIC Measure distance from each point to the nearest intersection

// COMMAND ----------

val result = resultRDD.map(element =>
                           {val accident = element._1
                           val intersection = element._2
                           (accident._payload, accident._pointCoord._1, accident._pointCoord._2, intersection(0)._payload, intersection(0)._pointCoord._1, intersection(0)._pointCoord._2)})

// COMMAND ----------

//result.take(1)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC Array[(String, Int, Int, String, Int, Int)] = Array((LT20xyABCDEF,520xxxx,361yyyy,LineString:92414ssss,520zzzz,361zzzz))
// MAGIC ```

// COMMAND ----------

val result_df = result.toDF("acc_id", "acc_long", "acc_lat", "inters_id", "inters_long", "inters_lat")

// COMMAND ----------

result_df.select("acc_id", "inters_id").coalesce(1).write.parquet("dbfs:/datasets/lithuania/acc_inters_ids")

// COMMAND ----------

//result_df.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+--------+-------+---------------------+-----------+----------+
// MAGIC |acc_id      |acc_long|acc_lat|inters_id            |inters_long|inters_lat|
// MAGIC +------------+--------+-------+---------------------+-----------+----------+
// MAGIC |LT20xyABCDEF|520xxxx |361yyyy|LineString:92414sssss|520zzzz    |361zzzz   |
// MAGIC +------------+--------+-------+---------------------+-----------+----------+
// MAGIC only showing top 1 rows.
// MAGIC ```

// COMMAND ----------

// MAGIC %md
// MAGIC To measure the geodesic distance we need to transform the coordinates to WGS84 system

// COMMAND ----------

def dist(lat1: String, long1: String, lat2: String, long2: String): Double = {
    val p1 = new Points(long1.toDouble, lat1.toDouble)
    val p2 = new Points(long2.toDouble, lat2.toDouble)
    geodesicDistanceOnWGS84(p1, p2)
  }
spark.udf.register("dist", dist(_:String, _:String, _:String, _:String):Double)

// COMMAND ----------

def project_to_wgs(lon: Int, lat: Int): String = { 
    
    if(!ArcGISRuntimeEnvironment.isInitialized())
    {
      ArcGISRuntimeEnvironment.setInstallDirectory("/dbfs/arcGISRuntime/arcgis-runtime-sdk-java-100.4.0/")
      ArcGISRuntimeEnvironment.initialize() 
    }
  
    val initial_point = new Point(lon.toDouble, lat.toDouble, SpatialReference.create(3035))
    val reprojection = GeometryEngine.project(initial_point, SpatialReference.create(4326))
    reprojection.toString
}
spark.udf.register("project_to_wgs", project_to_wgs(_:Int, _:Int):String)

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

val result_wgs_packed = result_df.selectExpr("acc_id", "inters_id", "project_to_wgs(acc_long, acc_lat) as acc_coord_wgs", "project_to_wgs(inters_long, inters_lat) as inters_coord_wgs")

// COMMAND ----------

val result_wgs = result_wgs_packed.selectExpr("acc_id", "inters_id", "unpack_lon(acc_coord_wgs) as acc_long_wgs", "unpack_lat(acc_coord_wgs) as acc_lat_wgs", "unpack_lon(inters_coord_wgs) as inters_long_wgs", "unpack_lat(inters_coord_wgs) as inters_lat_wgs")

// COMMAND ----------

//result_wgs.show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+---------------------+------------+-----------+---------------+--------------+
// MAGIC |acc_id      |inters_id            |acc_long_wgs|acc_lat_wgs|inters_long_wgs|inters_lat_wgs|
// MAGIC +------------+---------------------+------------+-----------+---------------+--------------+
// MAGIC |LT20xyABCDEF|LineString:92414sssss|23.xxxxxx   |54.yyyyyy  |23.xxxxxx      |54.yyyyyyy    |
// MAGIC +------------+---------------------+------------+-----------+---------------+--------------+
// MAGIC only showing top 1 rows.
// MAGIC ```

// COMMAND ----------

val acc_inters_distances = result_wgs.selectExpr("acc_id", "dist(acc_lat_wgs, acc_long_wgs, inters_lat_wgs, inters_long_wgs) as distance_acc_inters")

// COMMAND ----------

//acc_inters_distances.sort(col("distance_acc_inters").desc).show(1,false)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+-------------------+
// MAGIC |acc_id      |distance_acc_inters|
// MAGIC +------------+-------------------+
// MAGIC |LT20xyABCDEF|3915.650145464729  |
// MAGIC +------------+-------------------+
// MAGIC only showing top 1 rows.
// MAGIC ```

// COMMAND ----------

acc_inters_distances.coalesce(1).write.parquet("dbfs:/datasets/lithuania/acc_inters_distances")
