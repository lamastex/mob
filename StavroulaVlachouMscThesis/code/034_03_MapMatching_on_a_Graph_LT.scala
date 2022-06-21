// Databricks notebook source
// MAGIC %md
// MAGIC ## Map-Matching Events on a State Space / Road Graph with GeoMatch 
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

// MAGIC %md
// MAGIC ## Map-Matching with GeoMatch
// MAGIC [GeoMatch](https://ieeexplore.ieee.org/document/8622488) is a novel, scalable, and efficient big-data pipeline for large-scale map-matching on Apache Spark. It improves existing spatial big data solutions by utilizing a novel spatial partitioning scheme inspired by Hilbert space-filling curves.
// MAGIC 
// MAGIC The library can be found in the following git repository [GeoMatch](https://github.com/bdilab/GeoMatch).
// MAGIC 
// MAGIC The necessary files to generate the jar for this work can be found in the following fork [https://github.com/StavroulaVlachou/GeoMatch](https://github.com/StavroulaVlachou/GeoMatch). 

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

// MAGIC %md 
// MAGIC ## Map-Matching 

// COMMAND ----------

//This allows easy embedding of publicly available information into any other notebook
//when viewing in git-book just ignore this block - you may have to manually chase the URL in frameIt("URL").
//Example usage:
// displayHTML(frameIt("https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation#Topics_in_LDA",250))
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
displayHTML(frameIt("https://en.wikipedia.org/wiki/Map_matching",600))

// COMMAND ----------

import org.apache.spark.graphx._
import sqlContext.implicits._
import org.apache.spark.sql.functions._
import scala.collection.JavaConversions._
import org.cusp.bdi.gm.GeoMatch
import org.cusp.bdi.gm.geom.GMPoint
import org.cusp.bdi.gm.geom.GMLineString
import com.esri.arcgisruntime.geometry.{Point, SpatialReference, GeometryEngine}
import com.esri.arcgisruntime.geometry.GeometryEngine.project
import com.esri.arcgisruntime._

// COMMAND ----------

// MAGIC %md
// MAGIC ## State Space / Road Graph
// MAGIC - In this work, we wish to match points of interest - events - against states of a State Space. The State Space consists of elements of the Road Graph. Specifically, a state is either a vertex that corresponds to an intersection point or an edge which is essentially a road segment. 

// COMMAND ----------

// MAGIC %md
// MAGIC - First we obtain the nodes and ways of the underlying road network.

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
))

// COMMAND ----------

// MAGIC %md
// MAGIC - The next step is to obtain the intersection points and associate them with their corresponding vertices on the graph.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Intersection points

// COMMAND ----------

val intersections = spark.read.parquet("dbfs:/LT/intersections")

// COMMAND ----------

intersections.count //in this area there are 162325 intersection points 

// COMMAND ----------

// MAGIC %md
// MAGIC - GeoMatch deals with points whose coordinates are measured in meters. However, OSM data have their coordinates expressed in degrees (WGS84 - spatial reference index 4326). Thus, for each point that is to participate in the matching we identify it's OSM coordinates and reproject them onto the European Grid (spatial reference index 3035).

// COMMAND ----------

val intersection_points = nodeDS.join(intersections, intersections("intersectionNode") === nodeDS("nodeId")).drop("tags", "nodeId").select("intersectionNode", "latitude", "longitude")

// COMMAND ----------

val concat_coordinates = intersection_points.select($"intersectionNode",concat($"latitude",lit(" "),$"longitude").alias("coordinates"))

// COMMAND ----------

val firstIntersectionStates = concat_coordinates.select(concat(lit("LineString:"),$"intersectionNode").alias("LineString"),$"coordinates")
val firstIntersectionStates_rdd = firstIntersectionStates.rdd

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

val intersections_reprojected = firstIntersectionStates_rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]",""))
              .map(line => {val parts = line.replaceAll("\"","").split(",");
                            val arrCoords = parts.slice(1,parts.length)
              .map(xyStr => {val xy = xyStr.split(" ");
                             val reprojection = project_to_meters(xy(1).toString, xy(0).toString);
                             val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);
                             val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});
                            (parts(0).toString, arrCoords)})

// COMMAND ----------

val intersections_unpacked = intersections_reprojected.map(item => item._1.toString + "," + item._2(0).toString)

// COMMAND ----------

val rdd_first_set_intersections = intersections_unpacked.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});new GMPoint(parts(0), arrCoords(0))}))


// COMMAND ----------

// MAGIC %md
// MAGIC - The next step is to fetch the events that are to be map-matched and transform their coordinates as well. Note that for this work, the events of interest are accidents recorded within Lithuania's road network. 

// COMMAND ----------

val events = spark.read.format("csv").load("/FileStore/tables/LTnodes.csv").rdd.map(line => line.toString)
events.count() //there are 11989 events to be matched 

// COMMAND ----------

val all_accidents = spark.read.format("csv").load("/FileStore/tables/LTnodes.csv").toDF("PointId", "longitude", "latitude")

// COMMAND ----------

val rddSecondSet = events.mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").replaceAll("\\[","").replaceAll("\\]","").split(',');new GMPoint(parts(0), (parts(1).toDouble.toInt, parts(2).toDouble.toInt))}))

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1st round of Map Matching
// MAGIC 
// MAGIC - In this first round the focus is around the intersection points and the events occurring within a predefined distance from them. Here the distance tolerance is set to 20 meters and the number of neighbours to be found is 1. 

// COMMAND ----------

val geoMatch = new GeoMatch(false, 256, 20, (-1, -1, -1, -1)) //dimension of the Hilber curve=256, default value,  should be a power of 2. 

// COMMAND ----------

val resultRDD = geoMatch.spatialJoinKNN(rdd_first_set_intersections, rddSecondSet, 1, false)

// COMMAND ----------

// MAGIC %md 
// MAGIC - 3743 events (out of 11989) are found to be within a 20 meter distance radius from intersection points. 

// COMMAND ----------

resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty)).count()

// COMMAND ----------

val result_first_round = resultRDD.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty)).map(element => (element._1, element._2(0))).toDF("PointId", "State")

// COMMAND ----------

val intersection_counts = result_first_round.groupBy("State").count

// COMMAND ----------

// MAGIC %md 
// MAGIC - One of the advantages of GeoMatch is that it carried all of the data points that are to be matched throughout the pipeline, even in the case where no match is found. This is key in this case, since the points were not matched successfully during this first round are subject to a second iteration where they are to be matched against the remaining of the State Space. 

// COMMAND ----------

val unmatched_events = resultRDD.filter(element => (element._2.isEmpty)).map(element => element._1.payload).toDF("id")
val second_set_second_round = unmatched_events.join(all_accidents, unmatched_events("id") === all_accidents("PointId")).drop("id").rdd.map(line => line.toString)
val rddSecondSetSecondRound = second_set_second_round
.mapPartitions(_.map(line => {val parts = line.replaceAll("\"","").replaceAll("\\[","").replaceAll("\\]","").split(',');
                              new GMPoint(parts(0),(parts(1).toDouble.toInt, parts(2).toDouble.toInt))}))

// COMMAND ----------

// MAGIC %md 
// MAGIC - The remaining of the State Space consists of the edges of the Road Graph. In the following cells, we fetch these edges and associate them with their OSM coordinates and their reporjection. 

// COMMAND ----------

val edges = spark.read.parquet("dbfs:/_checkpoint/edges_LT_initial") //edges of G0
val vertices = spark.read.parquet("dbfs:/_checkpoint/vertices_LT_initial").toDF("vertexId", "latitude", "longitude") //vertices of G0

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

val ways_reprojected = first.rdd.map(line => line.toString.replaceAll("\\[","").replaceAll("\\]","")).map(line => {val parts = line.replaceAll("\"","").split(",");val arrCoords = parts.slice(1,parts.length).map(xyStr => {val xy = xyStr.split(' ');val reprojection = project_to_meters(xy(1).toString, xy(0).toString);val coords = reprojection.replaceAll(",","").replaceAll("\\[","").split(" ").slice(1,reprojection.length);val xy_new = coords(0).toString +" "+ coords(1).toString;xy_new});(parts(0).toString, arrCoords)})

// COMMAND ----------

val ways_unpacked = ways_reprojected.map(item => item._1.toString + "," + item._2(0).toString + "," + item._2(1).toString)

// COMMAND ----------

val rdd_first_set = ways_unpacked
.mapPartitions(_.map(line =>{val parts = line.replaceAll("\"","").split(',');
                             val arrCoords = parts.slice(1, parts.length).map(xyStr => {val xy = xyStr.split(' ');(xy(0).toDouble.toInt, xy(1).toDouble.toInt)});
                             new GMLineString(parts(0), arrCoords)}))

// COMMAND ----------

// MAGIC %md 
// MAGIC - In this second round of Map-Matching, the distance threshold is set to be 200 meters. The dimension of the Hilbert index curve is again set to each desault value (256) and the number of nearest neighnours to be found is 1. 

// COMMAND ----------

val geoMatchSecond = new GeoMatch(false, 256, 200, (-1, -1, -1, -1)) 

// COMMAND ----------

val resultRDDsecond = geoMatchSecond.spatialJoinKNN(rdd_first_set, rddSecondSetSecondRound, 1, false)

// COMMAND ----------

// MAGIC %md 
// MAGIC - The number of events that do not lie within a 200 meter radius from road segments is 269.

// COMMAND ----------

resultRDDsecond.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => (element._2.isEmpty)).count()

// COMMAND ----------

// MAGIC %md 
// MAGIC ### How many events have occurred in each state? 

// COMMAND ----------

// MAGIC %md
// MAGIC - We are interested in how many events are matched against each state. 

// COMMAND ----------

val res = resultRDDsecond.map(element => (element._1.payload, element._2.map(_.payload))).filter(element => !(element._2.isEmpty))

// COMMAND ----------

val result_second_round = res.map(element => (element._1, element._2(0))).toDF("PointId", "State")

// COMMAND ----------

val edge_counts = result_second_round.groupBy("State").count

// COMMAND ----------

val state_counts = edge_counts.union(intersection_counts)

// COMMAND ----------

val all_intersection_states = rdd_first_set_intersections.toDF("stateId", "coords").drop("coords")
val all_edge_states = rdd_first_set.toDF("stateId", "coords").drop("coords")
val all_states = all_intersection_states.union(all_edge_states)
all_states.count //number of states 

// COMMAND ----------

// MAGIC %md
// MAGIC - Find the states with no event has been matched against, assign count value equal to 0 and union them with the rest of the states_counts. This way, each state in the State Space is assigned a numerical value representing the number of accidents that have occurred within that state. 

// COMMAND ----------

val s1 = all_states.join(state_counts, all_states("stateId") === state_counts("State"), "left_outer").drop("State")
val s_final = s1.na.fill(0)

// COMMAND ----------

s_final.distinct.agg(sum("count")).show()  //11720 events in total successfully matched 

// COMMAND ----------

def trim_id(stateId: String): String = {
  val res = stateId.split(":")(1)
  return res
}

def trim_point(pointId: String): String = {
  val res = pointId.split(" ")(1)
  return res
}
spark.udf.register("trim_point", trim_point(_:String): String)
spark.udf.register("trim_id", trim_id(_:String): String)



// COMMAND ----------

val total_result = result_first_round.union(result_second_round)
val trimed_total_result = total_result.selectExpr("trim_point(PointId) as point", "trim_id(State) as state")

// COMMAND ----------

// MAGIC %md 
// MAGIC - Return here after notebook ```034_06SimulatingArrivalTimesNHPP_Inversion```

// COMMAND ----------

// MAGIC %md 
// MAGIC - We want to map the simulated graph elements into an exact location

// COMMAND ----------

val df = spark.read.parquet("dbfs:/roadSafety/simulation_location").toDF("simulated_location", "arrival_time")
val location_id = df.select("simulated_location")

// COMMAND ----------

import org.apache.spark.sql.functions._
val intersection_samples = location_id.join(nodes_df, col("simulated_location") === col("id")).select("simulated_location", "latitude", "longitude")
intersection_samples.count
val edge_ids = edge_coordinates.withColumn("edge_id", concat(col("src"), lit("+"), col("dst")))
val edge_samples = location_id.join(edge_ids, col("simulated_location") === col("edge_id")).drop("src", "dst", "edge_id")

// COMMAND ----------

import org.apache.spark.mllib.random.RandomRDDs
val random_edge_coordinates = edge_samples.withColumn("random_sample", rand())

// COMMAND ----------

// MAGIC %md 
// MAGIC - For each simulated edge, generate a two dimensional uniform sample and scale it according to the coordinates of the edge's source and destination

// COMMAND ----------

def random_lat(src_lat: Double, dst_lat: Double, sample: Double): Double = {
  val lat_min = src_lat.min(dst_lat)
  val lat_max = src_lat.max(dst_lat)
  val lat = sample * (lat_max - lat_min) + lat_min
  return lat
}
def random_lon(src_lon: Double, dst_lon: Double, sample: Double): Double = {
  val lon_min = src_lon.min(dst_lon)
  val lon_max = src_lon.max(dst_lon)
  val lon = sample * (lon_max - lon_min) + lon_min
  return lon
}

spark.udf.register("random_lat", random_lat(_: Double, _: Double, _: Double): Double)
spark.udf.register("random_lon", random_lon(_: Double, _: Double, _: Double): Double)


val random_coordinates = random_edge_coordinates.selectExpr("random_lat(src_latitude, dst_latitude, random_sample) as latitude", "random_lon(src_longitude, dst_longitude, random_sample) as longitude")

// COMMAND ----------

val df_final = random_coordinates.union(intersection_samples.select("latitude", "longitude"))
df_final.count()

// COMMAND ----------

// MAGIC %md 
// MAGIC ```
// MAGIC df_final.show()
// MAGIC ```
// MAGIC Output: 
// MAGIC ```
// MAGIC +------------------+------------------+
// MAGIC |          latitude|         longitude|
// MAGIC +------------------+------------------+
// MAGIC |54.66xxx          |25.29yyy          |
// MAGIC +------------------+------------------+
// MAGIC 
// MAGIC ```

// COMMAND ----------


