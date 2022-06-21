// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Creating G0 with edges in the two directions.
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
// MAGIC ###Step 1 - OSM to GraphX

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

import org.apache.spark.sql.functions._
import org.apache.spark.graphx._

// COMMAND ----------

// MAGIC %md
// MAGIC Read the parquet files for nodes and ways obtained from the osm-parquetizer.

// COMMAND ----------

spark.conf.set("spark.sql.parquet.binaryAsString", true)

val nodes_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.node.parquet")
val ways_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.way.parquet")

// COMMAND ----------

val allowableWays2 = Seq(
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
  "motorway_junction",
  "unclassified"
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

val wayDS2 = ways_df.flatMap(way => {
        val tagSet = way.getAs[Seq[Row]]("tags").map{case Row(key:String, value:String) =>  value}.toArray
        if (tagSet.intersect(allowableWays2).nonEmpty ){
            Array(WayEntry(way.getAs[Long]("id"),
            tagSet,
            way.getAs[Seq[Row]]("nodes").map{case Row(index:Integer, nodeId:Long) =>  nodeId}.toArray
            ))
        }
        else { Array[WayEntry]()}
}
).cache()

// COMMAND ----------

wayDS2.count()

// COMMAND ----------

import org.apache.spark.sql.functions.explode
val nodeCounts2 = wayDS2
                    .select(explode('nodes).as("node"))
                    .groupBy('node).count

//nodeCounts.show(5)

// COMMAND ----------

val intersectionNodes2 = nodeCounts2.filter('count >= 2).select('node.alias("intersectionNode"))
val true_intersections2 = intersectionNodes2

// COMMAND ----------

intersectionNodes2.count()

// COMMAND ----------

val distinctNodesWays2 = wayDS2.flatMap(_.nodes).distinct

// COMMAND ----------

distinctNodesWays2.count()

// COMMAND ----------

val wayNodes2 = nodeDS.as("nodes") //nodes that are in a way + nodes info from nodeDS
  .joinWith(distinctNodesWays2.as("ways"), $"ways.value" === $"nodes.nodeId")
  .map(_._1).cache

// COMMAND ----------

import org.apache.spark.sql.functions.{collect_list, map, udf}
import org.apache.spark.sql.functions._

// I assume that each "nodes" sequence contains at least one node
// We do not really need first and last elements from the sequence and when combining with original nodes, just we assign them "true"

val remove_first_and_last = udf((x: Seq[Long]) => x.drop(1).dropRight(1))

val nodes2 = wayDS2.
  select($"wayId", $"nodes").
  withColumn("node", explode($"nodes")).
  drop("nodes")

val get_first_and_last = udf((x: Seq[Long]) => {val first = x(0); val last = x.reverse(0); Array(first, last)})

val first_and_last_nodes2 = wayDS2.
  select($"wayId", get_first_and_last($"nodes").as("nodes")).
  withColumn("node", explode($"nodes")).
  drop("nodes")

val fake_intersections2 = first_and_last_nodes2.select($"node").distinct().withColumnRenamed("node", "value")

// Turn intersection set into a dataset to join (all values must be unique)
val intersections2 = intersectionNodes2.union(fake_intersections2).distinct

val wayNodesLocated2 = nodes2.join(wayNodes2, wayNodes2.col("nodeId") === nodes2.col("node")).select($"wayId", $"node", $"latitude", $"longitude")

case class MappedWay(wayId: Long, labels_located: Seq[Map[Long, (Boolean, Double, Double)]])

val maps2 = wayNodesLocated2.join(intersections2, 'node === 'intersectionNode, "left_outer").
  //left outer joins returns all rows from the left DataFrame/Dataset regardless of match found on the right dataset
    select($"wayId", $"node", $"intersectionNode".isNotNull.as("contains"), $"latitude", $"longitude").
   groupBy("wayId").agg(collect_list(map($"node", struct($"contains".as("_1"), $"latitude".as("_2"), $"longitude".as("_3")))).as("labels_located")).as[MappedWay]  

val combine = udf((nodes: Seq[Long], labels_located: Seq[scala.collection.immutable.Map[Long, (Boolean, Double, Double)]]) => {
  // If labels does not have "node", then it is either start/end - we assign label = true, latitude = 0, longitude = 0 for it, TO DO: revise it later, not sure
  val m = labels_located.map(_.toSeq).flatten.toMap

  nodes.map { node => (node, m.getOrElse(node, (true, 0D, 0D))) } //add structure

})


val strSchema = "array<struct<nodeId:long, nodeInfo:struct<label:boolean, latitude:double, longitude: double>>>"
val labeledWays2 = wayDS2.join(maps2, "wayId")
                     .select($"wayId", $"tags", combine($"nodes", $"labels_located").as("labeledNodes").cast(strSchema))

// COMMAND ----------

case class Intersection(OSMId: Long , latitude: Double, longitude: Double, inBuf: ArrayBuffer[(Long, Double, Double)], outBuf: ArrayBuffer[(Long, Double, Double)])

val segmentedWays2 = labeledWays2.map(way => {
  
  val labeledNodes = way.getAs[Seq[Row]]("labeledNodes").map{case Row(k: Long, Row(v: Boolean, w:Double, x:Double)) => (k, v,w,x)}.toSeq //labeledNodes: (nodeid, label, lat, long)
  val wayId = way.getAs[Long]("wayId")
  
  val indexedNodes: Seq[((Long, Boolean, Double, Double), Int)] = labeledNodes.zipWithIndex //appends an integer as an index to every labeledNodes in a way
  
  val intersections = ArrayBuffer[Intersection]()  
  
  val currentBuffer = ArrayBuffer[(Long, Double, Double)]()
  
  val way_length = labeledNodes.length //number of nodes in a way
  
  if (way_length == 1) {

    val intersect = new Intersection(labeledNodes(0)._1, labeledNodes(0)._3, labeledNodes(0)._4, ArrayBuffer((-1L, 0D, 0D)), ArrayBuffer((-1L, 0D, 0D))) //include lat and long info

    var result = Array((intersect.OSMId, intersect.latitude, intersect.longitude, intersect.inBuf.toArray, intersect.outBuf.toArray))
    (wayId, result) //return
  }
  else {
    indexedNodes.foreach{ case ((id, isIntersection, latitude, longitude), i) => // id is nodeId and isIntersection is the node label
      if (isIntersection) {
        val newEntry = new Intersection(id, latitude, longitude, currentBuffer.clone, ArrayBuffer[(Long, Double, Double)]())
        intersections += newEntry
        currentBuffer.clear
      }
      else {
        currentBuffer ++= Array((id, latitude, longitude))  //if the node is not an intersection append the nodeId to the current buffer 
      }
      
      // Reaches the end of the way while the outBuffer is not empty
      // Append the currentBuffer to the last intersection
      if (i == way_length - 1 && !currentBuffer.isEmpty) {  
        if (intersections.isEmpty){
        intersections += new Intersection(-1, 0D, 0D, currentBuffer, ArrayBuffer[(Long, Double, Double)]()) 
        }
        else {
          intersections.last.outBuf ++= currentBuffer
        }
        currentBuffer.clear
      }
    }
    var result = intersections.map(i => (i.OSMId, i.latitude, i.longitude, i.inBuf.toArray, i.outBuf.toArray)).toArray  
    (wayId, result) 
  }
})

// COMMAND ----------

segmentedWays2.count()

// COMMAND ----------

//The nested structure of the segmentedWays is unwrapped
val waySegmentDS2 = segmentedWays2.flatMap(way => way._2.map(node => (way._1, node))) 

// for each (wayId, Array(IntersectionNode) => (wayId, IntersectionNode)

// COMMAND ----------

import scala.collection.immutable.Map

//returns the intersection nodes with the ways where they appear mapped with the nodes in those ways (inBuff, outBuff) 
val intersectionVertices2 = waySegmentDS2
  .map(way => 
   //nodeId     latitude   longitude      wayId      inBuff      outBuff
   (way._2._1, (way._2._2, way._2._3, Map(way._1 -> (way._2._4, way._2._5))))) 
  .rdd
  //                     latitude, long, Map(wayId, inBuff, outBuff)
  .reduceByKey((a,b) => (a._1,     a._2, a._3 ++ b._3)) 

//intersectionVertices =  RDD[(nodeId, (latitude, longitude, wayMap(wayId -> inBuff, outBuff)))]

// COMMAND ----------

intersectionVertices2.count()

// COMMAND ----------

intersectionVertices2.map(vertex => (vertex._1, vertex._2)).toDF("id", "Map").write.mode("overwrite").parquet("/_checkpoint/vertices_LT_initial_added_tags")

// COMMAND ----------

val edges2 = segmentedWays2
  .filter(way => way._2.length > 1) //ways with more than one intersections
  .flatMap{ case (wayId, nodes_info) => {  
             nodes_info.sliding(2) // For each way it takes nodes in pairs
               .flatMap(segment => //segment is the pair of two nodes
                   List(Edge(segment(0)._1, segment(1)._1, wayId))
               )
   }}

// COMMAND ----------

val edges2_two_directions = segmentedWays2
  .filter(way => way._2.length > 1) //ways with more than one intersections
  .flatMap{ case (wayId, nodes_info) => {  
             nodes_info.sliding(2) // For each way it takes nodes in pairs
               .flatMap(segment => //segment is the pair of two nodes
                   List(Edge(segment(0)._1, segment(1)._1, wayId),
                        Edge(segment(1)._1, segment(0)._1, wayId))
               )
   }}

// COMMAND ----------

edges2_two_directions.map(edge => (edge.srcId, edge.dstId)).toDF("src","dst").write.mode("overwrite").parquet("/_checkpoint/edges_LT_initial_two_directions_tags_added")
