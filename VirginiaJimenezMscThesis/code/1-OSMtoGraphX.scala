// Databricks notebook source
// MAGIC %md
// MAGIC ## Creating a road graph from OpenStreetMap (OSM) data with GraphX
// MAGIC 
// MAGIC Stavroula Rafailia Vlachou ([LinkedIn](https://www.linkedin.com/in/stavroula-rafailia-vlachou/)), Virginia Jimenez Mohedano ([LinkedIn](https://www.linkedin.com/in/virginiajimenezmohedano/)) and Raazesh Sainudiin ([LinkedIn](https://www.linkedin.com/in/raazesh-sainudiin-45955845/)).
// MAGIC 
// MAGIC ```
// MAGIC This project was supported by SENSMETRY through a Data Science Project Internship 
// MAGIC between 2022-01-17 and 2022-06-05 to Stavroula R. Vlachou and Virginia J. Mohedano 
// MAGIC and Databricks University Alliance with infrastructure credits from AWS to 
// MAGIC Raazesh Sainudiin, Department of Mathematics, Uppsala University, Sweden.
// MAGIC ```
// MAGIC 
// MAGIC 2022, Uppsala, Sweden
// MAGIC 
// MAGIC This project builds on top of the work of Dillon George (2016-2018). 
// MAGIC 
// MAGIC 
// MAGIC ```
// MAGIC Licensed under the Apache License, Version 2.0 (the "License");
// MAGIC you may not use this file except in compliance with the License.
// MAGIC You may obtain a copy of the License at
// MAGIC 
// MAGIC     http://www.apache.org/licenses/LICENSE-2.0
// MAGIC 
// MAGIC Unless required by applicable law or agreed to in writing, software
// MAGIC distributed under the License is distributed on an "AS IS" BASIS,
// MAGIC WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// MAGIC See the License for the specific language governing permissions and
// MAGIC limitations under the License.
// MAGIC ```

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 0. Download the data - once per cluster

// COMMAND ----------

// MAGIC %md
// MAGIC Download the road network representation of Lithuania through OSM data distributed from GeoFabrik [https://download.geofabrik.de/europe/lithuania.html](https://download.geofabrik.de/europe/lithuania.html)

// COMMAND ----------

// MAGIC %sh 
// MAGIC curl -O https://download.geofabrik.de/europe/lithuania-latest.osm.pbf

// COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/lithuania-latest.osm.pbf", "dbfs:/datasets/osm/lithuania/lithuania.osm.pbf")

// COMMAND ----------

// MAGIC %md
// MAGIC ###Step 1 - Load the data 

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
// MAGIC 
// MAGIC - For the ingestion of the entire OSM Lithuanian road network dataset, the PBF file obtained from OSM is transformed to three parquet files; one for each primitive (nodes, ways and relations), by utilising methods of the [osm-parquetizer project](https://github.com/adrianulbona/osm-parquetizer). The first two generated files, corresponding to the nodes and ways are then transferred into the distributed file system for further exploitation. 

// COMMAND ----------

//Run this command only once per cluster 
%sh 
java -jar /dbfs/FileStore/jars/2706d711_3963_4d88_92e7_a8870d0164d1-osm_parquetizer_1_0_1_SNAPSHOT-80d25.jar /dbfs/datasets/osm/lithuania/lithuania.osm.pbf

// COMMAND ----------

// MAGIC %sh
// MAGIC ls /dbfs/datasets/osm/lithuania/

// COMMAND ----------

// MAGIC %md
// MAGIC Read the parquet files of the nodes and ways obtained from the osm-parquetizer.

// COMMAND ----------

spark.conf.set("spark.sql.parquet.binaryAsString", true)

val nodes_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.node.parquet")
val ways_df = spark.read.parquet("dbfs:/datasets/osm/lithuania/lithuania.osm.pbf.way.parquet")

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Step 2 - Construction of a Road Graph out of a Road Network 

// COMMAND ----------

// MAGIC %md 
// MAGIC 
// MAGIC - The list of tags chosen for this work. For the semantic meaning of each tag see the [OSM description](https://wiki.openstreetmap.org/wiki/Map_features). The list is non exhaustive and should be adapted according to the desired granulatiry of and level of detail of the project at hand. 

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

nodeDS.count()

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

wayDS.count()

// COMMAND ----------

val nodeCounts = wayDS
                    .select(explode('nodes).as("node"))
                    .groupBy('node).count

// COMMAND ----------

// MAGIC %md 
// MAGIC 
// MAGIC - An intersection node is defined here as a node that lies in at least two ways. 

// COMMAND ----------

val intersectionNodes = nodeCounts.filter('count >= 2).select('node.alias("intersectionNode"))
val true_intersections = intersectionNodes

// COMMAND ----------

intersectionNodes.count()

// COMMAND ----------

val distinctNodesWays = wayDS.flatMap(_.nodes).distinct //the distinct nodes within the ways 

// COMMAND ----------

distinctNodesWays.count()

// COMMAND ----------

val wayNodes = nodeDS.as("nodes") 
  .joinWith(distinctNodesWays.as("ways"), $"ways.value" === $"nodes.nodeId")
  .map(_._1).cache

// COMMAND ----------

wayNodes.count()

// COMMAND ----------

val intersectionSetVal = intersectionNodes.as[Long].collect.toSet; //turn intersectionNodes to Set 

// COMMAND ----------

import org.apache.spark.sql.functions.{collect_list, map, udf}
import org.apache.spark.sql.functions._

val remove_first_and_last = udf((x: Seq[Long]) => x.drop(1).dropRight(1))

val nodes = wayDS.
  select($"wayId", $"nodes").
  withColumn("node", explode($"nodes")).
  drop("nodes")

val get_first_and_last = udf((x: Seq[Long]) => {val first = x(0); val last = x.reverse(0); Array(first, last)})

val first_and_last_nodes = wayDS.
  select($"wayId", get_first_and_last($"nodes").as("nodes")).
  withColumn("node", explode($"nodes")).
  drop("nodes")

val dead_end_points = first_and_last_nodes.select($"node").distinct().withColumnRenamed("node", "value")

// Turn intersection set into a dataset to join (all values must be unique)
val intersections = intersectionNodes.union(dead_end_points).distinct      
 
val wayNodesLocated = nodes.join(wayNodes, wayNodes.col("nodeId") === nodes.col("node")).select($"wayId", $"node", $"latitude", $"longitude")


case class MappedWay(wayId: Long, labels_located: Seq[Map[Long, (Boolean, Double, Double)]])


val maps = wayNodesLocated.join(intersections, 'node === 'intersectionNode, "left_outer").
  //left outer joins returns all rows from the left DataFrame/Dataset regardless of match found on the right dataset
    select($"wayId", $"node", $"intersectionNode".isNotNull.as("contains"), $"latitude", $"longitude").
   groupBy("wayId").agg(collect_list(map($"node", struct($"contains".as("_1"), $"latitude".as("_2"), $"longitude".as("_3")))).as("labels_located")).as[MappedWay] 
 

val combine = udf((nodes: Seq[Long], labels_located: Seq[scala.collection.immutable.Map[Long, (Boolean, Double, Double)]]) => {
  // If labels does not have "node", then it is either start/end - we assign label = true, latitude = 0, longitude = 0 for it, TO DO: revise it later, not sure
  val m = labels_located.map(_.toSeq).flatten.toMap

  nodes.map { node => (node, m.getOrElse(node, (true, 0D, 0D))) } //add structure

})


val strSchema = "array<struct<nodeId:long, nodeInfo:struct<label:boolean, latitude:double, longitude: double>>>"
val labeledWays = wayDS.join(maps, "wayId")
                     .select($"wayId", $"tags", combine($"nodes", $"labels_located").as("labeledNodes").cast(strSchema))

// COMMAND ----------

case class Intersection(OSMId: Long , latitude: Double, longitude: Double, inBuf: ArrayBuffer[(Long, Double, Double)], outBuf: ArrayBuffer[(Long, Double, Double)])

val segmentedWays = labeledWays.map(way => {
  
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
    indexedNodes.foreach{ case ((id, isIntersection, latitude, longitude), i) => // id is nodeId and isIntersection is the node's boolean label
      if (isIntersection) {
        val newEntry = new Intersection(id, latitude, longitude, currentBuffer.clone, ArrayBuffer[(Long, Double, Double)]())
        intersections += newEntry
        currentBuffer.clear
      }
      else {
        currentBuffer ++= Array((id, latitude, longitude))  //if the node is not an intersection append the nodeId to the current buffer 
      }
      
      // Reaches the end of the way while the outBuffer is not empty
      // Append the currentBuffer to the last existing intersection
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

val schema = "array<struct<nodeId:bigint,latitude:double,longitude:double,inBuff:array<struct<nodeId:bigint,latitude:double,longitude:double>>,outBuff:array<struct<nodeId:bigint,latitude:double,longitude:double>>>>"
segmentedWays.select($"_1".alias("wayId"), $"_2".cast(schema).alias("nodeInfo")).printSchema()

// COMMAND ----------

//Unwrap the nested structure of the segmentedWays

val waySegmentDS = segmentedWays.flatMap(way => way._2.map(node => (way._1, node))) 

// COMMAND ----------

import scala.collection.immutable.Map

val intersectionVertices = waySegmentDS
  .map(way => 
   //nodeId     latitude   longitude      wayId      inBuff      outBuff
   (way._2._1, (way._2._2, way._2._3, Map(way._1 -> (way._2._4, way._2._5))))) 
  .rdd
  //                     latitude, long, Map(wayId, inBuff, outBuff)
  .reduceByKey((a,b) => (a._1,     a._2, a._3 ++ b._3)) 

//intersectionVertices =  RDD[(nodeId, (latitude, longitude, wayMap(wayId -> inBuff, outBuff)))]

// COMMAND ----------

intersectionVertices.count()

// COMMAND ----------

val edges = segmentedWays
  .filter(way => way._2.length > 1) //ways with more than one nodes
  .flatMap{ case (wayId, nodes_info) => {  
             nodes_info.sliding(2) 
               .flatMap(segment => //segment is the pair of two nodes
                   List(Edge(segment(0)._1, segment(1)._1, wayId))
               )
   }}

// COMMAND ----------

edges.count()

// COMMAND ----------

sc.setCheckpointDir("/_checkpoint") // just a directory in distributed file system
val edges_rdd = edges.rdd
intersectionVertices.checkpoint()
edges_rdd.checkpoint()

// COMMAND ----------

val roadGraph = Graph(intersectionVertices, edges_rdd).cache

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Step 3 - Construction of a Weighted Road Graph

// COMMAND ----------

import com.esri.core.geometry.GeometryEngine.geodesicDistanceOnWGS84
import com.esri.core.geometry.Point

// COMMAND ----------

val weightedRoadGraph = roadGraph.mapTriplets{triplet => 
  def dist(lat1: Double, long1: Double, lat2: Double, long2: Double): Double = {
    val p1 = new Point(long1, lat1)
    val p2 = new Point(long2, lat2)
    geodesicDistanceOnWGS84(p1, p2)
  }
  
  val wayNodesInBuff = triplet.dstAttr._3(triplet.attr)._1 //dstAttr is the vertex attribute (latitude, longitude, wayMap(wayId -> inBuff, outBuff))
  
  if (wayNodesInBuff.isEmpty) {
      (triplet.attr, dist(triplet.srcAttr._1, triplet.srcAttr._2, triplet.dstAttr._1, triplet.dstAttr._2))
  
  } else {
      var distance: Double = 0.0

      distance += dist(triplet.srcAttr._1, triplet.srcAttr._2, wayNodesInBuff(0)._2, wayNodesInBuff(0)._3 )
    
      if (wayNodesInBuff.length > 1) {
      //accumulate the intermediate distances 
        distance += wayNodesInBuff.sliding(2).map{
        buff => dist(buff(0)._2, buff(0)._3, buff(1)._2, buff(1)._3)}
        .reduce(_ + _)
     }
    
      distance += dist(wayNodesInBuff.last._2, wayNodesInBuff.last._3, triplet.dstAttr._1, triplet.dstAttr._2)

      (triplet.attr, distance)
    }
  
}.cache

// COMMAND ----------

weightedRoadGraph.edges.count() //number of edges 

// COMMAND ----------

weightedRoadGraph.edges.filter(edge => (edge.attr._2 > 100.0)).count() //number of suffering edges with a distance tolerance of 100 meters 

// COMMAND ----------

weightedRoadGraph.vertices.count() //number of vertices 

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Step 4 - Construction of Coarsened Road Graph
// MAGIC 
// MAGIC - The distance tolerance here is set to 100 meters. 

// COMMAND ----------

import org.apache.spark.graphx.{Edge => Edges}
val splittedEdges = weightedRoadGraph.triplets.flatMap{triplet => {
  def dist(lat1: Double, long1: Double, lat2: Double, long2: Double): Double = {
    val p1 = new Point(long1, lat1)
    val p2 = new Point(long2, lat2)
    geodesicDistanceOnWGS84(p1, p2)
  }
  val maxDist = 100
  var finalResult = Array[(Edges[(Long,  Double)], (Long, (Double, Double, Map[Long,(Array[(Long, Double, Double)], Array[(Long, Double, Double)])])), (Long, (Double, Double, Map[Long,(Array[(Long, Double, Double)], Array[(Long, Double, Double)])])))]()
  
  if(triplet.attr._2 > maxDist){                            
    val wayId = triplet.attr._1
    var wayNodesBuff = triplet.dstAttr._3(wayId)._1 
    var wayNodesBuffSize = wayNodesBuff.length
    
    if(wayNodesBuffSize > 0){
      var previousSrc = triplet.srcId

      var distance: Double = 0.0
      var currentBuff = Array[(Long, Double, Double)]()
      
      distance += dist(triplet.srcAttr._1, triplet.srcAttr._2, wayNodesBuff(0)._2, wayNodesBuff(0)._3) 
      
      var newVertex = (triplet.srcId, triplet.srcAttr)
      var previousVertex = newVertex
      
      if (distance > maxDist){
        newVertex = (wayNodesBuff(0)._1, (wayNodesBuff(0)._2, wayNodesBuff(0)._3, Map(wayId -> (Array[(Long, Double, Double)](), Array[(Long, Double, Double)]()))))
            
        finalResult +:= (Edges(previousSrc, wayNodesBuff(0)._1, (wayId, distance)), previousVertex, newVertex) 
        
        previousVertex = newVertex
        
        distance = 0
        previousSrc = wayNodesBuff(0)._1
      }
      else 
      {
        currentBuff +:= wayNodesBuff(0)
      }
         
      //loop through pairs of nodes in the way (in the buffer)
      if (wayNodesBuff.length > 1){
      wayNodesBuff.sliding(2).foreach{segment => {
        
        val tmp_dst = distance
        distance += dist(segment(0)._2, segment(0)._3, segment(1)._2, segment(1)._3)
        
        if (distance > maxDist)
        {
          if(segment(0)._1 != previousSrc){
              //      Vertex(nodeId,      (lat,                long,     Map(wayId->inBuff, outBuff)))
            newVertex = (segment(0)._1, (segment(0)._2, segment(0)._3, Map(wayId -> (currentBuff, Array[(Long, Double, Double)]()))) )

            //adds the edge to the array
            finalResult +:= (Edges(previousSrc, segment(0)._1, (wayId, tmp_dst)), previousVertex, newVertex)

            previousVertex = newVertex
            distance -= tmp_dst
            previousSrc = segment(0)._1
            currentBuff = Array[(Long, Double, Double)]()
          }    
        }
        else 
        {
          currentBuff +:= segment(0)
        }
      }}}
      
      
      //from last node in the inBuff to the dst
      val tmp_dist = distance
      distance += dist(wayNodesBuff.last._2, wayNodesBuff.last._3, triplet.dstAttr._1, triplet.dstAttr._2)
      if (distance > maxDist){
        if (wayNodesBuff.last._1 != previousSrc){
            newVertex = (wayNodesBuff.last._1, (wayNodesBuff.last._2, wayNodesBuff.last._3, Map(wayId -> (currentBuff, Array[(Long, Double, Double)]()))))
            finalResult +:= (Edges(previousSrc, wayNodesBuff.last._1, (wayId, tmp_dist)), previousVertex, newVertex) 
            previousVertex = newVertex
            distance -= tmp_dist
            previousSrc = wayNodesBuff.last._1 
            currentBuff = Array[(Long, Double, Double)]()
            newVertex = (triplet.dstId, (triplet.dstAttr._1, triplet.dstAttr._2, Map(wayId -> (currentBuff, triplet.dstAttr._3(wayId)._2))) )
        }
      }
      finalResult +:= (Edges(previousSrc, triplet.dstId, (wayId, distance)), previousVertex, newVertex)
      
    }
    // Distance > threshold but no nodes in the way (buffer)
    else
    {
      finalResult +:= (Edges(triplet.srcId, triplet.dstId, triplet.attr), (triplet.srcId, triplet.srcAttr), (triplet.dstId, triplet.dstAttr))
    }
  }
  // Distance < threshold
  else
  {
    finalResult +:= (Edges(triplet.srcId, triplet.dstId, triplet.attr), (triplet.srcId, triplet.srcAttr), (triplet.dstId, triplet.dstAttr))
  }
  
  // return
  finalResult
}}

// COMMAND ----------

splittedEdges.count() 

// COMMAND ----------

// Taking each edge and its reverse
val segmentedEdges = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => Array(edge)}
segmentedEdges.count() 


// COMMAND ----------

// Taking the individual vertices
val segmentedVertices = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => Array(srcVertex) ++ Array(dstVertex)}

segmentedVertices.map(node => node._1).distinct().count()

// COMMAND ----------

// Converting the vertices to a df
val verticesDF = segmentedVertices.toDF("nodeId","attr").select($"nodeId",$"attr._1".as("lat"),$"attr._2".as("long"),explode($"attr._3"))
    .withColumnRenamed("key","wayId").withColumnRenamed("value","buffers")
    .select($"nodeId",$"lat",$"long",$"wayId",$"buffers._1".as("inBuff"),$"buffers._2".as("outBuff"))
  
verticesDF.show(1,false)

// COMMAND ----------

//unique wayIds of the edges
val nodesWayId = splittedEdges.map{case(edge, srcVertex, dstVertex) => edge.attr._1}.toDF("nodesWayId").dropDuplicates() 


// COMMAND ----------

// Only vertices which have a wayId in their Map that is not included in any edge
// Dead end means there are no other intersection vertex in the way
val verticesWithDeadEndWays = verticesDF.join(nodesWayId, $"nodesWayId" === $"wayId", "leftanti") 


// COMMAND ----------

//convert df to rdd to be joined later with the rest of the vertices
import scala.collection.mutable.WrappedArray
val verticesWithDeadEndWaysRDD = verticesWithDeadEndWays.rdd.map(row => (row.getLong(0),(row.getDouble(1),row.getDouble(2),Map(row.getLong(3)-> (row.getAs[WrappedArray[(Long, Double, Double)]](4).array,row.getAs[WrappedArray[(Long, Double, Double)]](5).array)))))



// COMMAND ----------

// for a node appearing in different ways, returns one vertex for each way
val verticesWithSharedWays = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => 
  {
    val srcVertex1 = (srcVertex._1,(srcVertex._2._1,srcVertex._2._2,Map(edge.attr._1 -> srcVertex._2._3(edge.attr._1))))
    val dstVertex1 = (dstVertex._1,(dstVertex._2._1,dstVertex._2._2,Map(edge.attr._1 -> dstVertex._2._3(edge.attr._1))))

    Array(srcVertex1) ++ Array(dstVertex1)
  }}.distinct()


// COMMAND ----------

//union of verticesWithDeadEndWaysRDD and verticesWithSharedWays and reduced adding the maps 
val allVertices = verticesWithSharedWays.union(verticesWithDeadEndWaysRDD).reduceByKey((a,b) => (a._1, a._2, a._3 ++ b._3)) 
allVertices.count()

// COMMAND ----------

dbutils.fs.mkdirs("/_checkpoint1")

// COMMAND ----------

sc.setCheckpointDir("/_checkpoint1") // just a directory in distributed file system
allVertices.checkpoint()
segmentedEdges.checkpoint()

// COMMAND ----------

val coarsened_graph_100 = Graph(allVertices, segmentedEdges)
