// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

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

// MAGIC %fs ls /datasets/osm/uppsala

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

// COMMAND ----------

val allowableWays = Set(
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

val fs = FileSystem.get(new Configuration())
val path = new Path("dbfs:/datasets/osm/uppsala/uppsalaTinyR.pbf")
val file = fs.open(path)

var nodes: ArrayBuffer[Node] = ArrayBuffer()
var ways: ArrayBuffer[Way] = ArrayBuffer()
var relations: ArrayBuffer[Relation] = ArrayBuffer()

val osmosisReader = new OsmosisReader(file)
  osmosisReader.setSink(new Sink {
    override def process(entityContainer: EntityContainer): Unit = {
      
      if (entityContainer.getEntity.getType != EntityType.Bound) {
        val entity = entityContainer.getEntity
        entity match {
          case node: Node => nodes += node
          case way: Way => {
            val tagSet = way.getTags.map(_.getValue).toSet
            if ( !(tagSet & allowableWays).isEmpty ) {
              // way has at least one tag of interest
              ways += way
            }
          }
          case relation: Relation => relations += relation
        }
      }
    }

    override def initialize(map: java.util.Map[String, AnyRef]): Unit = {
      nodes = ArrayBuffer()
      ways = ArrayBuffer()
      relations = ArrayBuffer()
    }

    override def complete(): Unit = {}

    override def release(): Unit = {} // this is 4.6 method
    
    def close(): Unit = {}
  })

osmosisReader.run() 

// COMMAND ----------

case class WayEntry(wayId: Long, tags: Array[String], nodes: Array[Long])
case class NodeEntry(nodeId: Long, latitude: Double, longitude: Double, tags: Array[String])

// COMMAND ----------

//convert the nodes array to Dataset
val nodeDS = nodes.map{node => 
  NodeEntry(node.getId, 
       node.getLatitude, 
       node.getLongitude, 
       node.getTags.map(_.getValue).toArray
)}.toDS

// COMMAND ----------

nodeDS.count()

// COMMAND ----------

nodeDS.show(5, false)

// COMMAND ----------

//convert the ways array to Dataset
val wayDS = ways.map(way => 
  WayEntry(way.getId,
      way.getTags.map(_.getValue).toArray,
      way.getWayNodes.map(_.getNodeId).toArray)
).toDS.cache

// COMMAND ----------

wayDS.count()

// COMMAND ----------

wayDS.show(9, false)

// COMMAND ----------

import org.apache.spark.sql.functions.explode

val nodeCounts = wayDS
                    .select(explode('nodes).as("node"))
                    .groupBy('node).count

nodeCounts.show(5)


// COMMAND ----------

val intersectionNodes = nodeCounts.filter('count >= 2).select('node.alias("intersectionNode"))

// COMMAND ----------

intersectionNodes.count() //there are 6 intersections in this area 

// COMMAND ----------

val true_intersections = intersectionNodes

// COMMAND ----------

true_intersections.count

// COMMAND ----------

intersectionNodes.show()

// COMMAND ----------

val distinctNodesWays = wayDS.flatMap(_.nodes).distinct //the distinct nodes within the ways 

// COMMAND ----------

distinctNodesWays.printSchema

// COMMAND ----------

distinctNodesWays.count()

// COMMAND ----------

distinctNodesWays.show(5)

// COMMAND ----------

val wayNodes = nodeDS.as("nodes") //nodes that are in a way + nodes info from nodeDS
  .joinWith(distinctNodesWays.as("ways"), $"ways.value" === $"nodes.nodeId")
  .map(_._1).cache

// COMMAND ----------

wayNodes.printSchema

// COMMAND ----------

wayNodes.count()

// COMMAND ----------

wayNodes.show(5, false) //the nodes and their coordinates that participate in the ways 25734373, 312352

// COMMAND ----------

wayDS.printSchema

// COMMAND ----------

val intersectionSetVal = intersectionNodes.as[Long].collect.toSet; //turn intersectionNodes to Set 

// COMMAND ----------

//new 
import org.apache.spark.sql.functions.{collect_list, map, udf}
import org.apache.spark.sql.functions._

// You could try using `getItem` methods
// I assume that each "nodes" sequence contains at least one node
// We do not really need first and last elements from the sequence and when combining with original nodes, just we assign them "true"

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

val fake_intersections = first_and_last_nodes.select($"node").distinct().withColumnRenamed("node", "value")

// // Turn intersection set into a dataset to join (all values must be unique)
// //val intersections = intersectionSetVal.toSeq.toDF("value")
val intersections = intersectionNodes.union(fake_intersections).distinct      //virginia
 
val wayNodesLocated = nodes.join(wayNodes, wayNodes.col("nodeId") === nodes.col("node")).select($"wayId", $"node", $"latitude", $"longitude")


// case class MappedWay(wayId: Long, labels: Seq[Map[Long, Boolean]])
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

labeledWays.printSchema

// COMMAND ----------

labeledWays.select("wayId", "labeledNodes").show(9, false)

// COMMAND ----------

case class Intersection(OSMId: Long , latitude: Double, longitude: Double, inBuf: ArrayBuffer[(Long, Double, Double)], outBuf: ArrayBuffer[(Long, Double, Double)])

// COMMAND ----------

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
        //intersections += new Intersection(-1L, 0D, 0D, ArrayBuffer[(Long, Double, Double)](), currentBuffer) //not sure about this but I'll keep it by now
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

//segmentedWays contains two columns:
  //_1: wayId
  //_2: Array[(nodeId, latitude, longitude, inBuff, outBuff)] for each intersection node in the way

// COMMAND ----------

val schema = "array<struct<nodeId:bigint,latitude:double,longitude:double,inBuff:array<struct<nodeId:bigint,latitude:double,longitude:double>>,outBuff:array<struct<nodeId:bigint,latitude:double,longitude:double>>>>"
segmentedWays.select($"_1".alias("wayId"), $"_2".cast(schema).alias("nodeInfo")).printSchema()

// COMMAND ----------

segmentedWays.show(2, false)

// COMMAND ----------

//The nested structure of the segmentedWays is unwrapped
val waySegmentDS = segmentedWays
.flatMap(way => way._2.map(node => (way._1, node))) 
// for each (wayId, Array(IntersectionNode) => (wayId, IntersectionNode)

// COMMAND ----------

waySegmentDS.printSchema

// COMMAND ----------

waySegmentDS.show(5, false)

// COMMAND ----------

import scala.collection.immutable.Map

// COMMAND ----------

//returns the intersection nodes with the ways where they appear mapped with the nodes in those ways (inBuff, outBuff) 
val intersectionVertices = waySegmentDS
  .map(way => 
   //nodeId     latitude   longitude      wayId      inBuff      outBuff
   (way._2._1, (way._2._2, way._2._3, Map(way._1 -> (way._2._4, way._2._5))))) 
  .rdd
  //                     latitude, long, Map(wayId, inBuff, outBuff)
  .reduceByKey((a,b) => (a._1,     a._2, a._3 ++ b._3)) 

//intersectionVertices =  RDD[(nodeId, (latitude, longitude, wayMap(wayId -> inBuff, outBuff)))]

// COMMAND ----------

intersectionVertices.map(vertex => (vertex._1, vertex._2._1, vertex._2._2)).toDF("vertexId", "latitude", "longitude").write.mode("overwrite").parquet("dbfs:/graphs/uppsala/vertices")

// COMMAND ----------

intersectionVertices.count()

// COMMAND ----------

intersectionVertices.take(10)

// COMMAND ----------

val edges = segmentedWays
  .filter(way => way._2.length > 1) //ways with more than one intersections
  .flatMap{ case (wayId, nodes_info) => {  
             nodes_info.sliding(2) // For each way it takes nodes in pairs
               .flatMap(segment => //segment is the pair of two nodes
                   List(Edge(segment(0)._1, segment(1)._1, wayId))
               )
   }}

// COMMAND ----------

edges.map(edge => (edge.srcId, edge.dstId)).toDF("src","dst").write.mode("overwrite").parquet("dbfs:/graphs/uppsala/edges")

// COMMAND ----------

edges.printSchema

// COMMAND ----------

edges.count

// COMMAND ----------

val roadGraph = Graph(intersectionVertices, edges.rdd).cache

//intersectionVertices =  RDD[(nodeId, (latitude, longitude, wayMap(wayId -> inBuff, outBuff)))]
//edges = srcId, dstId, attribute (attribute is the wayId)


// COMMAND ----------

roadGraph.edges.take(10).foreach(println)

// COMMAND ----------

package d3
// We use a package object so that we can define top level classes like Edge that need to be used in other cells
// This was modified by Ivan Sadikov to make sure it is compatible the latest databricks notebook

import org.apache.spark.sql._
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

case class Edge(src: String, dest: String, count: Long)

case class Node(name: String)
case class Link(source: Int, target: Int, value: Long)
case class Graph(nodes: Seq[Node], links: Seq[Link])

object graphs {
// val sqlContext = SQLContext.getOrCreate(org.apache.spark.SparkContext.getOrCreate())  /// fix
val sqlContext = SparkSession.builder().getOrCreate().sqlContext
import sqlContext.implicits._
  
def force(clicks: Dataset[Edge], height: Int = 100, width: Int = 960): Unit = {
  val data = clicks.collect()
  val nodes = (data.map(_.src) ++ data.map(_.dest)).map(_.replaceAll("_", " ")).toSet.toSeq.map(Node)
  val links = data.map { t =>
    Link(nodes.indexWhere(_.name == t.src.replaceAll("_", " ")), nodes.indexWhere(_.name == t.dest.replaceAll("_", " ")), t.count / 20 + 1)
  }
  showGraph(height, width, Seq(Graph(nodes, links)).toDF().toJSON.first())
}

/**
 * Displays a force directed graph using d3
 * input: {"nodes": [{"name": "..."}], "links": [{"source": 1, "target": 2, "value": 0}]}
 */
def showGraph(height: Int, width: Int, graph: String): Unit = {

displayHTML(s"""
<style>

.node_circle {
  stroke: #777;
  stroke-width: 1.3px;
}

.node_label {
  pointer-events: none;
}

.link {
  stroke: #777;
  stroke-opacity: .2;
}

.node_count {
  stroke: #777;
  stroke-width: 1.0px;
  fill: #999;
}

text.legend {
  font-family: Verdana;
  font-size: 13px;
  fill: #000;
}

.node text {
  font-family: "Helvetica Neue","Helvetica","Arial",sans-serif;
  font-size: 17px;
  font-weight: 200;
}

</style>

<div id="clicks-graph">
<script src="//d3js.org/d3.v3.min.js"></script>
<script>

var graph = $graph;

var width = $width,
    height = $height;

var color = d3.scale.category20();

var force = d3.layout.force()
    .charge(-700)
    .linkDistance(180)
    .size([width, height]);

var svg = d3.select("#clicks-graph").append("svg")
    .attr("width", width)
    .attr("height", height);
    
force
    .nodes(graph.nodes)
    .links(graph.links)
    .start();

var link = svg.selectAll(".link")
    .data(graph.links)
    .enter().append("line")
    .attr("class", "link")
    .style("stroke-width", function(d) { return Math.sqrt(d.value); });

var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter().append("g")
    .attr("class", "node")
    .call(force.drag);

node.append("circle")
    .attr("r", 10)
    .style("fill", function (d) {
    if (d.name.startsWith("other")) { return color(1); } else { return color(2); };
})

node.append("text")
      .attr("dx", 10)
      .attr("dy", ".35em")
      .text(function(d) { return d.name });
      
//Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
force.on("tick", function () {
    link.attr("x1", function (d) {
        return d.source.x;
    })
        .attr("y1", function (d) {
        return d.source.y;
    })
        .attr("x2", function (d) {
        return d.target.x;
    })
        .attr("y2", function (d) {
        return d.target.y;
    });
    d3.selectAll("circle").attr("cx", function (d) {
        return d.x;
    })
        .attr("cy", function (d) {
        return d.y;
    });
    d3.selectAll("text").attr("x", function (d) {
        return d.x;
    })
        .attr("y", function (d) {
        return d.y;
    });
});
</script>
</div>
""")
}
  
  def help() = {
displayHTML("""
<p>
Produces a force-directed graph given a collection of edges of the following form:</br>
<tt><font color="#a71d5d">case class</font> <font color="#795da3">Edge</font>(<font color="#ed6a43">src</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">dest</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">count</font>: <font color="#a71d5d">Long</font>)</tt>
</p>
<p>Usage:<br/>
<tt><font color="#a71d5d">import</font> <font color="#ed6a43">d3._</font></tt><br/>
<tt><font color="#795da3">graphs.force</font>(</br>
&nbsp;&nbsp;<font color="#ed6a43">height</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">width</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">clicks</font>: <font color="#795da3">Dataset</font>[<font color="#795da3">Edge</font>])</tt>
</p>""")
  }
}

// COMMAND ----------

import d3._
import org.apache.spark.sql.functions.lit
val G0 = roadGraph.edges.toDF().select($"srcId".as("src"), $"dstId".as("dest"),  lit(1L).as("count"))

d3.graphs.force(
  height = 800,
  width = 800,
  clicks = G0.as[d3.Edge])

// COMMAND ----------

import com.esri.core.geometry.GeometryEngine.geodesicDistanceOnWGS84
import com.esri.core.geometry.Point

// COMMAND ----------

val weightedRoadGraph = roadGraph.mapTriplets{triplet => //mapTriplets gives EdgeTriplet https://spark.apache.org/docs/2.3.1/api/java/org/apache/spark/graphx/EdgeTriplet.html
  def dist(lat1: Double, long1: Double, lat2: Double, long2: Double): Double = {
    val p1 = new Point(long1, lat1)
    val p2 = new Point(long2, lat2)
    geodesicDistanceOnWGS84(p1, p2)
  }
  
  //A triplet represents an edge along with the vertex attributes of its neighboring vertices (srcAttr, dstAttr)
  //triplet.attr is the same as edge.attr
  val wayNodesInBuff = triplet.dstAttr._3(triplet.attr)._1 //dstAttr is the vertex attribute (latitude, longitude, wayMap(wayId -> inBuff, outBuff))
  // inBuff -> array(nodeId, lat, long)
  
  if (wayNodesInBuff.isEmpty) {
      (triplet.attr, dist(triplet.srcAttr._1, triplet.srcAttr._2, triplet.dstAttr._1, triplet.dstAttr._2))
  
  } else {
      var distance: Double = 0.0

      //adds the distance between the src node and the first node in the InBuff
      distance += dist(triplet.srcAttr._1, triplet.srcAttr._2, wayNodesInBuff(0)._2, wayNodesInBuff(0)._3 )
    
     //more than one node in the inBuffer
      if (wayNodesInBuff.length > 1) {
        //adds the distance between every pair of nodes inside the inBuffer 
        distance += wayNodesInBuff.sliding(2).map{
        buff => dist(buff(0)._2, buff(0)._3, buff(1)._2, buff(1)._3)}
        .reduce(_ + _)
     }
    
      //adds the distance between the dst node and the last node in the InBuff
      distance += dist(wayNodesInBuff.last._2, wayNodesInBuff.last._3, triplet.dstAttr._1, triplet.dstAttr._2)

      (triplet.attr, distance)
    }
  
}.cache

// COMMAND ----------

weightedRoadGraph.edges.count()

// COMMAND ----------

weightedRoadGraph.edges.take(8).foreach(println)

// COMMAND ----------

weightedRoadGraph.vertices.count()

// COMMAND ----------

weightedRoadGraph.vertices.map(node => node._1).take(11)

// COMMAND ----------

weightedRoadGraph.vertices.take(11)

// COMMAND ----------

import org.apache.spark.graphx.{Edge => Edges}
val splittedEdges = weightedRoadGraph.triplets.flatMap{triplet => {
  def dist(lat1: Double, long1: Double, lat2: Double, long2: Double): Double = {
    val p1 = new Point(long1, lat1)
    val p2 = new Point(long2, lat2)
    geodesicDistanceOnWGS84(p1, p2)
  }
  val maxDist = 200
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

// Taking each edge and its reverse
val segmentedEdges = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => Array(edge) ++ Array(Edges(edge.dstId, edge.srcId, edge.attr))}
segmentedEdges.count() 

// COMMAND ----------

segmentedEdges.take(36).foreach(println)


// COMMAND ----------

// Taking the individual vertices
val segmentedVertices = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => Array(srcVertex) ++ Array(dstVertex)}

segmentedVertices.map(node => node._1).distinct().take(16)
//25812013, 455006648, 25735257, 3431600977, 3963994985, 3067700641, 312353, 312363, 3067700668, 25734373, 2206536278) initial nodes 

// COMMAND ----------

// Converting the vertices to a df
val verticesDF = segmentedVertices.toDF("nodeId","attr").select($"nodeId",$"attr._1".as("lat"),$"attr._2".as("long"),explode($"attr._3"))
    .withColumnRenamed("key","wayId").withColumnRenamed("value","buffers")
    .select($"nodeId",$"lat",$"long",$"wayId",$"buffers._1".as("inBuff"),$"buffers._2".as("outBuff"))
  
verticesDF.show(24,false)

// COMMAND ----------

//unique wayIds of the edges
val nodesWayId = splittedEdges.map{case(edge, srcVertex, dstVertex) => edge.attr._1}.toDF("nodesWayId").dropDuplicates() 
nodesWayId.show(10)

// COMMAND ----------

// Only vertices which have a wayId in their Map that is not included in any edge
// Dead end means there are no other intersection vertex in the way
val verticesWithDeadEndWays = verticesDF.join(nodesWayId, $"nodesWayId" === $"wayId", "leftanti") //leftanti is a special join which returns the rows that don't match
verticesWithDeadEndWays.show(20,false)

// COMMAND ----------

//convert df to rdd to be joined later with the rest of the vertices
import scala.collection.mutable.WrappedArray
val verticesWithDeadEndWaysRDD = verticesWithDeadEndWays.rdd.map(row => (row.getLong(0),(row.getDouble(1),row.getDouble(2),Map(row.getLong(3)-> (row.getAs[WrappedArray[(Long, Double, Double)]](4).array,row.getAs[WrappedArray[(Long, Double, Double)]](5).array)))))

verticesWithDeadEndWaysRDD.take(10)

// COMMAND ----------

// for a node appearing in different ways, returns one vertex for each way
val verticesWithSharedWays = splittedEdges.flatMap{case(edge, srcVertex, dstVertex) => 
  {
    val srcVertex1 = (srcVertex._1,(srcVertex._2._1,srcVertex._2._2,Map(edge.attr._1 -> srcVertex._2._3(edge.attr._1))))
    val dstVertex1 = (dstVertex._1,(dstVertex._2._1,dstVertex._2._2,Map(edge.attr._1 -> dstVertex._2._3(edge.attr._1))))

    Array(srcVertex1) ++ Array(dstVertex1)
  }}.distinct()


verticesWithSharedWays.take(10)


// COMMAND ----------

//union of verticesWithDeadEndWaysRDD and verticesWithSharedWays and reduced adding the maps 
val allVertices = verticesWithSharedWays.union(verticesWithDeadEndWaysRDD).reduceByKey((a,b) => (a._1, a._2, a._3 ++ b._3)) 
allVertices.count()

// COMMAND ----------

import org.apache.spark.graphx.Graph
val segmentedGraph = Graph(allVertices, segmentedEdges).cache()

// COMMAND ----------

//allVertices.map(vertex => (vertex._1,(vertex._2._1, vertex._2._2))).toDF("id","coordinates").write.mode("overwrite").parquet("dbfs:/graphs/uppsala/vertices")

// COMMAND ----------

// spark.read.parquet("dbfs:/graphs/uppsala/edges").rdd.take(1)

// COMMAND ----------

segmentedGraph.vertices.take(11) 

// COMMAND ----------

segmentedGraph.edges.count

// COMMAND ----------

segmentedGraph.edges.take(18).foreach(println)

// COMMAND ----------

val G1 = segmentedGraph.edges.toDF().select($"srcId".as("src"), $"dstId".as("dest"), lit(1L).as("count"))

d3.graphs.force(
  height = 1000,
  width = 1000,
  clicks = G1.as[d3.Edge])
