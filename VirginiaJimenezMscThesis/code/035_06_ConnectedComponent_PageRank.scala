// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Connected component and PageRank in the graph.
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

import sqlContext.implicits._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import scala.collection.JavaConversions._

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.graphx._
import org.graphframes._


import org.apache.spark.graphx.lib.PageRank


// COMMAND ----------

// MAGIC %md
// MAGIC In this notebook, the connected component algorithm is applied to a versioned G0 (with edges in both directions) and we keep the largest component. Later, we applied the PageRank algorithm to the largest component and join the accidents with their matched intersection and the PageRank score.

// COMMAND ----------

//read the graph edges and vertices
val edges_g0_undirected_taggs = spark.read.parquet("/_checkpoint/edges_LT_initial_two_directions_tags_added")
val vertices_g0_undirected_taggs = spark.read.parquet("/_checkpoint/vertices_LT_initial_added_tags")

// COMMAND ----------

import org.apache.spark.graphx.Graph
import org.graphframes.GraphFrame

sc.setCheckpointDir("/_checkpoint") // just a directory in distributed file system
val g0_two_directions_graphframe = GraphFrame(vertices_g0_undirected_taggs, edges_g0_undirected_taggs)
val cc_g0_two_directions_taggs = g0_two_directions_graphframe.connectedComponents.run() 

// COMMAND ----------

cc_g0_two_directions_taggs.write.mode("overwrite").parquet("/_checkpoint/one_connected_component_g0_two_directions")

// COMMAND ----------

val one_cc_g0_two_directions = spark.read.parquet("/_checkpoint/one_connected_component_g0_two_directions")

// COMMAND ----------

one_cc_g0_two_directions.groupBy("component").count().sort(col("count").desc).limit(10).show()

// COMMAND ----------

val cc = vertices_g0_undirected_taggs.join(one_cc_g0_two_directions, vertices_g0_undirected_taggs.col("id") === one_cc_g0_two_directions.col("id"))
                                      .drop(one_cc_g0_two_directions.col("Map")).drop(one_cc_g0_two_directions.col("id"))

// COMMAND ----------

val one_cc_vertices = cc.filter(col("component") === "15389886").drop(col("component")) //the largest connected component

// COMMAND ----------

one_cc_vertices.count()

// COMMAND ----------

val vertices_ids = one_cc_vertices.select($"id")

// COMMAND ----------

val one_cc_edges_tmp = edges_g0_undirected_taggs.join(vertices_ids, vertices_ids.col("id") === edges_g0_undirected_taggs.col("src")).drop(col("id"))
val one_cc_edges = one_cc_edges_tmp.join(vertices_ids, vertices_ids.col("id") === edges_g0_undirected_taggs.col("dst")).drop(col("id"))

// COMMAND ----------

import org.apache.spark.graphx.Graph
import org.graphframes.GraphFrame
import org.apache.spark.graphx._

val r = GraphFrame(one_cc_vertices, one_cc_edges)

// COMMAND ----------

val new_g0 = r.toGraphX

// COMMAND ----------

val pagerank_newg0 = PageRank.run(new_g0, numIter=100, resetProb=0.15).cache()

// COMMAND ----------

val ranks = pagerank_newg0.vertices

// COMMAND ----------

val ranks_DF = ranks.toDF("id", "pageRank")

// COMMAND ----------

ranks_DF.write.mode("overwrite").parquet("/_checkpoint/pagerank_cc_newg0")

// COMMAND ----------

ranks_DF.show(10,false)

// COMMAND ----------

val pagerank = spark.read.parquet("/_checkpoint/pagerank_cc_newg0").toDF("id", "pageRank")

// COMMAND ----------

val top_nodes_pageRank = pagerank.sort(col("pageRank").desc).limit(10)

// COMMAND ----------

top_nodes_pageRank.show()

// COMMAND ----------

val nodes_coordinates = one_cc_vertices.select(col("id").as("node_id"), col("Map._1").as("longitude"), col("Map._2").as("latitude"))
val nodes_topRank_located = top_nodes_pageRank.join(nodes_coordinates, col("id") === col("node_id")).drop(col("node_id")).sort(col("pageRank").desc)
nodes_topRank_located.show(20,false)

// COMMAND ----------

val nodes_topRank_located = top_nodes_pageRank.join(nodes_coordinates, col("id") === col("node_id")).drop(col("node_id")).sort(col("pageRank").desc)


// COMMAND ----------

val acc_inters_ids = spark.read.parquet("dbfs:/datasets/lithuania/acc_inters_ids.parquet")
val acc_inters_ids_new = acc_inters_ids.select(col("acc_id"), split(col("inters_id"),":").getItem(1).as("inters_id"))
val acc_inters_pagerank = acc_inters_ids_new.join(pagerank, acc_inters_ids_new("inters_id") === pagerank("id")).select("acc_id", "inters_id", "pageRank")

// COMMAND ----------

//acc_inters_pagerank.show(4,false)


// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The output of the above command with IDs and locations anonymised is as follows:
// MAGIC 
// MAGIC ```
// MAGIC +------------+----------+------------------+
// MAGIC |acc_id      |inters_id |pageRank          |
// MAGIC +------------+----------+------------------+
// MAGIC |LT20xyABCDEF|177xxxxxxx|1.259926172353153 |
// MAGIC |LT20ghIJKLMN|203yyyyyyy|1.7267425897576583|
// MAGIC |LT20opQRSTUV|278zzzzzzz|1.012061858801204 |
// MAGIC |LT20wxYZABCD|455aaaaaa |0.9969143738356286|
// MAGIC +------------+----------+------------------+
// MAGIC only showing top 4 rows.
// MAGIC ```

// COMMAND ----------

acc_inters_pagerank.coalesce(1).write.mode("overwrite").parquet("dbfs:/datasets/lithuania/acc_inters_pagerank_2") //this will be used for the regression
