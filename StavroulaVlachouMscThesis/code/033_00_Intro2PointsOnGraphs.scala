// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Notebooks structure and necessary libraries 
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

// COMMAND ----------

// MAGIC %md
// MAGIC The common notebooks are:
// MAGIC 
// MAGIC **1. 033_01_OSMtoGraphXUppsalaTiny:** Construction of a road graph from OpenStreetMap (OSM) data with GraphX and finer partitions for a small area in Uppsala. 
// MAGIC 
// MAGIC **2. 033_02_OSMtoGraphX_LT:** Construction of a road graph corresponding to Lithuania's road network from OSM data with GraphX. Ingestion of OSM data with methods from the ```osm-parquetizer``` project; suitable for big data. Further segmentation.

// COMMAND ----------

// MAGIC %md 
// MAGIC The project's open source code regarding Rafailia's part is structures as follows:
// MAGIC 
// MAGIC **1. 034_01_MapMatching_with_GeoMatch_UppsalaTiny:** GeoMatch: Map-matching OSM nodes to OSM ways (showcase)
// MAGIC 
// MAGIC **2. 034_02_MapMatching_on_a_Graph_UppsalaTiny:** GeoMatch: Map-matching OSM nodes to a road graph G0. The latter is constructed by a discretization of the road network provided by OSM. 
// MAGIC 
// MAGIC **3. 034_03_MapMatching_on_a_Graph_LT:** GeoMatch: Map-matching events of interest (vehicle collisions) onto Lithuania's road graph G0. Revisit end of the notebook after ```034_06SimulatingArrivalTimesNHPP_Inversion``` for the generation of location for each time variate simulated for the NHPP.
// MAGIC 
// MAGIC **4. 034_04_MapMatching_on_a_G1_LT:** GeoMatch: Map-matching events of interest (vehicle collisions) onto Lithuania's coarsened road graph G1 (under a distance threshold of 100 meters).
// MAGIC 
// MAGIC **5. 034_05DistributionOfStates:** The conditional/posterior distributions of the states given a time unit and the distribution of the states independent of time. 
// MAGIC 
// MAGIC **6. 034_06SimulatingArrivalTimesNHPP_Inversion:** Simulation of the arrival times of a NHPP from one or more realisations.    

// COMMAND ----------

// MAGIC %md
// MAGIC The project's open source code regarding Virginia's part is structured as follows:
// MAGIC 
// MAGIC **1. 035_01_Arcgis_coordinates_transformation:** Transformation of coordinates using Arcgis Runtime library.
// MAGIC 
// MAGIC **2. 035_02_Segmentation_municipalities_Magellan:** Magellan: locating the accidents within each municipality.
// MAGIC 
// MAGIC **3. 035_03_Visualization_municipalities:** Visualizations of accidents in municipalities using Python.
// MAGIC 
// MAGIC **4. 035_04_MapMatching_intersections:** GeoMatch: map-matching accidents with their closest intersection and measuring the distance between them.
// MAGIC 
// MAGIC **5. 035_05_UndirectedG0:** Undirected graph from the topology road graph created using Open Street Maps (OSM) data. 
// MAGIC 
// MAGIC **6. 035_06_ConnectedComponent_PageRank:** Connected component alogrithm is applied to undirected G0 together with pagerank algorithm. 
// MAGIC 
// MAGIC **7. 035_07_PoissonRegression:** Poisson regression on the number of accidents based on different factors.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Maven libraries that need to be installed in the cluster
// MAGIC com.graphhopper:map-matching:0.6.0
// MAGIC 
// MAGIC io.spray:spray-json_2.11:1.3.4
// MAGIC 
// MAGIC org.openstreetmap.osmosis:osmosis-osm-binary:0.45
// MAGIC 
// MAGIC org.openstreetmap.osmosis:osmosis-pbf:0.45
// MAGIC 
// MAGIC org.openstreetmap.osmosis:osmosis-core:0.45
// MAGIC 
// MAGIC com.esri.geometry:esri-geometry-api:2.1.0
// MAGIC 
// MAGIC org.cusp.bdi.gm.GeoMatch

// COMMAND ----------

// MAGIC %md
// MAGIC #### PyPI libraries that need to be installed in the cluster
// MAGIC gmaps
// MAGIC 
// MAGIC geopy
// MAGIC 
// MAGIC pymc3
// MAGIC 
// MAGIC plotly==5.4.0
