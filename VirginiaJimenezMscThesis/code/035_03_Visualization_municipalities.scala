// Databricks notebook source
// MAGIC %md
// MAGIC ScaDaMaLe Course [site](https://lamastex.github.io/scalable-data-science/sds/3/x/) and [book](https://lamastex.github.io/ScaDaMaLe/index.html)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Visualization of the Segmentation by municipalities using Python.
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
// MAGIC ### Visualizing relative frequency of the accidents.

// COMMAND ----------

// MAGIC %python
// MAGIC # Reading accident frequencies for each municipality previously obtained
// MAGIC 
// MAGIC import pyspark
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql.types import StructType, StringType, DoubleType
// MAGIC 
// MAGIC schema = StructType() \
// MAGIC       .add("municipality", StringType(), True) \
// MAGIC       .add("frequency", DoubleType(), True)
// MAGIC 
// MAGIC municipality_freq = spark.read.format("csv").option("header", True).schema(schema).load("dbfs:/datasets/lithuania/municipalities_freq.csv")

// COMMAND ----------

// MAGIC %python
// MAGIC municipality_freq.show(1000)

// COMMAND ----------

// MAGIC %python
// MAGIC # Calculating colors
// MAGIC 
// MAGIC # https://matplotlib.org/stable/tutorials/colors/colormaps.html
// MAGIC from matplotlib.cm import viridis
// MAGIC from matplotlib.colors import to_hex
// MAGIC 
// MAGIC min_freq = municipality_freq.agg({"frequency":"min"}).collect()[0][0]
// MAGIC max_freq = municipality_freq.agg({"frequency":"max"}).collect()[0][0]
// MAGIC freq_range = max_freq - min_freq
// MAGIC 
// MAGIC def calculate_color(row):
// MAGIC     freq = row["frequency"]
// MAGIC     """
// MAGIC     Convert the freq to a color
// MAGIC     """
// MAGIC     # make freq a number between 0 and 1
// MAGIC     normalized_freq = (freq - min_freq) / freq_range
// MAGIC     
// MAGIC     # This is because in viridis colormap, darker is lower values and we want the opposite
// MAGIC     inverse_freq = 1-normalized_freq
// MAGIC     
// MAGIC     # transform the freq coefficient to a matplotlib color
// MAGIC     mpl_color = viridis(inverse_freq)
// MAGIC 
// MAGIC     # transform from a matplotlib color to a valid CSS color
// MAGIC     gmaps_color = to_hex(mpl_color, keep_alpha=False)
// MAGIC 
// MAGIC     return (row["municipality"],gmaps_color)
// MAGIC 
// MAGIC # Calculate a color for each district
// MAGIC colors = municipality_freq.rdd.map(lambda row: calculate_color(row)).collectAsMap()

// COMMAND ----------

//Temporary copy of geojson so python can read it
dbutils.fs.cp("dbfs:/datasets/magellan/municipalities.geojson", "file:/databricks/driver/")

// COMMAND ----------

// MAGIC %python
// MAGIC # Reading and processing geojson (map and borders)
// MAGIC 
// MAGIC import json
// MAGIC import gmaps
// MAGIC import gmaps.datasets
// MAGIC import gmaps.geojson_geometries
// MAGIC from ipywidgets.embed import embed_minimal_html
// MAGIC 
// MAGIC gmaps.configure(api_key="AIzaSyDEHHgMMS33M5AT8lav2Q-sem5KOyFx9Sc") # Your Google API key
// MAGIC 
// MAGIC # municipalities / Savivaldybės
// MAGIC municipalities = json.load(open('municipalities.geojson', 'r'))
// MAGIC 
// MAGIC # Removing municipality capitals
// MAGIC list_to_remove = []
// MAGIC i = 0
// MAGIC for feature in municipalities['features']:
// MAGIC   if feature["geometry"]["type"] != "Polygon":
// MAGIC     list_to_remove.append(i)
// MAGIC   i+=1
// MAGIC   
// MAGIC # Removing what was found before
// MAGIC for index in sorted(list_to_remove, reverse=True):
// MAGIC     del municipalities['features'][index]
// MAGIC     

// COMMAND ----------

// MAGIC %python
// MAGIC # Order the colors by the geojson order
// MAGIC 
// MAGIC ordered_colors = []
// MAGIC for feature in municipalities['features']:
// MAGIC   municipality = feature['properties']['name']
// MAGIC   color = colors[municipality]
// MAGIC   ordered_colors.append(color)

// COMMAND ----------

// MAGIC %python
// MAGIC from pylab import *
// MAGIC 
// MAGIC # Generating map
// MAGIC 
// MAGIC fig = gmaps.figure()
// MAGIC freq_layer = gmaps.geojson_layer(
// MAGIC     municipalities,
// MAGIC     fill_color=ordered_colors,
// MAGIC     fill_opacity=0.8,
// MAGIC     stroke_color='black',
// MAGIC     stroke_opacity=1.0,
// MAGIC     stroke_weight=0.2)
// MAGIC fig.add_layer(freq_layer)
// MAGIC 
// MAGIC embed_minimal_html("export.html", views=[fig])

// COMMAND ----------

// MAGIC %python
// MAGIC # Adding color legend to map
// MAGIC cmap = cm.get_cmap('viridis', 20)
// MAGIC 
// MAGIC gradient = ""
// MAGIC for i in reversed(range(cmap.N)):
// MAGIC     rgba = cmap(i)
// MAGIC     # rgb2hex accepts rgb or rgba
// MAGIC     gradient = gradient + "," + matplotlib.colors.rgb2hex(rgba)
// MAGIC 
// MAGIC # Removing first comma
// MAGIC gradient = gradient[1:]
// MAGIC 
// MAGIC html_file_content = open("export.html", 'r').read()\
// MAGIC                     .replace("</head>", """<style>
// MAGIC                                  .legend {
// MAGIC                                    max-width: 430px;
// MAGIC                                  }
// MAGIC                                   .legend div{
// MAGIC                                    background: linear-gradient(to right, """ + gradient + """);
// MAGIC                                    border-radius: 4px;
// MAGIC                                    padding: 10px;
// MAGIC                                  }
// MAGIC 
// MAGIC                                 .legend p {
// MAGIC                                   text-align: justify;
// MAGIC                                   text-justify: inter-word;
// MAGIC                                   margin: 0px;
// MAGIC                                       margin-block-start: 0em;
// MAGIC                                     margin-block-end: 0em;
// MAGIC                                     height: 1em;
// MAGIC                                 }
// MAGIC                                 .legend p:after {
// MAGIC                                   content: "";
// MAGIC                                   display: inline-block;
// MAGIC                                   width: 100%;
// MAGIC                                 }
// MAGIC                               </style>
// MAGIC                             </head>""")\
// MAGIC                     .replace("</body>","""
// MAGIC                           <h2>Relative frequency of accidents</h2>
// MAGIC                           <div class="legend">
// MAGIC                             <p>""" + str(round(min_freq,2)) + " " + str(round(max_freq,2)) +"""</p>
// MAGIC                             <div></div>
// MAGIC                           </div>
// MAGIC                         </body>""")

// COMMAND ----------

// MAGIC %python
// MAGIC # !!!!!!!!!!!!!!!!!!!!!
// MAGIC # Can only be run once per cluster restart
// MAGIC displayHTML(html_file_content)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Visualizing accidents normalised by population

// COMMAND ----------

// MAGIC %python
// MAGIC # Reading district frequencies previously obtained
// MAGIC 
// MAGIC import pyspark
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql.types import StructType, StringType, DoubleType
// MAGIC 
// MAGIC schema = StructType() \
// MAGIC       .add("municipality", StringType(), True) \
// MAGIC       .add("frequency", DoubleType(), True)
// MAGIC 
// MAGIC municipalities_pop = spark.read.format("csv").option("header", True).schema(schema).load("dbfs:/datasets/lithuania/municipalities_pop.csv")

// COMMAND ----------

// MAGIC %python
// MAGIC # Calculating colors
// MAGIC 
// MAGIC # https://matplotlib.org/stable/tutorials/colors/colormaps.html
// MAGIC from matplotlib.cm import viridis
// MAGIC from matplotlib.colors import to_hex
// MAGIC 
// MAGIC min_freq = municipalities_pop.agg({"frequency":"min"}).collect()[0][0]
// MAGIC max_freq = municipalities_pop.agg({"frequency":"max"}).collect()[0][0]
// MAGIC freq_range = max_freq - min_freq
// MAGIC 
// MAGIC def calculate_color(row):
// MAGIC     freq = row["frequency"]
// MAGIC     """
// MAGIC     Convert the freq to a color
// MAGIC     """
// MAGIC     # make freq a number between 0 and 1
// MAGIC     normalized_freq = (freq - min_freq) / freq_range
// MAGIC     
// MAGIC     # This is because in viridis colormap, darker is lower values and we want the opposite
// MAGIC     inverse_freq = 1-normalized_freq
// MAGIC     
// MAGIC     # transform the freq coefficient to a matplotlib color
// MAGIC     mpl_color = viridis(inverse_freq)
// MAGIC 
// MAGIC     # transform from a matplotlib color to a valid CSS color
// MAGIC     gmaps_color = to_hex(mpl_color, keep_alpha=False)
// MAGIC 
// MAGIC     return (row["municipality"],gmaps_color)
// MAGIC 
// MAGIC # Calculate a color for each district
// MAGIC colors = municipalities_pop.rdd.map(lambda row: calculate_color(row)).collectAsMap()

// COMMAND ----------

// MAGIC %python
// MAGIC # Order the colors by the geojson order
// MAGIC 
// MAGIC ordered_colors = []
// MAGIC for feature in municipalities['features']:
// MAGIC   municipality = feature['properties']['name']
// MAGIC   color = colors[municipality]
// MAGIC   ordered_colors.append(color)

// COMMAND ----------

// MAGIC %python
// MAGIC from pylab import *
// MAGIC 
// MAGIC # Generating map
// MAGIC 
// MAGIC fig = gmaps.figure()
// MAGIC freq_layer = gmaps.geojson_layer(
// MAGIC     municipalities,
// MAGIC     fill_color=ordered_colors,
// MAGIC     fill_opacity=0.8,
// MAGIC     stroke_color='black',
// MAGIC     stroke_opacity=1.0,
// MAGIC     stroke_weight=0.2)
// MAGIC fig.add_layer(freq_layer)
// MAGIC 
// MAGIC embed_minimal_html("export.html", views=[fig])

// COMMAND ----------

// MAGIC %python
// MAGIC # Adding color legend to map
// MAGIC cmap = cm.get_cmap('viridis', 20)
// MAGIC 
// MAGIC gradient = ""
// MAGIC for i in reversed(range(cmap.N)):
// MAGIC     rgba = cmap(i)
// MAGIC     # rgb2hex accepts rgb or rgba
// MAGIC     gradient = gradient + "," + matplotlib.colors.rgb2hex(rgba)
// MAGIC 
// MAGIC # Removing first comma
// MAGIC gradient = gradient[1:]
// MAGIC 
// MAGIC html_file_content = open("export.html", 'r').read()\
// MAGIC                     .replace("</head>", """<style>
// MAGIC                                  .legend {
// MAGIC                                    max-width: 430px;
// MAGIC                                  }
// MAGIC                                   .legend div{
// MAGIC                                    background: linear-gradient(to right, """ + gradient + """);
// MAGIC                                    border-radius: 4px;
// MAGIC                                    padding: 10px;
// MAGIC                                  }
// MAGIC 
// MAGIC                                 .legend p {
// MAGIC                                   text-align: justify;
// MAGIC                                   text-justify: inter-word;
// MAGIC                                   margin: 0px;
// MAGIC                                       margin-block-start: 0em;
// MAGIC                                     margin-block-end: 0em;
// MAGIC                                     height: 1em;
// MAGIC                                 }
// MAGIC                                 .legend p:after {
// MAGIC                                   content: "";
// MAGIC                                   display: inline-block;
// MAGIC                                   width: 100%;
// MAGIC                                 }
// MAGIC                               </style>
// MAGIC                             </head>""")\
// MAGIC                     .replace("</body>","""
// MAGIC                           <h2>Accidents normalized by population</h2>
// MAGIC                           <div class="legend">
// MAGIC                             <p>0 """ + str(round(max_freq,4)) +"""</p>
// MAGIC                             <div></div>
// MAGIC                           </div>
// MAGIC                         </body>""")

// COMMAND ----------

// MAGIC %python
// MAGIC # !!!!!!!!!!!!!!!!!!!!!
// MAGIC # Can only be run once per cluster restart
// MAGIC displayHTML(html_file_content)

// COMMAND ----------


