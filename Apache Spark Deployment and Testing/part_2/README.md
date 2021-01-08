Before running the program, set "spark-defaults.conf" in spark/conf/ with the following values

spark.master                     spark://node0:7077

spark.driver.memory              30g

spark.executor.memory            30g

spark.executor.cores             5

spark.task.cpus                  1

spark.eventLog.enabled           true

spark.eventLog.dir               hdfs://10.10.1.1:9000/spark-events

spark.history.fs.logDirectory    hdfs://10.10.1.1:9000/spark-events

How to run the scripts:
./bin/spark-submit (scirp path) --input-path (input path) --output-path (output path)

e.g.
./bin/spark-submit ./sort_rdd.py --input-path hdfs://10.10.1.1:9000/sort/export.csv --output-path hdfs://10.10.1.1:9000/result

you can execute the shell script run.sh from the spark home directory like:
./part_2/run.sh hdfs://10.10.1.1:9000/sort/export.csv output-path hdfs://10.10.1.1:9000/result
