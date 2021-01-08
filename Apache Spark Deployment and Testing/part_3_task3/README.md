Before running the program, set "spark-defaults.conf" in spark/conf/ with the following values. We will try to set the required values for the given run in the shell script as well

spark.master                            spark://node0:7077  
spark.driver.memory                30g  
spark.executor.memory           30g  
spark.executor.cores               5  
spark.task.cpus                       1  
spark.eventLog.enabled          true  
spark.eventLog.dir                   hdfs://10.10.1.1:9000/spark-events  
spark.history.fs.logDirectory    hdfs://10.10.1.1:9000/spark-events  
spark.local.dir                          /mnt/data/temp   

run.sh expects two parameters: the first parameter the input path, the second parameter the output path. It will then pass that into the spark-submit command like so:
./part_3_task3/run.sh hdfs://10.10.1.1:9000/Pagerank2/link* hdfs://10.10.1.1:9000/pagerank_result2/
./bin/spark-submit --master spark://10.10.1.1:7077 --conf spark.local.dir=/mnt/data --conf spark.driver.memory=30g --conf spark.executor.memory=30g --conf spark.executor.cores=5 --conf spark.task.cpus=1 ./part_3_task3/pagerank_task3.py --input-path hdfs://10.10.1.1:9000/Pagerank2/link* --output-path hdfs://10.10.1.1:9000/pagerank_result2/


