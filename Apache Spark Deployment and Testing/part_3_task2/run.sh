#!/usr/bin/env bash

input=$1
output=$2

if [[ -z "$input" ]]; then
    echo "missing input file path" ; exit 1;
fi	
if [[ -z "$output" ]]; then
    echo "missing output file path"; exit 1;
fi



#/bin/spark-submit ./part_3_task2_pagerank/pagerank_task2.py --input-path hdfs://10.10.1.1:9000/Pagerank2/link* --output-path hdfs://10.10.1.1:9000/pagerank_test
#assumes that the script is in the same location as the python file and that both are in a directory inside of the spark directory
#the the script should be run from within the spark folder but not the task specific folder
./bin/spark-submit --master spark://10.10.1.1:7077 --conf spark.local.dir=/mnt/data --conf spark.driver.memory=30g --conf spark.executor.memory=30g --conf spark.executor.cores=5 --conf spark.task.cpus=1 part_3_task2/pagerank_task2.py --input-path $input --output-path $output 