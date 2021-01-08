import argparse
from pyspark.sql import SparkSession

def parse_args():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='csv file path')
    parser.add_argument('--output-path', required=True, help='result path')
    return parser.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder.appName("task2 spark").getOrCreate()

    # read in the csv file
    export_df = spark.read.csv(args.input_path, header = True, inferSchema = True)
    # sort by country first, if tie, sort by timestamp
    sorted_df = export_df.orderBy(['cca2', 'timestamp'])
    # cache it
    sorted_df.cache().count()
    # save result to specified location
    sorted_df.repartition(1).write.csv(args.output_path, 'append')

if __name__ == '__main__':
    main()
~                            

