import argparse
from pyspark.context import SparkContext

def parse_args():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='csv file path')
    parser.add_argument('--output-path', required=True, help='result path')
    return parser.parse_args()

def main():
    args = parse_args()
    sc = SparkContext.getOrCreate();

    # prep data: read csv and split comma delimited lines into tuples
    rdd = sc.textFile(args.input_path)
    temp = rdd.map(lambda x : x.split(','))
	
    # sort by country first if tie, sort by timestamp
    sorted_rdd = temp.sortBy(lambda x : x[2], lambda x : x[14])
    #cacat array into one string
    ans = sorted_rdd.map(lambda x : ','.join((str(y) for y in x)))
    
	# save result to specified location
    ans.repartition(1).saveAsTextFile(args.output_path + 'result.csv')


if __name__ == '__main__':
    main()

