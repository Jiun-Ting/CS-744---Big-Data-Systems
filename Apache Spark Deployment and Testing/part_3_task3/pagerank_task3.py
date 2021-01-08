import argparse
from pyspark.context import SparkContext

def parse_args():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='csv file path')
    parser.add_argument('--output-path', required=True, help='result path')
    return parser.parse_args()

def update_rank(neighbors, rank):
    """ calculate contribution for each neighbor for the current node """
    res = []
    for node in neighbors:
      res.append((node, rank / len(neighbors)))
    return res

def main():
    args = parse_args()
    sc = SparkContext.getOrCreate();
	
	#load and clean data. Split data by tab max one occurence because some outbound links had \t characters in them.
	# as a sanity check ensure each tuple has two elements
    rdd = sc.textFile(args.input_path)
    lower_list = rdd.map(lambda x : x.lower())
    valid_list = lower_list.filter(lambda x: (':' not in x or x.startswith('category:')))
    final_list = valid_list.map(lambda x : x.split('\t', 1)).filter(lambda y : len(y) == 2)

    # consolidate outgoing nodes for each node into list of the form (outbound node, (target node1, target node2)
    graph = final_list.groupByKey()

    # cache the graph
    graph.cache().count()

    # set initial page value to 1
    ranks = graph.mapValues(lambda x : float(1))

	#in this phase we repeatedly join graph and rank to pull in the new rank values for each outbound node, 
	#then caluclate the new contributions in the flatMap and reduce by the target node
	#needs to be a flatMap because we change the keys and because there will be multiple tuples for a given target node
    for i in range(10):
      graph_ranks = graph.join(ranks)
      updated_ranks = graph_ranks.flatMap(lambda x : update_rank(x[1][0], x[1][1])).reduceByKey(lambda y, z : y + z)
	  
	  # update ranks for each node
      ranks = updated_ranks.mapValues(lambda x : x * .85 + .15)

    # write the result to output location
    # ans = ranks.map(lambda x : ' '.join((str(y) for y in x)))
    ranks.saveAsTextFile(args.output_path)

if __name__ == '__main__':
    main()
