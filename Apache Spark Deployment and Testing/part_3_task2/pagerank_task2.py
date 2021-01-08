import argparse
from pyspark.context import SparkContext
from pyspark import SparkConf
from operator import add

number_partitions = 150

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

def filter_colons(link):
    if ':' in link and not link.startswith('category:'):
        return False
    return True

def hash_djb2(key):                                                                                                                                
    hash_accum = 5381
    for c in key:
        hash_accum = (( hash_accum << 5) + hash_accum) + ord(c)
    return hash_accum & 0xFFFFFFFF

def main():
    args = parse_args()

    conf = SparkConf().setAppName('baseline PR for task 3')   
    sc = SparkContext.getOrCreate(conf)   

    #rdd = sc.wholeTextFiles(args.input_path, number_partitions)
    rdd = sc.textFile(args.input_path)
    #parse the input files, wholeTextFiles pulls in the data as one key value pair per file, with organized as (fileName, content)
    #so we split the lines and then split the tabs. Only split first tab for when tabs in the link nice
    #sanity check to ensure key-value pairs, then filter out lines with colons if they don't start with Category:
    #rdd_split = rdd.values().flatMap(lambda x : x.splitlines()).map(lambda x : x.lower().split('\t', 1))
    rdd_split = rdd.map(lambda x : x.lower().split('\t', 1))
    valid_list = rdd_split.filter(lambda y : len(y) == 2).filter(lambda x: filter_colons(x[0]) and filter_colons(x[1])).partitionBy(number_partitions)

    # consolidate outgoing nodes for each node into list
    # filters nodes which have no outbound links, README says to exclude these. Since they have no outbound, they will never contribute to other links
    # example node 426655 from berkely dataset
    #optimizations: increase number of partitions to ensure each task is sufficiently small. Use a known word based hash function djb2 to distribute words evenly
    graph = valid_list.groupByKey(number_partitions, hash_djb2)

    #need to find the nodes which are outbound only nodes to add them back in later
    #we don't partition this because it will be appeneded to the output of update_rank before reduceByKey. reduceByKey needs to reshuffle any.
    #out_only_nodes = graph.keys().subtract(valid_list.values()).map(lambda x : (x,0))

    # set initial page value to 1
    ranks = graph.mapValues(lambda x : float(1))

    # iterate 10 times
    for i in range(10):
        # put the ranks to their corresponding node
        graph_ranks = graph.join(ranks)
        
        # calculate contribution of each destination node by creating a row for each inbound link to the destination with the corresponding contribution
        #then group by the destination node and add them up

        # filters out only nodes - we need these in the output because they will affect other ranks
        # example node 2609 from berkely dataset
        updated_ranks = graph_ranks.flatMap(lambda x : update_rank(x[1][0], x[1][1])).reduceByKey(add, numPartitions = number_partitions)
        
        # update ranks for each node
        ranks = updated_ranks.mapValues(lambda x : x * .85 + .15)

    # write the result to output location
    ans = ranks.map(lambda x : x[0] + '\t' + str(x[1]))
    ans.saveAsTextFile(args.output_path)


if __name__ == '__main__':
    main()