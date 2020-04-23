from pyspark import RDD, SparkConf, SparkContext
import os
import numpy as np

# get the union of two sorted transactions
def unionT(t1: list, t2: list):
    t1 = sorted(t1)
    t2 = sorted(t2)
    uT = []
    j = 0
    for i in range(len(t1)):
        while t2[j] < t1[i]:
            if j == len(t2) - 1:
                return uT
            j += 1
        if t2[j] == t1[i]:
            uT.append(t1[i])

    return uT
        

# implementation of eclat using pyspark
def runEclat(prefix: list, supportList: list, min_support: int):
    support_list = []
    supportK = supportList.copy()
    while supportK:
        itemset, trans = supportK.pop(0)
        uT = unionT(trans, prefix[1])
        support = len(uT)

        if support >= min_support:
            print("now at: ", prefix[0] + itemset)
            support_list.append(prefix[0] + itemset) #for strings, use concat
            support_list += runEclat([prefix[0] + itemset, uT], supportK[1:],\
                 min_support)
        
    return support_list

    # should return a list



if __name__ == '__main__':
    conf = SparkConf().setAppName("EClaT")\
        .setMaster("local[*]")
    sc = SparkContext(conf = conf)
    inFile = "transData"

    minsup = 2
    triMatrixMode = True

    # Phase 1: generate Frequent items; produce vertical dataset

    # tranData is ans RDD, consider using zipwithUniqueId
    # TODO: add support for diffsets

    transDataFile = sc.textFile(inFile)
    transDataIndex = transDataFile.zipWithIndex()
    transData = transDataIndex.map(lambda v: (v[1], v[0].split()))
    print(transData.take(5))

    itemTids = transData.flatMap(lambda t: [(i, t[0]) for i in t[1]])\
        .groupByKey()\
        .map(lambda t: (t[0], list(t[1])))

    print(itemTids.take(5))

    freqItems = itemTids.filter(lambda t: len(t[1]) >= minsup)

    # TODO: phase 2 By dist-Eclat/bigFim, use dist-Apriori (data distribution) 
    # for k-itemsets generation

    freqItemsCnt = freqItems.map(lambda t: (t[0], len(t[1])))
    #freqItemsCnt.saveAsTextFile("FrequentItems")
    
    #sort the RDD
    freqItemTidsList = freqItems.sortByKey()
    print(freqItemTidsList.take(5))

    
    # use the configuration as the number of partitions
    print("number of partitions used: {}".format(sc.defaultParallelism))
    itemTidsParts = itemTids.repartition(sc.defaultParallelism).glom()
    print(itemTidsParts.take(5))


    #phase 3: EClaT from k-itemsets
    freqItemsList = freqItemTidsList.collect()
    freqAtoms = freqItemTidsList.keys().map(lambda t : t[0]).collect()
    freqRange = sc.parallelize(range(0, len(freqItemsList) - 1))
    freqItemsListToRun = freqRange.map(\
        lambda t: (freqItemsList[t], freqItemsList[t+1:]))

    print(freqItemsListToRun.take(5))

    res = freqItemsListToRun.flatMap(lambda t: runEclat(t[0], t[1], 2)).collect()
    res = freqAtoms + res
    print(res)
    
    
    


