from pyspark import SparkContext
import json
import itertools
import time
import sys
from collections import defaultdict
import random
import re
import math

def main(trainFile,modelFile,stopFile):
    with open(stopFile,"r") as f:
        stopwords=list(map(lambda x:x.strip(),f.readlines()))
        stopwords.extend(["(","[", ",", ".", "!", "?", ":", ";", "]", ")",""])

    def TF_IDF(x):
        termFreq=defaultdict(int)
        for i in x[1]:
            termFreq[i]+=1
        maxFre=max(termFreq.values())
        for k,v in termFreq.items():
            termFreq[k]=(v/maxFre)*IDF[k]
        result=dict(sorted(termFreq.items(),key=lambda x:x[1],reverse=True)[:200])
        return (x[0],list(result.keys()))

    sc = SparkContext(appName="task1")
    lines = sc.textFile(trainFile)

    sc.setLogLevel("ERROR")

    lines_json=lines.map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],x["text"])).flatMap(lambda x: [(x[0],i) for i in re.split("[\(\[,\.\!\?:;\]\)\s\n]+",x[1])]).filter(lambda x:x[1] not in stopwords).cache()
    total_documents=lines_json.groupByKey().count()

    IDF = lines_json.map(lambda x:(x[1],x[0])).distinct().groupByKey().map(lambda x:(x[0],math.log2(total_documents/len(x[1])))).collect()
    IDF=dict(IDF)

    business_profile=lines_json.groupByKey().repartition(40).map(TF_IDF).map(lambda x:("business",x))
    user_profile=lines.map(lambda x: json.loads(x)).map(lambda x:(x["user_id"],x["business_id"])).groupByKey().map(lambda x:(x[0],list(set(x[1])))).map(lambda x:("user",x))


    model=user_profile.union(business_profile).collect()
    with open(modelFile,"w") as f:
        for i in model:
            json.dump(i,f)
            f.write("\n")

if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    stopwords_file = sys.argv[3]
    start_time= time.time()
    main(train_file,model_file,stopwords_file)
    print("--- %s seconds ---" % (time.time() - start_time))
    