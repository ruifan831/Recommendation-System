from pyspark import SparkContext
import json
import itertools
import time
import sys
from collections import defaultdict
import random
import re
import math


def main(testFile,modelFile,outFile):

    sc = SparkContext(appName="task1")
    lines=sc.textFile(modelFile)

    lines=lines.map(lambda x:json.loads(x)).cache()

    business_dict=dict(lines.filter(lambda x:x[0]=="business").map(lambda x:x[1]).collect())

    lines2=sc.textFile(testFile)

    pairs=lines2.map(lambda x:json.loads(x)).map(lambda x:(x["user_id"],x["business_id"]))

    def similarity_between_user_bus(x):
        try:
            user_profile=x[1][0]
            business_id=x[1][1]
            temp = map(lambda x: business_dict[x],user_profile)
            user_vector = set([item for sublist in temp for item in sublist])
            business_vector=set(business_dict[business_id])
            sim=cosineSimilarity(user_vector,business_vector)
            return ((x[0],business_id),sim)
        except:
            return ((x[0],business_id),0)

    def cosineSimilarity(user,business):
        sim = len(user.intersection(business))/(math.sqrt(len(user))*math.sqrt(len(business)))
        return sim

    result=lines.filter(lambda x:x[0]=="user").map(lambda x:x[1]).join(pairs).map(similarity_between_user_bus).filter(lambda x:x[1]>0.01)




    with open(outFile,"w") as f:
        def result_to_file(x):
            temp_dict={
                "user_id":x[0][0],
                "business_id":x[0][1],
                "sim":x[1]
            }
            return temp_dict
            
        results=result.map(result_to_file).collect()
        for i in results:
            json.dump(i,f)
            f.write("\n")

if __name__ == "__main__":
    test_file=sys.argv[1]
    model_file=sys.argv[2]
    output_file=sys.argv[3]
    start_time= time.time()
    main(test_file,model_file,output_file)
    print("--- %s seconds ---" % (time.time() - start_time))
    