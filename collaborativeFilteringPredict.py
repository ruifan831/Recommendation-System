from pyspark import SparkContext,SparkConf
import json
import itertools
import time
import sys
from collections import defaultdict
import random
import re
import math

def main(trainFile,testFile,modelFile,outFile,cfType):
    config=(SparkConf().setAppName("task3").set("spark.driver.memory","4g").set("spark.executor.memory","4g"))
    sc = SparkContext(conf=config)
    sc.setLogLevel("WARN")
    train_file=sc.textFile(trainFile)
    model_file= sc.textFile(modelFile)
    test_file = sc.textFile(testFile)
    if cfType == "item_based":

        pearson=dict(model_file.map(lambda x:json.loads(x)).map(lambda x:((x["b1"],x["b2"]),x["sim"])).collect())

        temp = train_file.map(lambda x:json.loads(x)).map(lambda x: (x["user_id"],(x["business_id"],x["stars"]))).groupByKey().join(test_file.map(lambda x:json.loads(x)).map(lambda x: (x["user_id"],x["business_id"])))

        def cal_rating(item):
            business_reviewed=dict(item[1][0])
            business_pre=item[1][1]
            # try:
            #     return (item[0],business_pre,business_reviewed[business_pre])
            # except:
            pearsons=list(map(lambda x: [x,business_pre],business_reviewed.keys()))

            first_5_sim=sorted(map(lambda x:(x[0],pearson.get(tuple(sorted(x)),0)),pearsons),key=lambda x:x[1],reverse=True)[:5]
            # valid_business=list(filter(lambda x: x[1]>0,first_5_sim))
            # if len(valid_business)>=5:
            denominator = sum(map(lambda x: x[1],first_5_sim))
            nominator = sum(map(lambda x: business_reviewed[x[0]]*x[1] ,first_5_sim))
            if denominator > 0:
                result=nominator/denominator
            else:
                result = None
            return (item[0],business_pre,result)
            # else:
            #     return (item[0],business_pre,None)

        final_result = temp.map(cal_rating).filter(lambda x:x[2] is not None).collect()
    if cfType == "user_based":
        def avg(x):
            return (x[0],sum(x[1])/len(x[1]))

        def cal_rating_user(cousers):
            corate_users= dict(cousers[1][0])
            current_user = cousers[1][1]
            keys= list(map(lambda x : [current_user,x] , list(corate_users.keys())))
            pearsons = dict(list(map(lambda x: (x[1],pearson.get(tuple(sorted(x)),0)),keys)))
            nominator = sum(map(lambda x: (corate_users[x] - user_avg[x])*pearsons[x],list(corate_users.keys())))
            denominator = sum(map(lambda x: pearsons[x],list(corate_users.keys())))
            if denominator == 0:
                return (current_user,cousers[0],user_avg[current_user])
            else:
                return (current_user,cousers[0],user_avg[current_user]+nominator/denominator)
        
        pearson=dict(model_file.map(lambda x:json.loads(x)).map(lambda x:((x["u1"],x["u2"]),x["sim"])).collect())

        temp = train_file.map(lambda x:json.loads(x)).map(lambda x: (x["business_id"],(x["user_id"],x["stars"]))).groupByKey().join(test_file.map(lambda x:json.loads(x)).map(lambda x: (x["business_id"],x["user_id"])))

        user_avg=dict(train_file.map(lambda x:json.loads(x)).map(lambda x: (x["user_id"],x["stars"])).groupByKey().map(avg).collect())
        final_result = temp.map(cal_rating_user).collect()
    


    with open(outFile,"w") as f:
        for i in final_result:
            temp_dict={
                "user_id":i[0],
                "business_id" : i[1],
                "stars" : i[2]
            }
            json.dump(temp_dict,f)
            f.write("\n")

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    model = sys.argv[3]
    out = sys.argv[4]
    cf_type = sys.argv[5]
    start_time = time.time()
    main(train,test,model,out,cf_type)
    print("--- %s seconds ---" % (time.time() - start_time))
