from pyspark import SparkContext,SparkConf
import json
import itertools
import time
import sys
from collections import defaultdict
import random
import re
import math

def main(train_file,model_file,cf_type):

    config=SparkConf().setAppName("task3").set("spark.driver.memory","4g").set("spark.executor.memory","4g")
    sc = SparkContext(conf=config)
    sc.setLogLevel("WARN")
    lines=sc.textFile(train_file)
    def average(x):
        return sum(x)/len(x)
    if cf_type=="item_based":
        

        business_profile=dict(lines.map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],(x["user_id"],x["stars"]))).groupByKey().map(lambda x:(x[0],dict(x[1]))).collect())

        def pearsonCorrelation_item(x):
            business_1=x[0]
            business_2=x[1]
            business_1_profile=business_profile[business_1]
            business_2_profile=business_profile[business_2]

            cousers=set(business_1_profile.keys()).intersection(set(business_2_profile.keys()))
            if len(cousers)>=3:
                business_1_avg_rating=average(list(map(lambda x:business_1_profile[x],cousers)))
                business_2_avg_rating=average(list(map(lambda x:business_2_profile[x],cousers)))
                numerator = sum(map(lambda x: (business_1_profile[x]-business_1_avg_rating)*(business_2_profile[x]-business_2_avg_rating),cousers))
                business_1_len=sum(map(lambda x: (business_1_profile[x]-business_1_avg_rating)**2,cousers))
                business_2_len=sum(map(lambda x: (business_2_profile[x]-business_2_avg_rating)**2,cousers))
                try:

                    denominator=math.sqrt(business_1_len)*math.sqrt(business_2_len)

                    pearson_correlation= numerator/denominator
                    return (tuple(sorted(x)),pearson_correlation)
                except:
                    return (tuple(sorted(x)),-2)
            else:
                return (tuple(sorted(x)),-2)


        pairs = lines.map(lambda x:json.loads(x)).map(lambda x:(1,x["business_id"])).groupByKey().flatMap(lambda x: itertools.combinations(set(x[1]),2)).repartition(40).map(pearsonCorrelation_item).filter(lambda x: x[1] >0).collect()
        
        with open(model_file,"w") as f:
            for i in pairs:
                temp_dict={
                    "b1":i[0][0],
                    "b2":i[0][1],
                    "sim": i[1]
                    }
                json.dump(temp_dict,f)
                f.write("\n")
    if cf_type == "user_based":
        num_hash=30
        bands=30
        rows=1
        
        def generate_random_num(n):
            random_list=[]
            range_max=n
            while n>0:
                temp=random.randint(1,200)
                while temp in random_list:
                    temp=random.randint(1,200)
                random_list.append(temp)
                n-=1
            return random_list
        def minhash_list(x):
            profiles=x[1]
            final_list=[]
            buckets_belong=[]
            for i in range(0,num_hash):
                id_list=[]
                for profile in profiles:
                    new_id=business_to_index[profile][i]
                    id_list.append(new_id)
                final_list.append(min(id_list))
            for i,band in enumerate(chunks(final_list,rows)):
                buckets_belong.append((i,band[0]))
                # bucket=(hash(tuple(band))*band_a_list[i]+band_b_list[i])%buckets
                # buckets_belong.append(bucket+i*buckets)
            return [(j,x[0]) for j in buckets_belong]

        def chunks(lst, n):

            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        def check_vaild_pair(x):
            set_1 = set(user_dict[x[0]].keys())
            set_2 = set(user_dict[x[1]].keys())
            nominator = float(len(set_1.intersection(set_2)))
            if nominator >=3:
                if nominator/float(len(set_1.union(set_2))) >=0.01:
                    return True
                else:
                    return False
            else:
                return False
        def hash_fun(x):
            hash_x=business_id[x]
            minhash=[]
            for i in range(num_hash):
                minhash.append((a_list[i]*hash_x+b_list[i])%10253)
            return (x,minhash)

        def pearsonCorrelation_user(pair):
            user_1=pair[0]
            user_2=pair[1]
            user_1_profile=user_dict[user_1]
            user_2_profile=user_dict[user_2]

            corated=set(user_1_profile.keys()).intersection(set(user_2_profile.keys()))
            if len(corated)>=3:
                user_1_avg_rating=average(list(map(lambda x:user_1_profile[x],corated)))
                user_2_avg_rating=average(list(map(lambda x:user_2_profile[x],corated)))
                numerator = sum(map(lambda x: (user_1_profile[x]-user_1_avg_rating)*(user_2_profile[x]-user_2_avg_rating),corated))
                user_1_len=sum(map(lambda x: (user_1_profile[x]-user_1_avg_rating)**2,corated))
                user_2_len=sum(map(lambda x: (user_2_profile[x]-user_2_avg_rating)**2,corated))
                try:

                    denominator=math.sqrt(user_1_len)*math.sqrt(user_2_len)

                    pearson_correlation= numerator/denominator
                    return (tuple(sorted(pair)),pearson_correlation)
                except:
                    return (tuple(sorted(pair)),-2)
            else:
                return (tuple(sorted(pair)),-2)

        random.seed(20)
        a_list=generate_random_num(num_hash)
        random.seed(70)
        b_list=generate_random_num(num_hash)

        business_temp=lines.map(lambda x:json.loads(x)).map(lambda x:x["business_id"]).distinct().cache()
        # business_id=business_temp.map(lambda x: (1,x)).groupByKey().map(lambda x: dict(zip(x[1],list(range(len(x[1])))))).collect()[0]
        business_id=dict(zip(business_temp.collect(),list(range(len(business_temp.collect())))))
       
        business=business_temp.map(hash_fun).collect()

        business_to_index=dict(business)

        temp=lines.map(lambda x: json.loads(x)).map(lambda x: (x["user_id"],(x["business_id"],x["stars"]))).groupByKey().map(lambda x: (x[0],dict(x[1])) ).cache()
        user_dict=dict(temp.collect())

        pairs = temp.map(lambda x: (x[0],list(x[1].keys()))).flatMap(minhash_list).groupByKey().flatMap(lambda x:itertools.combinations(sorted(x[1]),2)).distinct()

        result= pairs.filter(check_vaild_pair).map(pearsonCorrelation_user).filter(lambda x:x[1]>0).collect()



        with open(model_file,"w") as f:
            for i in result:
                temp_dict= {
                    "u1":i[0][0],
                    "u2":i[0][1],
                    "sim":i[1]
                }
                json.dump(temp_dict,f)
                f.write("\n")


if __name__ == "__main__":
    trainFile= sys.argv[1]
    modelFile= sys.argv[2]
    cfType = sys.argv[3]
    start_time = time.time()
    main(trainFile,modelFile,cfType)
    print("--- %s seconds ---" % (time.time() - start_time))



    