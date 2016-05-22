from __future__ import print_function

import sys
import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

from pyspark import SparkContext
import pandas as pd

# $example on$
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# $example off$

train_file = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/train.csv"
test_file  = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/test.csv"
output_path = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/"

#define maps
user_item_rating = defaultdict(lambda:defaultdict(int))
best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))

start_time = datetime.datetime.now()

def get_training_file_name():
    return output_path+'train_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_testing_file_name():
    return output_path+'test_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_rdd_output():
    return output_path+'rdd_output_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def transform_test_file():
    print('Transforming test file...')
    f = open(test_file,'r')
    #read the title and ignore
    f.readline()
  
    test_file_name = get_testing_file_name()
    out  = open(test_file_name,"w")    
    
    total = 0
    while 1:
        line = f.readline().strip()
        total+=1
        #we just read 100 lines temporarily
        #if total % 100 == 0:
        #    break        
        
        if total % 1000000 == 0:
            print('Read {} lines...'.format(total))
            
        if line == '':
            break
        
        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        user_id = arr[8]
        srch_adults_cnt=int(arr[14])
        srch_children_cnt=int(arr[15])
        srch_rm_cnt=int(arr[16])
    
        # write to test_file

        out.write(user_id+","+srch_destination_id+","+"1.0"+"\n" )
        
    f.close()
    out.close()
            
def transform_train_file():
        
    print('Transforming train file...')
    f = open(train_file, 'r')
    #read the title and ignore
    f.readline()
    
    #read other lines and convert to a dictionary
    total = 0
    while 1:
        line = f.readline().strip()
        total+=1
        #we just read 100 lines temporarily
        #if total % 100 == 0:
        #    break
        
        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))
            
        if line == '':
            break
        
        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]
        user_id=arr[7] 
        srch_adults_cnt=int(arr[13])
        srch_children_cnt=int(arr[14])
        srch_rm_cnt=int(arr[15])
        
        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking
        
        if user_id != '' and srch_destination_id != '':
            user_item_rating[(user_id, srch_destination_id)][hotel_cluster] += append_1
        
        #if user_location_city != '' and orig_destination_distance != '' :
        #    best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1
    
    f.close()
    #convert dictionary to RDD
    
    #path = output_path+'train_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    train_file_name = get_training_file_name()
    out  = open(train_file_name,"w")
    #out.write("user_id, location, ratings\n")   #NO title
    
    for user_id, srch_destination_id in user_item_rating:
        
        cluster_score = user_item_rating[(user_id,srch_destination_id)]
        #get the hotel cluster that user booked or clicked mostly
        cluster_score_sorted = sorted( cluster_score.items(), key=lambda x:x[1], reverse=True )
        #print( user_id ,'\t', srch_destination_id, '\t',cluster_score_sorted  )
        for cluster_id, score in cluster_score_sorted:
            #print( user_id ,'\t', srch_destination_id, '\t', cluster_id  )
            #rating_train.append( [user_id,srch_destination_id,cluster_id] )
            out.write(user_id+","+srch_destination_id+","+ cluster_id+"\n" )
            
            break
    out.close()
    #print(rating_train)

def toCSVLine(data):
    return ','.join(str(d) for d in data)

def execute_recommendation():

    sc = SparkContext(appName="PythonCollaborativeFilteringExample")
    #sc = SparkContext( 'local', 'pyspark')
    
    #Load train data and train
    train_file_name = get_training_file_name()
    train_data = sc.textFile(train_file_name)
    ratings = train_data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    
    rank = 10
    number_iteration = 10
    model = ALS.train(ratings, rank, number_iteration)
    
    #load test data and do prediction 
    test_file_name = get_testing_file_name()
    test_data = sc.textFile(test_file_name)
    test_ranking=test_data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    
    
    testdata = test_ranking.map(lambda p: (p[0], p[1]))
    count_rdd=testdata.count()
    print("testdata count_rdd=",count_rdd)
    if count_rdd > 0:
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        
        #predictions_lines = predictions.map(toCSVLine)
        result_file = get_rdd_output()
        predictions.saveAsTextFile(result_file)
    
        count_rdd = predictions.count()
        print("after prediction: count_rdd=",count_rdd)
    else:
        print("Error: empty testdata")

    sc.stop()
    
transform_train_file()
transform_test_file()
execute_recommendation()
