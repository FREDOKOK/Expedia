#################################################################################
# Main idea: collabrative filtering: 
#          user = user    item = hotel+hotel_cluster
#################################################################################

from __future__ import print_function

import sys as Sys
import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import os.path

from pyspark import SparkContext
import pandas as pd

# $example on$
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# $example off$

###########################################################################
# Global variable
###########################################################################

quick_run = 2
total_train_rows=37670294
total_test_rows=2528244
train_file = "/home/ml1/Downloads/Kaggle-Expedia/MLLIB_ALS/train.csv"
test_file  = "/home/ml1/Downloads/Kaggle-Expedia/MLLIB_ALS/test.csv"
output_path = "/home/ml1/Downloads/Kaggle-Expedia/MLLIB_ALS/"


#define maps
train_dest_and_hotel_cluster = defaultdict(lambda:defaultdict(int))



#best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))

start_time = datetime.datetime.now()

def get_training_file_name():
    if quick_run >= 1:
        return output_path+'train_ratings_quick.csv'
    else:
        return output_path+'train_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_testing_file_name():
    if quick_run >= 1:
        return output_path+'test_ratings_quick.csv'
    else:
        return output_path+'test_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_rdd_output():
    return output_path+'rdd_output_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

# cannot be used in iPython notebook
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('\r%s [%s] %s%s %s' % (prefix, bar, percents, '%', suffix))
    Sys.stdout.flush()
    if iteration == total:
        print("\n")
        
def printSimpleProgress(iteration, total):
    sys.stdout.write("\r     Status - %d%%" % int(iteration)/int(total))
    sys.stdout.flush()

def get_key(key):
    try:
        return int(key)
    except ValueError:
        return key
    
def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024 * 10 # 10M buffer
    buf = f.read(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = f.read(buf_size)
    f.close()
    return lines
    
def transform_train_file():
    #declare local variable
    train_user_item_rating = defaultdict(lambda:defaultdict(int))
    
    
    train_file_name = get_training_file_name()    
    if quick_run >= 1:
        if os.path.exists(train_file_name):
            return
    
    print('Evaluate the workload')
    
    if quick_run>=2:
        line_number = total_train_rows
    else:
        line_number = rawcount(train_file)
    
    print('Transforming train file(',line_number,' lines)')
    
    f = open(train_file, 'r')
    #read the title and ignore
    f.readline()
    
    line_number = line_number - 1
    #read other lines and convert to a dictionary
    total = 0
    while 1:
        line = f.readline().strip()
        total+=1
        #we just read 10000 lines temporarily
        #if total % 100000 == 0:
        #   break
        
        #if total % 10000000 == 0:
        #    print('\nRead {} lines...'.format(total))
            
        if line == '':
            break
        
        if total % 100 == 0:
            #printSimpleProgress(total,line_number)
            printProgress(total,line_number,prefix='Progress:',suffix='Complete',barLength=50)
        
        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = int(arr[16])
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = int(arr[23])
        user_id=arr[7] 
        srch_adults_cnt=int(arr[13])
        srch_children_cnt=int(arr[14])
        srch_rm_cnt=int(arr[15])
        
        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking
        
        if user_id != '' and srch_destination_id != '':
            train_user_item_rating[(user_id, srch_destination_id)][hotel_cluster] += append_1
            train_dest_and_hotel_cluster[srch_destination_id][hotel_cluster] += append_1
        
    
    f.close()
    Sys.stdout.write('\n')
    #convert dictionary to RDD
    
    #path = output_path+'train_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    train_file_name = get_training_file_name()
    out  = open(train_file_name,"w")
    #out.write("user_id, location, ratings\n")   #NO title
    
    key_list = train_user_item_rating.keys()
    #key_list.sort(key=lambda i:i[0])
    
    sorted_key_list = sorted(key_list, key=lambda t:get_key(t[0]))
    
    del key_list
    
    #print( sorted_key_list )
    print("Generating new train file")
    
    line_number = len(sorted_key_list)
    i = 0
    for (user_id, srch_destination_id) in sorted_key_list:
        
        i=i+1
        printProgress(i,line_number,prefix='Progress:',suffix='Complete',barLength=50)
        
        cluster_score = train_user_item_rating[(user_id,srch_destination_id)]
        # sort by cluster id
        cluster_score_sorted = sorted( cluster_score.iteritems(), key=lambda x:x[0], reverse=True )
        for cluster_id, score in cluster_score_sorted:
            out.write(user_id+","+str(srch_destination_id*100+cluster_id)+","+ str(score)+"\n" )     
    out.close()
    #print(rating_train)
    Sys.stdout.write('\n')
    
    return train_dest_and_hotel_cluster

def transform_test_file(dest_and_hotel_cluster):
    #declare local variable
    test_user_dest = defaultdict(lambda:defaultdict(int))
    
    test_file_name = get_testing_file_name()    
    if quick_run >= 1:
        if os.path.exists(test_file_name):
            return
    
    if quick_run>=2:
        line_number = total_test_rows
    else:
        line_number = rawcount(test_file)
    
    print('Transforming test file(',line_number,'lines)')
    
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
        #if total % 1000 == 0:
        #    break        
        
        #if total % 1000000 == 0:
        #    print('\nRead {} lines...'.format(total))
            
        if line == '':
            break
        
        if total % 100 == 0:
            #printSimpleProgress(total,line_number)
            printProgress(total,line_number,prefix='Progress:',suffix='Complete',barLength=50)
        
        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = int(arr[17])
        hotel_country = arr[20]
        hotel_market = arr[21]
        user_id = arr[8]
        srch_adults_cnt=int(arr[14])
        srch_children_cnt=int(arr[15])
        srch_rm_cnt=int(arr[16])
        
        if user_id != '' and srch_destination_id != '':
            test_user_dest[user_id][srch_destination_id] += 1
   
    f.close()
    Sys.stdout.write('\n')
    
    print('Generating test file')
    line_number = len(test_user_dest)
    total=1
    for user_id in test_user_dest:
        for dest_id in test_user_dest[user_id]:
        # write to test_file
            for cluster_id in dest_and_hotel_cluster[dest_id]:
                out.write(user_id+","+str(dest_id*100+cluster_id)+","+"0.0"+"\n" )   
        printProgress(total,line_number,prefix='Progress:',suffix='Complete',barLength=50)
        total=total+1
    
    out.close()
    


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
    model = ALS.trainImplicit(ratings, rank, number_iteration)
    
    #load test data and do prediction 
    test_file_name = get_testing_file_name()
    test_data = sc.textFile(test_file_name)
    test_ranking=test_data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    
    
    testdata = test_ranking.map(lambda p: (p[0], p[1]))
    count_rdd=testdata.count()
    
    
    
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

print('Progam starts - contact Fred Gu')
train_dest_and_hotel_cluster = transform_train_file()
transform_test_file(train_dest_and_hotel_cluster)
#execute_recommendation()
