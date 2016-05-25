import datetime
import pandas as pd
import math
import numpy as np
from numpy import matrix
from collections import defaultdict
from scipy.stats.stats import pearsonr
from heapq import nlargest
from operator import itemgetter

#**************************************************************************************
# Global configuration
#**************************************************************************************
train_file = "/home/ml1/Downloads/Kaggle-Expedia/CollabrativeFiltering/train.csv"
test_file  = "/home/ml1/Downloads/Kaggle-Expedia/CollabrativeFiltering/test.csv"
output_path = "/home/ml1/Downloads/Kaggle-Expedia/CollabrativeFiltering/"

#**************************************************************************************
# Global variables
#**************************************************************************************
#train_data = pd.DataFrame(columns=('user_id','user_location','srch_destination','is_booking','hotel_cluster'))

start_time = datetime.datetime.now()

def get_training_file_name():
    return output_path+'compressed_train_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_testing_file_name():
    return output_path+'test_ratings_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def get_rdd_output():
    return output_path+'rdd_output_'+str(start_time.strftime("%Y-%m-%d-%H-%M")) + '.csv'

def Expedia_Competition():
    
    #local data set
    #train_data = pd.DataFrame(columns=('user_id','user_location_city','srch_destination_id','is_booking','hotel_cluster'))
    #train_data = pd.DataFrame()
    
    if 1 == 1:
        print('Compress train file...')
        sys.stdout.flush()
        
        f = open(train_file, 'r')
        #read the title and ignore
        f.readline()
        
        compressed_train_file = get_training_file_name()
        f_compressed = open(compressed_train_file,'w')
        #
        #read other lines and convert to a dataframe-traindata
        #
        total = 0
        while 1:
            line = f.readline().strip()
            total+=1
            #we just read 100 lines temporarily
            #if total % 100 == 0:
            #    break
            
            if total % 10000000 == 0:
                print('\nRead {} lines'.format(total))
                sys.stdout.flush()
                
            if line == '':
                break
            if total % 100000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            #get elements from each line
            arr = line.split(",")
            book_year = int(arr[0][:4])
            user_location = arr[5]
            orig_destination_distance = arr[6]
            srch_destination = arr[16]
            is_booking = (arr[18])
            hotel_country = arr[21]
            hotel_market = arr[22]
            hotel_cluster = arr[23]
            user_id=arr[7] 
            srch_adults_cnt=int(arr[13])
            srch_children_cnt=int(arr[14])
            srch_rm_cnt=int(arr[15])
            
            #append_1 = 3 + 17*is_booking
            #append_2 = 1 + 5*is_booking
    
            #each_line = pd.DataFrame({ 'user_id':user_id,'user_location_city':user_id,\
            #                 'srch_destination_id':srch_destination,'is_booking':is_booking,\
            #                 'hotel_cluster':hotel_cluster },index=[total-1])
            #train_data=train_data.append(each_line)
            #train_data[total-1][:]={user_id,user_location,srch_destination,is_booking,hotel_cluster}
            
            if total == 1: 
                f_compressed.write( "user_id,user_location_city,srch_destination_id,is_booking,hotel_cluster\n" )
                
            f_compressed.write(user_id+","+user_id+","+srch_destination+","+is_booking+","+hotel_cluster+"\n")
            
        f.close()
        f_compressed.close()
    
    #print(train_data)
    #return
    #
    # Transform train_data
    #
    print("\nread compressed train data")
    
    compressed_train_file = get_training_file_name()
    #compressed_train_file = "/home/ml1/Downloads/Kaggle-Expedia/CollabrativeFiltering/compressed_train_ratings_2016-05-25-00-05.csv"
    train_data=pd.read_csv(compressed_train_file, low_memory=False)
    
    print("transform train data")
    UserCity=train_data[['user_location_city','user_id']]
    """Converting the data fram UserCity to a dictionary that user_city_location is the ids
    and the user_ids are the values"""
    location_user_dict=UserCity.groupby('user_location_city')['user_id'].apply(set).to_dict()
    """UserCityDict---> the keys are user_id and the values are user_location_city"""
    user_city_dict=UserCity.set_index('user_id')['user_location_city'].to_dict()
    """Computing the ordered set of users"""
    UserList=train_data['user_id'].tolist()
    UserSet=set(UserList)
    SortedUserSet=sorted(UserSet)
    """Computing the ordered set of items"""
    LocationList=train_data['srch_destination_id'].tolist()
    LocationSet=set(LocationList)
    SortedLocationList=sorted(LocationSet)
    """Extracting user_id, destination and rating to build th matrix"""
    user_location_hotelCluster_matrix=train_data[['user_id','srch_destination_id','is_booking','hotel_cluster']]
    user_location_rating=user_location_hotelCluster_matrix.as_matrix()
    User_Item_Rating={}
    Item_User_Rating={}
    
    #delete dataframes that won't be used any more to reduce the memory needs
    print("delete intermediate data")
    del train_data
    del UserCity
    del location_user_dict
    #del user_city_dict  # dictionary for (user:city)
    del UserList
    del UserSet         
    #del SortedUserSet  # sorted set for (user)
    del LocationList
    del LocationSet
    #del SortedLocationList  # sorted set for (city)
    del user_location_hotelCluster_matrix
    #del user_location_rating # rating matrix n*m (user * item)
    
    #Input: user_city_dict  SortedUserSet SortedLocationList

    
    
    print('Start User-based Collaborative Filterting')
    sys.stdout.flush()
    
    for i in range(len(user_location_rating)):
      if user_location_rating[i,2]==1:
           User_Item_Rating.setdefault(user_location_rating[i,0],{})
           User_Item_Rating[user_location_rating[i,0]][user_location_rating[i,1]]=user_location_rating[i,3]
           Item_User_Rating.setdefault(user_location_rating[i,1],{})
           Item_User_Rating[user_location_rating[i,1]][user_location_rating[i,0]]=user_location_rating[i,3]    
    
    UserDictionary=dict(enumerate(SortedUserSet))
    UserDictionaryReverse=dict(map(reversed, UserDictionary.items()))
    ItemDictionary=dict(enumerate(SortedLocationList))
    ItemDictionaryReverse=dict(map(reversed, ItemDictionary.items()))
    
    del UserDictionary
    del ItemDictionary
    
    print("Allocate memory for the matrix")
    #    myarray = np.zeros((np.max(SortedUserSet), np.max(SortedLocationList)))
    num_user = len(SortedUserSet)
    num_item = len(SortedLocationList)
    #print("Allocate space for ",num_user," x ",num_item," matrix which might need ", num_user*num_item*4/1024/1024/1024,"GB")
    #myarray = np.zeros((len(SortedUserSet), len(SortedLocationList)))
    
    #print('Start Making the matrix')
    #for key1, row in User_Item_Rating.iteritems():
    #  for key2, value in row.iteritems():
    #       myarray[UserDictionaryReverse.get(key1), ItemDictionaryReverse.get(key2)] = value    
    
    print('Start User-based rating computation')     
    sys.stdout.flush()
          
    """Iterating over users"""
    countering=0
    for user in SortedUserSet:
      cityID=user_city_dict[user]
      countering+=1
      if countering % 1000==0:
          print('Read {} users...'.format(countering))
          sys.stdout.flush()
    
      for item in SortedLocationList:
    #              Same_Location_Users=location_user_dict[cityID]
           total1=0
           total2=0
           
           
           if User_Item_Rating.get(user):
                if User_Item_Rating[user].get(item):
                     continue
           for EachUser in SortedUserSet:
    #                   dict1=User_Item_Rating[user]
    #                    dict2=User_Item_Rating[EachUser]
                '''Pearson computation of two dictionary values with different keys'''
                FirstUser=UserDictionaryReverse[user]      # index
                SecondUser=UserDictionaryReverse[EachUser] 
                
                myarray = np.zeros((2, len(SortedLocationList)))
                for key1, row in User_Item_Rating.iteritems():
                   if key1 == FirstUser :
                       for key2, value in row.iteritems():
                          myarray[0, ItemDictionaryReverse.get(key2)] = value
                   if key1 == SecondUser:
                       for key2, value in row.iteritems():
                          myarray[1, ItemDictionaryReverse.get(key2)] = value
                         
                v1=myarray[0,:]
                v2=myarray[1,:]
                            
                weight=np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
                
    #                   keys = list(dict1.viewkeys() | dict2.viewkeys())
    #                    weight=np.corrcoef([dict1.get(x, 0) for x in keys],[dict2.get(x, 0) for x in keys])[0, 1]
    
    
                if(User_Item_Rating.get(EachUser) is None):
                     continue
                elif(User_Item_Rating.get(EachUser).get(item) is None):
                     continue
                elif(math.isnan(weight)):
                     continue
                else:
                     total1=total1+(weight*int(User_Item_Rating.get(EachUser).get(item)))
                     total2=total2+weight
           if total2>0:
                average=float(total1)/total2
                User_Item_Rating.setdefault(user,{})
                User_Item_Rating[user][item]=math.floor(average)
                Item_User_Rating.setdefault(item,{})
                Item_User_Rating[item][user]=math.floor(average)
           else:
                average=float(0)
                User_Item_Rating.setdefault(user,{})
                User_Item_Rating[user][item]=math.floor(average)
                Item_User_Rating.setdefault(item,{})
                Item_User_Rating[item][user]=math.floor(average)
                                   
    
    del User_Item_Rating
    
    #
    # perdict based on test data
    #
    test_data=pd.read_csv(test_file)
    ItemIDs=test_data['srch_destination_id'].tolist()
    TestItemSet=set(ItemIDs)
    SortedTestItemSet=sorted(TestItemSet)
    UserIDs=test_data['user_id'].tolist()
    TestUserSet=set(UserIDs)
    SortedTestUserSet=sorted(TestUserSet)
    now=datetime.datetime.now()
    path=output_path + 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    out=open(path,"w")
    out.write("id, hotel_cluster\n")
    
    for temp in ItemIDs:
      if Item_User_Rating.get(item):
           Temporary_Hotel_Clusters=Item_User_Rating.get(item).values()
      Top5=nlargest(5,enumerate(Temporary_Hotel_Clusters),itemgetter(1))
      Top_5=dict(Top5).values()
      out.write(str(temp) + ',')
      if len(Top_5)==0:
           continue
      
      for i,count in enumerate(Top_5):
           if not math.isnan(count):
                out.write(' '+ str(int(count)))
      out.write("\n")
    
    out.close()
    
    print('The program is completed!') 
    sys.stdout.flush()
Expedia_Competition()
