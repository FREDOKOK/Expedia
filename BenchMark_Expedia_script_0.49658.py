# code source:  https://www.kaggle.com/zfturbo/expedia-hotel-recommendations/leakage-solution/code

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

train_file = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/train.csv"
test_file  = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/test.csv"
output_path = "/home/ml1/Downloads/Kaggle-Expedia/AnalyzedData/"

def run_solution():
    print('Preparing arrays...')
    f = open(train_file, "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    ###################################################
    user_hotel_pereference= defaultdict(lambda: defaultdict(int))
    best_hotel_search_dest_with_purpose = defaultdict(lambda: defaultdict(int))
    
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

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

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        
        if srch_destination_id != '':
            best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
        
        if hotel_country != '':
            best_hotel_country[hotel_country][hotel_cluster] += append_2
        
        popular_hotel_cluster[hotel_cluster] += 1
        
        ####################################################3
        if user_id != '':
            user_hotel_pereference[user_id][hotel_cluster] += append_1
            
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014 and srch_rm_cnt > 0 and ( ( srch_adults_cnt + srch_children_cnt)*1.0 / srch_rm_cnt <= 1.0 ):
            best_hotel_search_dest_with_purpose[srch_destination_id, "Business"][hotel_cluster] += append_1
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014 and srch_rm_cnt > 0 and ( srch_children_cnt > 0 or ( srch_adults_cnt + srch_children_cnt)*1.0 / srch_rm_cnt > 1.0 ):                                               
            best_hotel_search_dest_with_purpose[srch_destination_id, "Leisure"][hotel_cluster] += append_1
    
    f.close()

    print('Generate submission...')
    now = datetime.datetime.now()
    path = output_path + 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open(test_file, "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        
        ##########################################################
        user_id = arr[8]
        srch_adults_cnt=int(arr[14])
        srch_children_cnt=int(arr[15])
        srch_rm_cnt=int(arr[16])

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        
        
        ###############################################################
        purpose = "None"
        if srch_rm_cnt > 0 and ( srch_adults_cnt + srch_children_cnt)*1.0 / srch_rm_cnt <= 1.0: 
            purpose = "Business"
        if srch_rm_cnt > 0 and ( srch_children_cnt >0 or ( srch_adults_cnt + srch_children_cnt)*1.0 / srch_rm_cnt > 1.0 ): 
            purpose = "Leisure"
        k1 = (srch_destination_id, hotel_country, hotel_market, purpose )
        if k1 in best_hotel_search_dest_with_purpose:
            d = best_hotel_search_dest_with_purpose[k1]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
        else:
            s2 = (srch_destination_id, hotel_country, hotel_market)
            if s2 in best_hotels_search_dest:
                d = best_hotels_search_dest[s2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    
            elif srch_destination_id in best_hotels_search_dest1:
                d = best_hotels_search_dest1[srch_destination_id]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

        if hotel_country in best_hotel_country:
            d = best_hotel_country[hotel_country]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')

run_solution()
