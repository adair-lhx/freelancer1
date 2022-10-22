import torch
import warnings
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pprint
warnings.filterwarnings('ignore')
"""数值特征归一化,min_max标准化"""
def MinMaxscale(bid_fl_pre):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    MinMaxscale_list = ["num_rating_fl", "num_client_rating", "num_reviews_fl", "num_earnings_fl", "num_client_reviews"]
    for i in MinMaxscale_list:
        MinMaxscale_col = bid_fl_pre[[i]].apply(max_min_scaler)

        bid_fl_pre = bid_fl_pre.drop(i, axis=1)

        bid_fl_pre[i] = MinMaxscale_col
        # bid_fl_pre1 = bid_fl_pre[['id_fl', 'id_job', 'num_rating_fl','num_client_rating',\
        #                                          'bool_corporate_fl', 'txt_country_fl', 'num_reviews_fl',\
        #                                          'num_earnings_fl', 'bool_preferred_fl', 'bool_verified_by_fl',\
        #                                          'bool_identity_verified_fl', 'bool_payment_verified_fl',\
        #                                          'bool_phone_verified_fl', 'bool_facebook_connected_fl',\
        #                                          'num_client_reviews', 'bool_payment_verified_cl',\
        #                                          'bool_profile_completed_cl','bool_phone_verified_cl',\
        #                                          'bool_deposit_made_cl',  'list_skill_fl', 'list_skill_cl',\
        #                                          'txt_title_fl', 'txt_description_fl', 'txt_title_cl', \
        #                                          'txt_description_cl', 'target_num_rating', \
        #                                          'list_certification_fl', 'bool_email_verified_fl',\
        #                                          'url_job', 'bool_awarded',"bid_id_fl"]]
        bid_fl_pre1 = bid_fl_pre[['id_fl', 'id_job', 'num_rating_fl', 'num_client_rating', \
                                  'bool_corporate_fl', 'txt_country_fl', 'num_reviews_fl', \
                                  'num_earnings_fl', 'bool_preferred_fl', 'bool_verified_by_fl', \
                                  'bool_identity_verified_fl', 'bool_payment_verified_fl', \
                                  'bool_phone_verified_fl', 'bool_facebook_connected_fl', \
                                  'num_client_reviews', 'bool_payment_verified_cl', \
                                  'bool_profile_completed_cl', 'bool_phone_verified_cl', \
                                  'bool_deposit_made_cl', 'list_skill_fl', 'list_skill_cl', \
                                  'txt_title_fl', 'txt_description_fl', 'txt_title_cl', \
                                  'txt_description_cl', 'target_num_rating', \
                                  'list_certification_fl', 'bool_email_verified_fl','url' \
                                  ]]
    return bid_fl_pre1

def pre_data():
    bid_fl_pre = pd.read_csv("./data/part_50.csv")
    """-----------对num_rating_fl,num_client_rating,num_reviews_fl,num_earnings_fl,num_client_reviews进行归一化-------"""
    bid_fl_pre = MinMaxscale(bid_fl_pre)
    """处理类别数据txt_country"""
    le = preprocessing.LabelEncoder()
    # print("le",le)
    le.fit(bid_fl_pre["txt_country_fl"].values.tolist())
    cou = bid_fl_pre["txt_country_fl"].values.tolist()
    index = le.transform(bid_fl_pre["txt_country_fl"].values.tolist())
    cou_index ={}
    for i in range(len(index)):
        if cou[i] not in cou_index.keys():
            cou_index[cou[i]] = index[i]


    # transform 以后，这一列数就变成了 [0,  n-1] 这个区间的数，即是  le.classes_ 中的索引
    bid_fl_pre["txt_country_fl"] = le.transform(bid_fl_pre["txt_country_fl"].values.tolist())

    """处理sparse_feature和dense_feature"""
    change_feas_fl_map = {}
    unchange_feas_fl_map = {}
    other_feas_cl_map = {}
    feature_info = {}
    unchange_feas_fl =["bool_corporate_fl","txt_country_fl","num_reviews_fl","num_earnings_fl"]
    feature_info['unchange_feas_fl']  = unchange_feas_fl

    change_feas_fl = ["bool_preferred_fl","bool_verified_by_fl","bool_identity_verified_fl","bool_payment_verified_fl",\
                      "bool_phone_verified_fl",'bool_facebook_connected_fl']
    feature_info['change_feas_fl']  = change_feas_fl

    other_feas_cl=["num_client_reviews","bool_payment_verified_cl","bool_profile_completed_cl","bool_phone_verified_cl",\
                   "bool_deposit_made_cl"]
    feature_info['other_feas_cl'] =  other_feas_cl
    """--------------对unchange_fea_fl、change_fea_fl和other_fea_cl进行sparse和dense的划分——————————————————"""

    for key in unchange_feas_fl[0:2]:
        unchange_feas_fl_map[key] = bid_fl_pre[key].nunique()
    for key in change_feas_fl:
        change_feas_fl_map[key] = bid_fl_pre[key].nunique()
    for key in other_feas_cl[1:]:
        other_feas_cl_map[key] = bid_fl_pre[key].nunique()
    # feature_info = [unchange_feas_fl, unchange_feas_fl_map, change_feas_fl,change_feas_fl_map,other_feas_cl,other_feas_cl_map]
    return feature_info,bid_fl_pre, cou_index
if __name__ == "__main__":
    bid_fl_pre = pd.read_csv("./data/part_50.csv")
    # bid_fl_pre = pd.read_csv("test2.csv")
    bid_fl_pre1 = MinMaxscale(bid_fl_pre)
    pre_data()
