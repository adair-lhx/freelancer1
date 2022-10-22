#coding:utf-8
# @Time      : 2022/5/30
# @Author    : lhx
import numpy as np
from gurobipy import *
import time
import Pretain_data
import skill_deal1
import torch
from sklearn.metrics.pairwise import cosine_similarity

def reindex(ratings):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    skill_fl ={}
    skill_cl = {}
    skill_fl_temp = {}

    user_list = list(ratings['id_fl'].drop_duplicates())

    user2id = {w: i for i, w in enumerate(user_list)}


    item_list = list(ratings['id_job'].drop_duplicates())

    item2id = {w: i for i, w in enumerate(item_list)}

    ratings['id_fl'] = ratings['id_fl'].apply(lambda x: user2id[x])
    ratings['id_job'] = ratings['id_job'].apply(lambda x: item2id[x])
    ratings['target_num_rating'] = ratings['target_num_rating'].apply(lambda x: float(x > 0))
    list_skill_fl = list(ratings['list_skill_fl'].drop_duplicates())

    # skil list

    list2 = ratings[['id_job', 'list_skill_cl']]
    list3 = list2.drop_duplicates()
    for i in range(len(user_list)):

        list_skill_fl[i] = re.sub("[\!\%\[\]\。\(\)\'\_\ ]", "", list_skill_fl[i])

        list_skill_fl[i] = list_skill_fl[i].split(',')

        temp_list = []
        for j in range(0, len(list_skill_fl[i]), 2):
            if int(list_skill_fl[i][j + 1]) > 0:
                temp_list.append(list_skill_fl[i][j])
        skill_fl[i] = temp_list.copy()
        skill_fl_temp[i] = temp_list.copy()


    for i in range(len(item_list)):

        temp = re.sub("[\!\%\[\]\。\(\)\'\_\  ]", "", list3.iloc[i, 1])

        temp = temp.split(',')
        temp_list = []
        for j in range(0, len(temp)):
            temp_list.append(temp[j])
        skill_cl[i] = temp_list.copy()

    return ratings,user_list,item_list,skill_cl,skill_fl,skill_fl_temp

def reindex1(ratings):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    skill_fl ={}
    skill_cl = {}

    user_list = list(ratings['id_fl'].drop_duplicates())
    print("freelancers",len(user_list))
    user2id = {w: i for i, w in enumerate(user_list)}


    item_list = list(ratings['id_job'].drop_duplicates())
    print("jobs", len(item_list))
    item2id = {w: i for i, w in enumerate(item_list)}

    ratings['id_fl'] = ratings['id_fl'].apply(lambda x: user2id[x])
    ratings['id_job'] = ratings['id_job'].apply(lambda x: item2id[x])
    ratings['target_num_rating'] = ratings['target_num_rating'].apply(lambda x: float(x > 0))
    list_skill_fl = list(ratings['list_skill_fl'].drop_duplicates())

    # skil list

    list2 = ratings[['id_job','list_skill_cl']]
    list3 = list2.drop_duplicates()
    for i in range(len(user_list)):

        list_skill_fl[i] = re.sub("[\!\%\[\]\。\(\)\'\_\ ]", "", list_skill_fl[i])

        list_skill_fl[i] = list_skill_fl[i].split(',')

        temp_list =[]
        for j in range(0,len(list_skill_fl[i]),2):
            if int(list_skill_fl[i][j+1]) > 0:
                temp_list.append(list_skill_fl[i][j])
        skill_fl[i] = temp_list


    for i in range(len(item_list)):

        temp = re.sub("[\!\%\[\]\。\(\)\'\_\  ]", "", list3.iloc[i,1])

        temp = temp.split(',')
        temp_list = []
        for j in range(0, len(temp)):
            temp_list.append(temp[j])
        skill_cl[i] = temp_list


    return ratings,skill_cl,skill_fl



feature_info,bid_fl_pre,_ = Pretain_data.pre_data()


bid_fl_pre = bid_fl_pre.reset_index()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# bid_fl_pre,skill_cl,skill_fl = reindex1(bid_fl_pre)
bid_fl_pre, freelancer_list, job_list,skill_cl,skill_fl,skill_fl_temp= reindex(bid_fl_pre)


# t_fl, list_fl_skill, fl_train_skill_list, cl_train_skill_list, = skill_deal1.skill_to_tensor3(bid_fl_pre, skill_cl, skill_fl, skill_fl_temp)
# t_fl, t_cl, fl_train_skill_list, cl_train_skill_list, graph = skill_deal1.skill_to_tensor2(bid_fl_pre, skill_cl,
#                                                                                                skill_fl)
list_fl_skill = skill_deal1.skill_list(skill_fl, skill_cl)

model_rec = torch.load('cpu_50_de_10_50.pth')
# model_rec = torch.load('cpu_50.pth')
# print(model_rec.parameters)
# # for param_tensor in model_rec.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
# #     print(param_tensor,'\t',model_rec.state_dict()[param_tensor].size())
# exit()
t_fl = torch.load('t_fl.pth')
fl_train_skill_list = torch.load('fl.pth')
cl_train_skill_list = torch.load('cl.pth')
# model_rec = torch.load('ssl_50_nonneg.pth')
# model_rec = torch.load('ssl_50_de.pth')
# job = 200
# freelancer_2 = 133
# freelancer_1 = 233
job = 50
freelancer_2 = 61
freelancer_1 = 401
print("fl_1",list(bid_fl_pre[bid_fl_pre['id_fl'] ==freelancer_1]['list_skill_fl'])[0])
print("fl_2",list(bid_fl_pre[bid_fl_pre['id_fl'] ==freelancer_2]['list_skill_fl'])[0])
print("cl",list(bid_fl_pre[bid_fl_pre['id_job'] ==job ]['list_skill_cl'])[0])



job = torch.tensor(job,device=device)
freelancer_2 = torch.tensor(freelancer_2,device=device)
freelancer_1 = torch.tensor(freelancer_1,device=device)


a = torch.tensor(cl_train_skill_list[job], device=device, dtype=torch.bool)

b_2 = torch.tensor(fl_train_skill_list[freelancer_2], device=device, dtype=torch.bool)
b_1 = torch.tensor(fl_train_skill_list[freelancer_1], device=device, dtype=torch.bool)

embedding_job_m,embedding_lancer  = model_rec.get_skill(t_fl,a)
'''job pipei'''
job_pipei ={}
job_num = {}
job_mean ={}
job_sum = {}
yu_num = {}
for i in range(cl_train_skill_list.size(0)):
    a = torch.tensor(cl_train_skill_list[i], dtype=torch.int64)
    yuzhi = torch.nonzero(a).size(0)
    if  yuzhi not in yu_num.keys():
        yu_num[ yuzhi] = 1
        for z in range(1,int( yuzhi)):
            if z not in yu_num.keys():
                yu_num[z] = 1
            else:
                yu_num[z] = yu_num[z] +1
    else:
        yu_num[ yuzhi] = yu_num[ yuzhi] +1
        for z in range(1,int( yuzhi)):
            if z not in yu_num.keys():
                yu_num[z] = 1
            else:
                yu_num[z] = yu_num[z] +1

    job_pipei[i] ={}
    for j in range(fl_train_skill_list.size(0)):

        b = torch.tensor(fl_train_skill_list[j], dtype=torch.int64)
        c = a & b
        pipei = torch.nonzero(c).size(0)

        job_pipei[i][yuzhi] = 0
        if pipei>0:
            for num in range(1,pipei+1):
               if num in  job_pipei[i].keys():
                   job_pipei[i][num] = job_pipei[i][num] +1
               else:
                   job_pipei[i][num] = 1
    for w in job_pipei[i].keys():
        if w not in job_num.keys():
            job_num[w] = [job_pipei[i][w]]
        else:
            job_num[w].append(job_pipei[i][w])

for q in job_num.keys():
    job_sum[q] = np.sum(np.array(job_num[q]))
    job_mean[q] = job_sum[q]/yu_num[q]
print(job_mean)
print( job_sum)
print(job_pipei)
print("yu_zhi",yu_num)
exit()
# '''yueshu jinsi'''
zhun = 0
yue = 0
cha = 0
num = 0
bili = 0
bili_1 = []
skill1 = skill_deal1.skill_list(skill_fl, skill_cl)
for i in range(fl_train_skill_list.size(0)):
    i=10
    a = torch.tensor(fl_train_skill_list[i], device=device, dtype=torch.bool)
    embedding_job_m, embedding_lancer = model_rec.get_skill(t_fl, a)
    embedding_job_new = embedding_job_m
    b2 = cosine_similarity(embedding_job_new.cpu().detach().numpy().reshape(1, -1),
                           embedding_lancer.cpu().detach().numpy())  # freelancer sim
    #
    b3 = torch.tensor(b2)
    _, indices = torch.topk(b3, 300)
    skill_freelancer_2 = skill_fl[i]
    indices = indices.cpu().detach().numpy().tolist()

    for i in skill_freelancer_2:
        if skill1.index(i) in indices[0]:
            indices[0].remove(skill1.index(i))
    embedding_lancer = embedding_lancer[indices]
    print(embedding_lancer.size())
    for j in range(embedding_lancer.size(0)):
        embedding_job_new = embedding_job_m
        for q in range(j):
            embedding_job_new  =embedding_job_new+ embedding_lancer[q]
        embedding_m1 = torch.mul( embedding_job_new, embedding_lancer[j])
        skill_embedding_m1 = torch.sum(embedding_m1).unsqueeze(0)


        w = skill_embedding_m1.detach().numpy()

        # bili = bili+(1 - skill_embedding_m1.detach().numpy() )/ np.exp(-skill_embedding_m1.detach().numpy())
        # bili_1.append(bili)
        cha =  np.exp(-w) - 1 + w
        zhun =  np.exp(-w)
        yue =  1 - w
        if yue < 0:
            print("j",j)
        print("yue",yue)
        print("zhun",zhun)
        print("cha",cha)
        print("bili", bili)
        if j ==250:
            exit()
#
# # print("bili",bili_1)
# exit()
#
# zhun = 0
# yue = 0
# cha = 0
# num = 0
# bili = 0
# bili_1 = []
# for j in range(embedding_lancer.size(0)):
#     for q in range(j + 1, embedding_lancer.size(0)):
#         num =num +1
#         embedding_m2 = torch.mul(embedding_lancer[j], embedding_lancer[q])
#         skill_embedding_m2 = torch.sum(embedding_m2).unsqueeze(0)
#         cha =cha + np.exp(-skill_embedding_m2.detach().numpy()) -1 + skill_embedding_m2.detach().numpy()
#         bili = bili +(1 - skill_embedding_m2.detach().numpy())/np.exp(-skill_embedding_m2.detach().numpy())
#         bili_1.append(bili)
#         zhun = zhun +np.exp(-skill_embedding_m2.detach().numpy())
#         yue =yue +1-skill_embedding_m2
# print("yue",yue/num)
# print("zhun",zhun/num)
# print("cha",cha/num)
# print("bili", bili/num)
# print("bili",bili_1)

zhun = 0
yue = 0
cha = 0
num = 0
bili = 0
bili_1 = []
skill1 = skill_deal1.skill_list(skill_fl, skill_cl)
for i in range(fl_train_skill_list.size(0)):

    a = torch.tensor(fl_train_skill_list[i], device=device, dtype=torch.bool)
    embedding_job_m, embedding_lancer = model_rec.get_skill(t_fl, a)
    b2 = cosine_similarity(embedding_job_m.cpu().detach().numpy().reshape(1, -1),
                           embedding_lancer.cpu().detach().numpy())  # freelancer sim
    #
    b3 = torch.tensor(b2)
    _, indices = torch.topk(b3, 200)
    skill_freelancer_2 = skill_fl[i]
    indices = indices.cpu().detach().numpy().tolist()
    for i in skill_freelancer_2:
        if skill1.index(i) in indices[0]:
            indices[0].remove(skill1.index(i))
    embedding_lancer = embedding_lancer[indices]
    # print(embedding_lancer.size())
    for j in range(embedding_lancer.size(0)):
        num = num +1
        embedding_m1 = torch.mul(embedding_job_m, embedding_lancer[j])
        skill_embedding_m1 = torch.sum(embedding_m1).unsqueeze(0)
        w = skill_embedding_m1.detach().numpy()
        bili = bili+(1 - skill_embedding_m1.detach().numpy() )/ np.exp(-skill_embedding_m1.detach().numpy())
        bili_1.append(bili)
        cha = cha + np.exp(-w) - 1 + w
        zhun = zhun + np.exp(-w)
        yue = yue + 1 - w
print("yue",yue/num)
print("zhun",zhun/num)
print("cha",cha/num)
print("bili", bili/num)
# print("bili",bili_1)
exit()

count = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
for i in range(fl_train_skill_list.size(0)):
    print(i)
    a = torch.tensor(fl_train_skill_list[i], device=device, dtype=torch.bool)
    embedding_job_m, embedding_lancer = model_rec.get_skill(t_fl, a)
    b2 = cosine_similarity(embedding_job_m.cpu().detach().numpy().reshape(1, -1),
                           embedding_lancer.cpu().detach().numpy())  # freelancer sim
    #
    b3 = torch.tensor(b2)
    _, indices = torch.topk(b3, 150)
    indices = indices.cpu().detach().numpy().tolist()
    embedding_lancer = embedding_lancer[indices]
    print("size***", embedding_lancer.size())

    for j in range(embedding_lancer.size(0)):
        embedding_m1 = torch.mul(embedding_job_m, embedding_lancer[j])
        skill_embedding_m1 = torch.sum(embedding_m1).unsqueeze(0)
        if skill_embedding_m1 > 0.5:
            count = count + 1

        if skill_embedding_m1 > 1:
            count1 = count1 + 1

        if skill_embedding_m1 > 0.1:
            count2 = count2 + 1
        for q in range(j + 1, embedding_lancer.size(0)):
            embedding_m2 = torch.mul(embedding_lancer[j], embedding_lancer[q])
            skill_embedding_m2 = torch.sum(embedding_m2).unsqueeze(0)
            if skill_embedding_m2 > 0.5:
                count3 = count3 + 1

            if skill_embedding_m2 > 0.1:
                count4 = count4 + 1

            if skill_embedding_m2 > 1:
                count5 = count5 + 1

print(count,count1,count2)
print(count3,count4,count5)
exit()