#coding:utf-8
import streamlit as st
import torch
import torch.nn as nn
import Pretain_data
import warnings
import skill_deal1
import data_utils
import argparse
import model
import torch.optim as optim
import time
import config
import  evaluate
import re
import numpy as np

from gurobipy import *
import time
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

survey1  = np.random.randint(10000)
index =  np.random.randint(1,5)

if 'compe' not in st.session_state:
    st.session_state['compe'] = index
else:
    index = st.session_state['compe']

if 'id' not in st.session_state:
    while os.path.exists('./data/'+str(survey1)):
        survey1 = np.random.randint(10000)
    st.session_state['id'] =survey1
    os.mkdir('./data/'+str(survey1)+'_'+str(st.session_state['compe']))
else:
    survey1 =  st.session_state['id']




print("index",index)
print("survey1 ",survey1 )

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--layers",
                        nargs='+',
                        default=[64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--hidden_units",
                        nargs='+',
                        default=[51, 32, 16, 8],
                        help="")
    parser.add_argument("--layer1",
                        nargs='+',
                        default=[512, 128, 64,8],
                        help="")
    parser.add_argument("--layer3",
                        nargs='+',
                        default=[50, 16, 8, 4],
                        help="")
    parser.add_argument("--layer4",
                        nargs='+',
                        default=[2, 2, 2],
                        help="")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test",
                        type=int,
                        default=80,
                        help="Number of negative samples for test set")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")

    # mf_dim = 4  # embeding_dim的维度
    # set device and parameters
    args = parser.parse_args()
    # 设置模型的参数
    return args

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

    item_list = list(ratings['id_job'].drop_duplicates())

    item2id = {w: i for i, w in enumerate(item_list)}



    ratings['id_job'] = ratings['id_job'].apply(lambda x: item2id[x])


    return ratings

def hackpack(model_rec,job,freelancer_2,freelancer_1,t_fl,fl_train_skill_list,cl_train_skill_list,skill1,feature2,device,feature_info,bid_fl_pre):


    job = torch.tensor(job, device=device)
    print("job",job)
    freelancer_2 = torch.tensor(freelancer_2, device=device)
    print("freelancer_2", freelancer_2)
    freelancer_1 = torch.tensor(freelancer_1, device=device)
    print("freelancer_1", freelancer_1)
    # get skill embedding
    a = torch.tensor(cl_train_skill_list[job], device=device, dtype=torch.bool)
    print("a",a)
    b_2 = torch.tensor(fl_train_skill_list[freelancer_2], device=device, dtype=torch.bool)
    print("b_2", b_2)
    b_1 = torch.tensor(fl_train_skill_list[freelancer_1], device=device, dtype=torch.bool)
    print("b_1", b_1)
    # embedding_lancer_m1 = model_rec.linear_cl_skill(b_1)
    #
    # embedding_lancer_m2 = model_rec.linear_cl_skill(b_2)
    t_fl = torch.tensor(t_fl, device=device)

    embedding_lancer_m1, _ = model_rec.get_skill(t_fl, b_1)

    embedding_lancer_m2, _ = model_rec.get_skill(t_fl, b_2)
    print("embedding_lancer_m2",  embedding_lancer_m2)
    embedding_job_m, embedding_lancer = model_rec.get_skill(t_fl, a)
    print(" embedding_job_m",  embedding_job_m)
    embedding_m1 = torch.mul(embedding_job_m, embedding_lancer_m1)

    embedding_m2 = torch.mul(embedding_job_m, embedding_lancer_m2)
    print("embedding_m2", embedding_m2)
    skill_embedding_m1 = torch.sum(embedding_m1).unsqueeze(0)

    skill_embedding_m2 = torch.sum(embedding_m2).unsqueeze(0)
    print("skill_embedding_m2", skill_embedding_m2)
    # get weight
    W_fea = model_rec.feature.weight
    W_score = model_rec.affine_output.weight
    W_C = model_rec.linear_skill.weight

    # get rate embedding
    log1, rate_vector_1 = model_rec.get_rate(job, freelancer_1)

    log2, rate_vector_2 = model_rec.get_rate(job, freelancer_2)

    # get feature embedding
    feature1 = {}


    # feature2 = {}
    # print(feature_info)
    # print("feature_info",feature_info)
    for index in feature_info.keys():
        w = feature_info[index]
        if index == 'other_feas_cl':
            for f in w:
                feature1[f] = list(bid_fl_pre[bid_fl_pre['id_job'] == int(job)][f])[0]
                feature1[f] = torch.tensor(feature1[f], device=device, dtype=torch.int32)
                feature2[f] = list(bid_fl_pre[bid_fl_pre['id_job'] == int(job)][f])[0]
                feature2[f] = torch.tensor(feature2[f], device=device, dtype=torch.int32)
        else:
            for f in w:
                feature1[f] = list(bid_fl_pre[bid_fl_pre['id_fl'] == int(freelancer_1)][f])[0]
                feature1[f] = torch.tensor(feature1[f], device=device, dtype=torch.int32)
                # feature2[f] = list(bid_fl_pre[bid_fl_pre['id_fl'] == int(freelancer_2)][f])[0]
                feature2[f] = int(feature2[f])
                feature2[f] = torch.tensor(feature2[f], device=device, dtype=torch.int32)

    feature_vector_1, feature_grade1 = model_rec.get_fea(feature1)

    feature_vector_2, feature_grade2 = model_rec.get_fea(feature2)

    # print("torch.matmul(W_fea,feature_vector_1)", torch.matmul(W_fea, feature_vector_1))
    # print("torch.matmul(W_score,rate_vector_1)", torch.matmul(W_score, rate_vector_1))
    # print("torch.matmul(W_C,skill_embedding_m1)", torch.matmul(W_C, skill_embedding_m1))
    # print("torch.matmul(W_fea,feature_vector_2)", torch.matmul(W_fea, feature_vector_2))
    # print("torch.matmul(W_score,rate_vector_2)", torch.matmul(W_score, rate_vector_2))

    V = torch.matmul(W_fea, feature_vector_1) + torch.matmul(W_score, rate_vector_1) + torch.matmul(W_C,skill_embedding_m1) \
        - torch.matmul(W_fea, feature_vector_2) - torch.matmul(W_score, rate_vector_2)
    print("V",V)
    print("W_C",W_C)
    # print("W_C", W_C)
    U = V / W_C - skill_embedding_m2


    # if W_C >0:
    #     U = -U
    print("U", U)

    # skill = skill_deal1.skill_list(skill_fl, skill_cl)

    # skill_freelancer_2 = skill_fl[int(freelancer_2)]
    skill_freelancer_2 = feature2['skill']
    print(skill_freelancer_2)
    skill = skill1.copy()
    mask = np.ones(len(skill))
    # print("mask",mask.shape)
    b2 = cosine_similarity(embedding_lancer_m2.cpu().detach().numpy().reshape(1, -1),
                           embedding_lancer.cpu().detach().numpy())  # freelancer sim
    #
    b3 = torch.tensor(b2)
    _, indices = torch.topk(b3, 892)
    indices = indices.cpu().detach().numpy().tolist()
    skill_index = {}
    num = 1
    for i in indices[0]:
        skill_index[skill1[i]] = num
        num = num + 1
    # print("index", skill_index)
    # new_train_mask = new_train_mask > 0
    # print("skill", skill)
    for i in skill_freelancer_2:
        mask[skill.index(i)] = 0
    mask = mask > 0

    for i in skill_freelancer_2:
        skill.remove(i)
    x = {}
    model = Model("freelancer")
    skill_name = {}
    # define the varaibles
    for i in range(len(skill)):
        x[i] = model.addVar(0, 1, vtype=GRB.BINARY, name="x{0}".format(i))
        skill_name[i] = skill[i]

    # set the objective
    d = -1
    expr = LinExpr(0)
    for i in range(len(skill)):
        expr.addTerms(d, x[i])
    # model.setObjective(expr, sense=GRB.MINIMIZE)
    model.setObjective(expr, sense=GRB.MAXIMIZE)
    # add the constrains_1
    expr = LinExpr(0)

    # embedding_lancer = model_rec.linear_fl_skill(torch.tensor(t_fl,device=device))
    # embedding_job = model_rec.linear_fl_skill(torch.tensor(t_cl,device=device))

    # a1 = torch.mul(embedding_job,embedding_lancer )

    a1 = torch.mul(embedding_job_m, embedding_lancer)
    a1 = torch.sum(a1, dim=-1)
    print("size",a1.size())
    a1 = a1[mask]
    # _, index = torch.topk(a1, 20)
    # print(index)
    # for i in index:
    #     print(skill_name[int(i)])

    for i in range(len(skill)):
        expr.addTerms(a1[i], x[i])

    model.addConstr(expr >= U, name="c_1")

    model.write("example.lp")
    # solution
    start_time = time.time()
    model.setParam("Thread", 8)
    model.setParam("TimeLimit", 10)
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time
    skill_add = []
    if model.status == GRB.OPTIMAL:
        s_x = {}
        for key in x.keys():
            s_x[key] = x[key].x
            if x[key].x != 0:
                # print("key", key)
                skill_add.append([skill_name[key],skill_index[skill_name[key]],892])
                # print("value", a1[key])
        # print("x = ", s_x)

        obj = model.getObjective()
        # print("obj = ", obj.getValue())
        #
        # print("cpu = %.2f sec" % cpu_time)
    return skill_add

def dp_hackpack(model_rec,job,freelancer_2,freelancer_1,t_fl,fl_train_skill_list,cl_train_skill_list,skill1,feature2,device,feature_info,bid_fl_pre):

    job = torch.tensor(job, device=device)
    print("job ", job )
    freelancer_2 = torch.tensor(freelancer_2, device=device)
    print("freelancer_2  ", freelancer_2 )
    freelancer_1 = torch.tensor(freelancer_1, device=device)
    # get skill embedding
    a = torch.tensor(cl_train_skill_list[job], device=device, dtype=torch.bool)

    b_2 = torch.tensor(fl_train_skill_list[freelancer_2], device=device, dtype=torch.bool)

    b_1 = torch.tensor(fl_train_skill_list[freelancer_1], device=device, dtype=torch.bool)
    # embedding_lancer_m1 = model_rec.linear_cl_skill(b_1)
    #
    # embedding_lancer_m2 = model_rec.linear_cl_skill(b_2)
    t_fl = torch.tensor(t_fl, device=device)
    embedding_lancer_m1, _ = model_rec.get_skill(t_fl, b_1)

    embedding_lancer_m2, _ = model_rec.get_skill(t_fl, b_2)

    embedding_job_m, embedding_lancer = model_rec.get_skill(t_fl, a)

    embedding_m1 = torch.mul(embedding_job_m, embedding_lancer_m1)

    embedding_m2 = torch.mul(embedding_job_m, embedding_lancer_m2)

    skill_embedding_m1 = torch.sum(embedding_m1).unsqueeze(0)

    skill_embedding_m2 = torch.sum(embedding_m2).unsqueeze(0)

    # get weight
    W_fea = model_rec.feature.weight
    W_score = model_rec.affine_output.weight
    W_C = model_rec.linear_skill.weight

    # get rate embedding
    log1, rate_vector_1 = model_rec.get_rate(job, freelancer_1)

    log2, rate_vector_2 = model_rec.get_rate(job, freelancer_2)

    # get feature embedding
    feature1 = {}



    # feature2 = {}
    # print(feature_info)

    for index in feature_info.keys():
        w = feature_info[index]
        if index == 'other_feas_cl':
            for f in w:
                feature1[f] = list(bid_fl_pre[bid_fl_pre['id_job'] == int(job)][f])[0]
                feature1[f] = torch.tensor(feature1[f], device=device, dtype=torch.int32)
                feature2[f] = list(bid_fl_pre[bid_fl_pre['id_job'] == int(job)][f])[0]
                feature2[f] = torch.tensor(feature2[f], device=device, dtype=torch.int32)
        else:
            for f in w:
                feature1[f] = list(bid_fl_pre[bid_fl_pre['id_fl'] == int(freelancer_1)][f])[0]
                feature1[f] = torch.tensor(feature1[f], device=device, dtype=torch.int32)
                # feature2[f] = list(bid_fl_pre[bid_fl_pre['id_fl'] == int(freelancer_2)][f])[0]
                feature2[f] = int(feature2[f])
                feature2[f] = torch.tensor(feature2[f], device=device, dtype=torch.int32)

    feature_vector_1, feature_grade1 = model_rec.get_fea(feature1)

    feature_vector_2, feature_grade2 = model_rec.get_fea(feature2)

    # print("torch.matmul(W_fea,feature_vector_1)", torch.matmul(W_fea, feature_vector_1))
    # print("torch.matmul(W_score,rate_vector_1)", torch.matmul(W_score, rate_vector_1))
    # print("torch.matmul(W_C,skill_embedding_m1)", torch.matmul(W_C, skill_embedding_m1))
    # print("torch.matmul(W_fea,feature_vector_2)", torch.matmul(W_fea, feature_vector_2))
    # print("torch.matmul(W_score,rate_vector_2)", torch.matmul(W_score, rate_vector_2))

    V = torch.matmul(W_fea, feature_vector_1) + torch.matmul(W_score, rate_vector_1) + torch.matmul(W_C,
                                                                                                    skill_embedding_m1) \
        - torch.matmul(W_fea, feature_vector_2) - torch.matmul(W_score, rate_vector_2)


    # print("W_C", W_C)
    U = V / W_C - skill_embedding_m2
    print("V", V)
    print("W_C", W_C)
    print("U",U )
    print("skill_embedding_m2",skill_embedding_m2)
    skill_freelancer_2 = feature2['skill']


    mask = np.zeros(len(skill1))




    # b2 = cosine_similarity(embedding_job_m.cpu().detach().numpy().reshape(1,-1),embedding_lancer.cpu().detach().numpy())#job sim
    b2 = cosine_similarity(embedding_lancer_m2.cpu().detach().numpy().reshape(1, -1),
                           embedding_lancer.cpu().detach().numpy())  # freelancer sim
    #
    b3 = torch.tensor(b2)
    _, indices = torch.topk(b3,100)
    indices = indices.cpu().detach().numpy().tolist()

    skill_index = {}
    skill = []
    num = 1
    for i in indices[0]:
        skill_index[skill1[i]] = num
        num = num + 1
    for i in skill_freelancer_2:
        if skill1.index(i) in indices[0]:
            indices[0].remove(skill1.index(i))

    for i in indices[0]:
        mask[i] = 1
        skill.append(skill1[i])
    embedding_lancer = embedding_lancer[indices]
    mask = mask > 0

    x = {}
    model = Model("freelancer")
    skill_name = {}
    # define the varaibles
    for i in range(len(skill)):
        x[i] = model.addVar(0, 1, vtype=GRB.BINARY, name="x{0}".format(i))
        skill_name[i] = skill[i]

    # embedding_lancer = embedding_lancer[mask]

    # set the objective
    # set the objective
    a1 = torch.mul(embedding_lancer_m2, embedding_lancer)
    a1 = torch.sum(a1, dim=-1)

    a2 = torch.mul(embedding_job_m, embedding_lancer)
    a2 = torch.sum(a2, dim=-1)

    expr = LinExpr(0)
    expr0 = LinExpr(0)
    expr1 = QuadExpr()
    d = -1
    ji = 7


    for i in range(len(skill)):
        # d = -1*(a1[i]+1)
        d0 = a1[i]
        expr.addTerms(d, x[i])
        expr0.addTerms(d0, x[i])
        for j in range(i + 1, len(skill)):
            et = torch.mul(embedding_lancer[i], embedding_lancer[j])
            et = -1 * torch.sum(et, dim=-1).cpu().detach().numpy().tolist()
            expr1.addTerms(et, x[i], x[j])

    # model.setObjective(expr, sense=GRB.MAXIMIZE)
    model.setObjective(expr + ji * expr0 + ji * expr1, sense=GRB.MAXIMIZE)
    # add the constrains_1
    expr2 = LinExpr(0)



    for i in range(len(skill)):
        expr2.addTerms(a2[i], x[i])

    model.addConstr(expr2 >= U, name="c_1")

    model.write("example.lp")
    # solution

    model.setParam("Thread", 8)
    model.setParam("TimeLimit", 10)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        s_x = {}
        skill_add = []
        for key in x.keys():
            s_x[key] = x[key].x
            if x[key].x != 0:
                skill_add.append([skill_name[key],skill_index[skill_name[key]],892])


    return skill_add

def test(model,jobs, freelancers_fal,freelancers, feature1, fl_train_skill_list, cl_train_skill_list, t_fl, device):
    job_bool = False

    fl_train_skill = fl_train_skill_list[freelancers_fal]

    cl_train_skill = cl_train_skill_list[jobs.tolist()]


    jobs = torch.tensor(jobs, device=device, dtype=torch.long)

    fl_train_skill = torch.tensor(fl_train_skill, device=device, dtype=torch.bool)
    cl_train_skill = torch.tensor(cl_train_skill, device=device, dtype=torch.bool)
    t_fl = torch.tensor(t_fl, device=device, dtype=torch.float)

    freelancers = torch.tensor(freelancers, device=device, dtype=torch.long)

    index_new = {}
    for i in range(len(freelancers_fal)):
        index_new[i] = freelancers_fal[i]
    print(" index_new",  index_new)
    print(" freelancers_fal",freelancers_fal)
    web = 1
    predictions, _ = model(jobs, freelancers, freelancers_fal,feature1, fl_train_skill, cl_train_skill, t_fl, web)

    # print("predictions",predictions)
    top_k =  predictions.size(0)


    shang = top_k*0.1

    xia = top_k * 0.3

    _, indices = torch.topk(predictions, top_k)
    print("value",_)
    print("indices",indices)
    lis_ind = indices.cpu().numpy().tolist()
    # print(lis_ind)
    # ind = lis_ind.index(indices.size(0)-1)

    print("3344",len(freelancers_fal) - 1)
    ind = lis_ind.index(len(freelancers_fal) - 1)

    print("ind",ind)
    print(int(top_k * 0.05))

    # ind = lis_ind.index(len(freelancers_fal)-1)


    # job_bool = True
    if ind < xia and ind > shang:
        job_bool = True
        print("ind",ind)
        print("top_k", top_k)
        print("job",jobs[0])

        # print("ind_save", ind)


    recommends = indices[:ind+100].tolist()
    print("recommends",recommends)

    #index = [21, 25, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000]

    index = [ind]
    # index =[39]
    freelancer_2 = []
    index_temp = {}
    for i in index:
        freelancer_2.append(index_new[recommends[i]])
        index_temp[index_new[recommends[i]]]  = i

    freelancer_1 = index_new[recommends[int(top_k*0.05)]]

    return freelancers[0],freelancer_1, freelancer_2,index_temp,job_bool

def intro():
    import streamlit as st

    st.write("# 感谢您参与本次调研！ 👋")


    st.markdown(
        """
        以下是关于您参加本调研需提前知晓的相关内容。在您决定同意参加本次调研之前，请仔细阅读本页面。

        **调研目的：**
        本调研旨在研究求职者如何评价基于人工智能（AI）的工作推荐系统。这是一项由大学研究人员开展的基础学术研究。
        
        **调研流程:**
       在本调研的过程中，我们会要求您回答问题或作出选择。请在回答问题前，仔细阅读每个页面上提供的信息。
        
        **调研价值：**
       本调研旨在获取新知识造福社会，例如帮助执政者和国民作出更好的决策。
        
        **调研报酬：**
        如果您完成本次调研，您将获得$$的报酬作为补偿。
        
        **参与条件：**
        您必须年满18岁才能参加本次调研。
        
        **隐私保密：**
        在参与本调研过程中，我们会依法对您所提供的信息和回答进行保密。在学术论文和研究报告中，我们也只会使用加总和平均后的数据。同时，我们不会要求您提供任何可能涉及个人身份的信息。
        
        **实验者权利:**
        您可自愿决定是否参加本调研，并可在调研途中随时停止。本调研已经过机构审查委员会审查和批准。
        
        **对本调研有疑问或对您作为调研对象的权利有疑问：**
        
        如果您有任何疑问，可以联系本调研的工作人员。
        
        如果您选择参与本次调研，即表示您已阅读本页信息，并愿意成为本调研中的志愿者。

        本调研团队对您的付出表示感谢！
    """
    )


def background():
    import streamlit as st

    st.write("# 工作推荐系统研究")

    st.markdown(
        """
        感谢您参与我们的调研！

        本团队构建了一个基于人工智能的工作推荐系统，从而为每位求职者推荐工作。在本次调研过程中，您将会与该人工智能进行互动，并回答一些问题，对提供的情境进行选择。请认真仔细地阅读这些问题，并坦率合理地进行作答。

        注意：无论您如何回答问题，都不会影响最终获得的报酬。此外，您的回答是匿名的，不会暴露您的个人身份。

    """
    )



@st.cache(allow_output_mutation=True)
def load_data():
    feature_info, bid_fl_pre, cou_index = Pretain_data.pre_data()
    money = pd.read_csv("./data/money.csv")
    return feature_info, bid_fl_pre, cou_index, money

def test_demo():
    if 'test_demo' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    st.markdown(
        """
        为了让我们的推荐系统为您推荐一些符合您实际情况的工作机会，请回答以下问题。
    """
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_info, bid_fl_pre, cou_index, money = load_data()
    num_list = []
    # bid_fl_pre = bid_fl_pre.iloc[:,:-2]
    # bid_fl_pre = bid_fl_pre[bid_fl_pre['bool_awarded'] == 1].drop_duplicates( )
    bid_fl_pre, freelancer_list, job_list, skill_cl, skill_fl, skill_fl_temp = reindex(bid_fl_pre)
    money = reindex1(money)
    skill = skill_deal1.skill_list(skill_fl, skill_cl)

    # print("skill",len(skill))
    bid_fl_pre = bid_fl_pre.reset_index()


    num_users = bid_fl_pre['id_job'].nunique() + 1
    num_items = bid_fl_pre['id_fl'].nunique() + 1
    # print("num_users",num_users)
    # print("num_items", num_items)
    # model = torch.load('./models/ssl_50.pth')
    model = torch.load('cpu_50_pos.pth')
    t_fl = torch.load('t_fl.pth')
    fl_train_skill_list = torch.load('fl.pth')
    cl_train_skill_list = torch.load('cl.pth')
    coun = pd.read_csv('country.csv')
    country = []
    for i in range(len(coun)):
        country.append(coun.iloc[i, 0])
    country = tuple(country)

    with st.form("my_form"):
        # a = st.number_input("flreelancer_id")
        job = st.selectbox(
            '您所从事的工作领域是?',
            ('algorithm','web','photo','code','secuirty','data','marketing','video','write'))
        cou = 'China'
        reviews = 10
        earnings = 10
        corporate = 1
        preferred = 1
        verified = 1
        identity = 1
        payment = 1
        phone = 1
        facebook = 1
        skill_fl1 = st.multiselect(
            '您现在掌握的技能有哪些？',
            skill,
            [])
        # Every form must have a submit button.
        submitted = st.form_submit_button("提交")
        job_like1 = []
        job_free ={}

        if submitted:

            # freelancers = [x for x in range(0, len(freelancer_list))]
            tf = open("feature.json", "r")
            feature1 = json.load(tf)

            # tf_fea = open("feature.json", "w")
            # json.dump(feature1, tf_fea)
            # tf_fea.close()

            cou_ind = cou_index[cou]



            feature1['corporate'].append(corporate)
            feature1['country'].append(cou_ind )
            feature1['reviews'].append(reviews)
            feature1['earnings'].append(earnings)
            feature1['preferred'].append(preferred)
            feature1['verified'].append(verified)
            feature1['identity'].append(identity)
            feature1['phone'].append(phone)
            feature1['facebook'].append(facebook)
            feature1['payment'].append(payment)

            feature2 ={}

            feature2['bool_corporate_fl']= str(corporate)
            feature2['txt_country_fl']= str(cou_ind)
            feature2['num_reviews_fl']= reviews
            feature2['num_earnings_fl']= earnings
            feature2['bool_preferred_fl']= str(preferred)
            feature2['bool_verified_by_fl']= str(verified)
            feature2['bool_identity_verified_fl']= str(identity)
            feature2['bool_phone_verified_fl']= str(phone)
            feature2['bool_facebook_connected_fl']= str(facebook)
            feature2['bool_payment_verified_fl']= str(payment)
            feature2['skill'] = skill_fl1


            tf = open("myDictionary.json", "w")
            json.dump(feature2, tf)
            tf.close()


            tensor_fl_i = torch.zeros(1, len(skill))

            for value_fl in skill_fl1:
                index_fl_i = skill.index(value_fl)
                tensor_fl_i = tensor_fl_i + t_fl[index_fl_i]

            fl_train_skill_list1 = torch.cat((fl_train_skill_list, tensor_fl_i), dim=0)


            # for job in range(len(job_list)):
            torch.save(fl_train_skill_list1, 'fl_new.pth')

            file = open('./data/'+job+'.txt', 'r')
            job_set = []
            context = file.readlines()
            for i in context:
                job_set.append(int(i))
            file.close()

            # job_all =[]

            for job in job_set:

                freelancers_fal = []
                for j in range(fl_train_skill_list.size(0)):
                    a = torch.tensor(cl_train_skill_list[job], dtype=torch.int64)
                    b = torch.tensor(fl_train_skill_list[j], dtype=torch.int64)
                    c = a & b
                    pipei = torch.nonzero(c).size(0)
                    yuzhi = torch.nonzero(a).size(0)
                    if pipei >=2 :
                        freelancers_fal.append(j)
                if len(freelancers_fal)<100:
                    continue
                freelancers_fal.append(len(freelancer_list))

                jobs = np.ones(len(freelancers_fal)) * job

                freelancers = np.ones(len(freelancers_fal))*10
                # freelancers = freelancers_fal
                hang = num_users

                feature1['client'] = [list(bid_fl_pre[bid_fl_pre['id_job'] == int(jobs[0])]['num_client_reviews'])[
                                          0]] * hang

                feature1['client'] = np.array(feature1['client']).reshape(hang).tolist()

                feature1['paymentcl'] = [list(
                    bid_fl_pre[bid_fl_pre['id_job'] == int(jobs[0])]['bool_payment_verified_cl'])[
                                             0]] * hang

                feature1['paymentcl'] = np.array(feature1['paymentcl']).reshape(hang).tolist()

                feature1['profile'] = [list(
                    bid_fl_pre[bid_fl_pre['id_job'] == int(jobs[0])]['bool_profile_completed_cl'])[
                                           0]] * hang
                feature1['profile'] = np.array(feature1['paymentcl']).reshape(hang).tolist()
                feature1['phonecl'] = [list(bid_fl_pre[bid_fl_pre['id_job'] == int(jobs[0])]['bool_phone_verified_cl'])[
                                           0]] * hang
                feature1['phonecl'] = np.array(feature1['paymentcl']).reshape(hang).tolist()
                feature1['deposit'] = [list(bid_fl_pre[bid_fl_pre['id_job'] == int(jobs[0])]['bool_deposit_made_cl'])[
                                           0]] * hang
                feature1['deposit'] = np.array(feature1['paymentcl']).reshape(hang).tolist()
                for i in feature1.keys():
                    feature1[i] = torch.tensor(feature1[i], device=device, dtype=torch.int32)


                freelancers_true, freelancer_1, freelancer_2, index1,job_bool = test(model, jobs, freelancers_fal,freelancers, feature1, fl_train_skill_list1,
                                                             cl_train_skill_list, t_fl, device)


                if job_bool:
                    job_like1.append(int(job))
                    job_free[job]=[freelancer_1,freelancer_2]
                    # st.write("job", job)
                    # st.write("freelancer_1", freelancer_1)
                    # st.write("freelancer_2", freelancer_2[0])
                    np.save('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/a.npy', job_like1)
                if len(job_like1)>=3:
                    st.write('根据您刚才提供的信息，我们的推荐系统为您推荐三个工作机会,请进入下一页面查看。')
                    break
            if len(job_like1) <3:
                st.write('根据您刚才提供的信息，无法为您提供合适的工作,请重新填写。')
                st.stop()
            # freelancer_job1 = freelancer_job
            with open('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/json_test.txt', 'w+') as f:
                json.dump( job_free, f)

            st.session_state['test_demo'] =1





            # print("123", freelancer_job1)
        # job_like2 = tuple(job_like1)

def job_smallskill():
    if 'test_demo'not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'skill' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_info, bid_fl_pre, cou_index, money = load_data()
    num_list = []
    # bid_fl_pre = bid_fl_pre.iloc[:,:-2]
    # bid_fl_pre = bid_fl_pre[bid_fl_pre['bool_awarded'] == 1].drop_duplicates( )
    bid_fl_pre, freelancer_list, job_list, skill_cl, skill_fl, skill_fl_temp = reindex(bid_fl_pre)
    money = reindex1(money)
    skill = skill_deal1.skill_list(skill_fl, skill_cl)

    # print("skill",len(skill))
    bid_fl_pre = bid_fl_pre.reset_index()

    model = torch.load('cpu_50_pos.pth')
    t_fl = torch.load('t_fl.pth')

    cl_train_skill_list = torch.load('cl.pth')

    st.write('根据您刚才提供的信息，我们的推荐系统为您找到了以下三个工作机会。请仔细查看这些工作机会的职位、工作描述和要求，然后选择您最感兴趣的一个。')
    job_like1 = np.load('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/a.npy')
    job_like1 = job_like1.tolist()
    job_l = []
    job_final = []
    num = 1
    for j in job_like1:
        job_l.append(num)
        job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['list_skill_cl'])[0])
        job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['txt_title_cl'])[0])
        job_l.append(list(money[money['id_job'] == int(j)]['txt_budget'])[0])
        num = num + 1
        # job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['txt_description_cl'])[0])
        job_final.append(job_l)
        job_l = []
    # print(job_final)
    df = pd.DataFrame(
        job_final,
        columns=('job_id', 'skill', 'title', 'money'))
    st.table(df)
    with st.form("my_form1"):

        fl_train_skill_list = torch.load('fl_new.pth')
        job_like2 = st.selectbox(
            'jobs',
            (1,2,3))

        # Every form must have a submit button.

        submitted2 = st.form_submit_button("获取技能")
        tf = open("myDictionary.json", "r")
        feature2 = json.load(tf)
        # print("feature2",feature2)
        b = np.load('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/a.npy')
        b = b.tolist()
        job_like = b[job_like2 - 1]
        with open('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/json_test.txt', 'r+') as f:
            free_job = json.load(f)
        #     print("free_job", free_job)
        # print("job_like", job_like)

        freelancer_1 = free_job[str(job_like)][0]
        freelancer_2 = free_job[str(job_like)][1]
        job_l1 = []

        job_l1.append(job_like2)
        job_l1.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(job_like)]['list_skill_cl'])[0])
        job_l1.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(job_like)]['txt_title_cl'])[0])
        job_l1.append(list(money[money['id_job'] == int(job_like)]['txt_budget'])[0])
        # print(job_l1[0])
        df1 = pd.DataFrame(
            [job_l1],
            columns=('job_id', 'skill', 'title', 'money'))

        st.table(df1)
        if submitted2:

            for lancer in freelancer_2:
                list_skill = dp_hackpack(model, job_like, lancer, freelancer_1, t_fl, fl_train_skill_list,
                                      cl_train_skill_list, skill, feature2,device,feature_info,bid_fl_pre)
                if 'skill_len' in st.session_state:
                    st.warning('您已经选择过最喜欢的工作')
                    st.stop()
                st.session_state['skill_len'] = len(list_skill)

                for i in range(len(list_skill)):
                    st.session_state['skill' + str(i)] = list_skill[i]
                    st.write('{}({}/892)'.format(list_skill[i][0], list_skill[i][1]))
            st.session_state['skill'] = 1

def job_bigskill():

    if 'test_demo' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'skill' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_info, bid_fl_pre, cou_index, money = load_data()
    num_list = []
    # bid_fl_pre = bid_fl_pre.iloc[:,:-2]
    # bid_fl_pre = bid_fl_pre[bid_fl_pre['bool_awarded'] == 1].drop_duplicates( )
    bid_fl_pre, freelancer_list, job_list, skill_cl, skill_fl, skill_fl_temp = reindex(bid_fl_pre)
    money = reindex1(money)
    skill = skill_deal1.skill_list(skill_fl, skill_cl)

    # print("skill",len(skill))
    bid_fl_pre = bid_fl_pre.reset_index()

    model = torch.load('cpu_50_pos.pth')
    t_fl = torch.load('t_fl.pth')

    cl_train_skill_list = torch.load('cl.pth')

    st.write('根据您刚才提供的信息，我们的推荐系统为您找到了以下三个工作机会。请仔细查看这些工作机会的职位、工作描述和要求，然后选择您最感兴趣的一个。')
    job_like1 = np.load('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/a.npy')
    job_like1 = job_like1.tolist()
    job_l = []
    job_final = []
    num = 1
    for j in job_like1:
        job_l.append(num)
        job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['list_skill_cl'])[0])
        job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['txt_title_cl'])[0])
        job_l.append(list(money[money['id_job'] == int(j)]['txt_budget'])[0])
        num = num + 1
        # job_l.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(j)]['txt_description_cl'])[0])
        job_final.append(job_l)
        job_l = []
    # print(job_final)
    df = pd.DataFrame(
        job_final,
        columns=('job_id', 'skill', 'title', 'money'))
    st.table(df)
    with st.form("my_form1"):

        fl_train_skill_list = torch.load('fl_new.pth')
        job_like2 = st.selectbox(
            'jobs',
            (1,2,3))

        # Every form must have a submit button.
        submitted1 = st.form_submit_button("获取技能")

        tf = open("myDictionary.json", "r")
        feature2 = json.load(tf)
        # print("feature2",feature2)
        b = np.load('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/a.npy')
        b = b.tolist()
        job_like = b[job_like2 - 1]
        with open('./data/'+str(st.session_state['id'])+'_'+str(st.session_state['compe'])+'/json_test.txt', 'r+') as f:
            free_job = json.load(f)
        #     print("free_job", free_job)
        # print("job_like", job_like)

        freelancer_1 = free_job[str(job_like)][0]
        freelancer_2 = free_job[str(job_like)][1]
        job_l1 = []

        job_l1.append(job_like2)
        job_l1.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(job_like)]['list_skill_cl'])[0])
        job_l1.append(list(bid_fl_pre[bid_fl_pre['id_job'] == int(job_like)]['txt_title_cl'])[0])
        job_l1.append(list(money[money['id_job'] == int(job_like)]['txt_budget'])[0])
        # print(job_l1[0])
        df1 = pd.DataFrame(
            [job_l1],
            columns=('job_id', 'skill', 'title', 'money'))

        st.table(df1)
        if submitted1:
            print("tijiao")
            for lancer in freelancer_2:
                list_skill = hackpack(model, job_like, lancer, freelancer_1, t_fl, fl_train_skill_list, cl_train_skill_list,skill,feature2,device,feature_info,bid_fl_pre)
                if 'skill_len' in st.session_state:
                    st.warning('您已经选择过最喜欢的工作')
                    st.stop()
                st.session_state['skill_len'] = len(list_skill)
                st.write('**为了获得您喜欢的工作，您需要获得以下技能:**')
                for i in range(len(list_skill)):
                    st.session_state['skill'+str(i)] = list_skill[i]
                    st.write('{}({}/892)'.format(list_skill[i][0],list_skill[i][1]))
            st.session_state['skill'] = 1



def survey1_demo():

    import streamlit as st
    import pymysql
    # ---------连接--------------
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'survey1_demo' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()

    with st.form("my_form1"):
        pro1 = st.radio(
        "1.在上一页，您看到了三个工作机会。您对推荐系统推荐的这组工作是否满意？",
        ('1-非常不满意', '2', '3','4', '5', '6','7-非常满意'))

        pro2= st.radio(
        "2.您认为推荐系统为您推荐的这组工作适合您现有的技能和过往的工作经验吗？",
        ('1-非常不相关', '2', '3','4', '5', '6','7-非常相关'))

        pro3 = st.radio(
        "3.您认为推荐系统为您推荐的这组工作机会有多真实？",
        ('1-非常不实际', '2', '3','4', '5', '6','7-非常实际'))

        pro4 = st.radio(
            "4.您有多大可能会把这个推荐系统推荐给您身边正在努力寻找工作的人?",
            ('1-根本不可能', '2', '3', '4', '5', '6', '7-非常可能'))

        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['pro1'] = pro1
            st.session_state['pro2'] = pro2
            st.session_state['pro3'] = pro3
            st.session_state['pro4'] = pro4
            st.session_state['survey1_demo'] = 1
            st.write('提交完成')

def skill_nocompe():

    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'compe1' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    # ---------连接--------------
    st.write('在我们推荐系统推荐的工作机会中，您选择了其中您最感兴趣的一个。请仔细查看这个工作机会的相关内容并回答下面的问题。')


    with st.form("my_form1"):
        pro5 = st.radio(
        "5.如果您今天申请这份工作（或类似的工作），您认为您能应聘成功的可能性有多大？",
        ('1-非常不可能', '2', '3','4', '5', '6','7-非常可能'))

        pro6 = st.radio(
        "6.如果您今天申请这份工作（或类似的工作），您有多大信心能应聘成功？",
        ('1-非常没有自信', '2', '3','4', '5', '6','7-非常有自信'))

        pro7 = st.radio(
            "7.如果您需要花费一些时间学习新技能以得到这份工作（或类似的工作），那么您近期愿意投入多少时间来掌握这些技能？",
            ('20小时', '40小时', '80小时', '160小时', '320小时'))

        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['pro5'] = pro5
            st.session_state['pro6'] = pro6
            st.session_state['pro7'] = pro7
            st.session_state['compe1'] = 1
            st.write('提交完成')

def skill_compe():
    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'compe1' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    st.write('根据您目前的求职信息，我们的推荐系统发现您的竞争力在潜在求职者中竟然未能跻身前10%。')
    with st.form("my_form1"):
        pro5 = st.radio(
        "5.如果您今天申请这份工作（或类似的工作），您认为您能应聘成功的可能性有多大？",
        ('1-非常不可能', '2', '3','4', '5', '6','7-非常可能'))

        pro6 = st.radio(
        "6.如果您今天申请这份工作（或类似的工作），您有多大信心能应聘成功？",
        ('1-非常不相关', '2', '3','4', '5', '6','7-非常相关'))



        pro7 = st.radio(
            "7.如果您需要花费一些时间学习新技能以得到这份工作（或类似的工作），那么您近期愿意投入多少时间来掌握这些技能？",
            ('20小时', '40小时', '80小时', '160小时', '320小时'))

        submitted2 = st.form_submit_button("提交")

        if submitted2:
            st.session_state['pro5'] = pro5
            st.session_state['pro6'] = pro6
            st.session_state['pro7'] = pro7
            st.session_state['compe1'] = 1
            st.write('提交完成')

def skill_demo():


    import streamlit as st

    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'skill_demo' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()

    # ---------连接--------------
    st.markdown(
        """
        掌握新技能可能会增加您获得这份工作（或类似工作）的机会。为了获得这份工作，您可以在以下两种学习技能的策略中选择一种：

        **策略A：**
        您可以力图掌握少量学习成本较高的技能。这些技能中的每一项都将大大提高您获得这份工作的机会。然而，您需要付出巨大的努力才能掌握这些技能。

        **策略B:**
        您可以力图掌握多个学习成本较低的技能。这些技能中的每一项都只会稍微增加您获得这份工作的机会。然而，您只需要付出极少的努力就能掌握这些技能。
    """
    )


    with st.form("my_form1"):
        pro8 = st.radio(
            "8.您会选择哪种策略来学习技能？",
            ('1–必定会选择策略A',	'2'	,'3'	,'4','	5'	,'6',	'7–必定会选择策略B'))

        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['pro8'] = pro8
            st.session_state['skill_demo'] = 1
            st.write('提交完成')

def skill_nocompe1():
    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state or 'skill_demo' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'compe2' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    st.write('根据您目前的求职信息，我们的推荐系统发现您需要掌握以下技能，才能显著提高获得这份工作（或类似工作）的机会。')
    for i in range(int(st.session_state['skill_len'])):
        st.write('{}({}/892)'.format(st.session_state['skill' + str(i)][0], st.session_state['skill' + str(i)][1]))

    with st.form("my_form1"):
        pro9 = st.radio(
            "9.根据我们推荐系统的建议，您近期愿意投入多少时间来掌握这些技能？",
            ('20小时', '40小时', '80小时', '160小时', '320小时'))
        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['pro9'] = pro9
            st.session_state['compe2'] = 1
            st.write('提交完成')

def skill_compe1():
    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state or 'skill_demo' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'compe2' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()

    st.write('根据您目前的求职信息，我们的推荐系统发现你在众多潜在求职者中，甚至还没有跻身前10%。你需要掌握以下技能，才能显著提高获得工作（或类似工作）的机会。')
    for i in range(int(st.session_state['skill_len'])):
        st.write('{}({}/892)'.format(st.session_state['skill' + str(i)][0], st.session_state['skill' + str(i)][1]))

    with st.form("my_form1"):
        pro9 = st.radio(
            "9.根据我们推荐系统的建议，您近期愿意投入多少时间来掌握这些技能？",
            ('20小时', '40小时', '80小时', '160小时', '320小时'))
        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['pro9'] = pro9
            st.session_state['compe2'] = 1
            st.write('提交完成')
def skill_demo1():
    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state or 'skill_demo' not in st.session_state or 'compe2' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'skill_demo1' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    num = 0
    with st.form("my_form1"):
        st.write("10.您在上一页面表明，您愿意投入{}学习新技能来掌握新技能以增加获得这份工作（或类似工作）的机会。对于每项新技能，您近期愿意分配{}的百分之多少在上面？".format(st.session_state['pro9'],st.session_state['pro9']))
        for i in range(1, int(st.session_state['skill_len']) + 1):
            st.session_state['pro' + '10_'+str(i)] = st.number_input("技能" + str(i) + st.session_state['skill' + str(i-1)][0],max_value=1.0,min_value=0.0,step = 0.001)
            num = num + st.session_state['pro' +'10_'+str(i)]


        st.write("11.根据您现有的技能和过往的工作经验，您认为以下技能有多适合您？")
        for i in range(1,int(st.session_state['skill_len'])+1):
            st.session_state['pro'+'11_'+str(i)] = st.radio("技能"+str(i)+st.session_state['skill'+str(i-1)][0],('1-非常不适合', '2', '3','4', '5', '6','7-非常适合'))

        st.write("12.您认为以下技能对您获得您喜欢的工作有多大用处？")
        for i in range(1, int(st.session_state['skill_len']) + 1):
            st.session_state['pro' + '12_'+str(i)] = st.radio("技能" + str(i) + st.session_state['skill' + str(i-1)][0],
                                                             ('1非常没有用', '2', '3', '4', '5', '6', '7-非常有用'))

        st.write("13.如果您要申请这份工作（或类似的工作），您认为您应该掌握以下技能的可能性有多大？")
        for i in range(1, int(st.session_state['skill_len']) + 1):
            st.session_state['pro' + '13_'+str(i)] = st.radio(
                "技能" + str(i) + st.session_state['skill' + str(i-1)][0],
                ('1非常不可能', '2', '3', '4', '5', '6', '7-非常可能'))


        submitted2 = st.form_submit_button("提交")

        if submitted2:
            if num > 1:
                st.warning('数值加和超过100%，请重新填写')
            else:
                st.session_state['skill_demo1'] = 1
                st.write('提交完成')

def inform():
    import streamlit as st
    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state or 'skill_demo' not in st.session_state or 'compe2' not in st.session_state or 'skill_demo1' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'inform' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    import pymysql
    # ---------连接--------------
    st.markdown(
        """
      感谢您向我们提供您的意见。

        接下来，我们希望您提供一些关于您自己的基本信息。

    """
    )

    with st.form("my_form1"):
        st.session_state['pro14'] = st.radio(
            "性别：",
            ('男', '女'))

        st.session_state['pro15'] = st.selectbox("出生年份：",range(1970,2005))
        st.session_state['pro16' ] = st.radio(
            "您获得的最高教育水平是什么？",
            ('专业教育', '高中以下','高中', '大学未毕业','2年制大学学位（助理）','四年制大学学位（文学学士、理学学士）', '硕士学位', '博士学位'))
        st.session_state['pro17'] = st.radio(
            "就业状况？",
            ('被雇佣', '个体经营', '失业且求职中', '失业且没有求职', '家务操持者', '学生', '退休', '无法工作'))
        submitted2 = st.form_submit_button("提交")
        if submitted2:
            st.session_state['inform'] = 1
            st.write('提交完成')
# import mysql.connector
import pymysql
# @st.experimental_singleton
# def init_connection():
#     return mysql.connector.connect(**st.secrets["mysql"])

# conn = init_connection()

# @st.experimental_memo(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()

def inform1():
    import streamlit as st

    if 'test_demo' not in st.session_state or 'skill' not in st.session_state or 'survey1_demo' not in st.session_state or 'compe1' not in st.session_state or 'skill_demo' not in st.session_state or 'compe2' not in st.session_state or 'skill_demo1' not in st.session_state  or 'inform' not in st.session_state:
        st.warning('您之前的内容还未填写')
        st.stop()
    if 'inform1' in st.session_state:
        st.warning('本页内容您已经填写过')
        st.stop()
    # ---------连接--------------
    st.markdown(
        """
     请在您喜欢的选项周围画一个圈，说明您同意或不同意以下陈述的程度。针对每个问题请勿花过长时间思考；请根据您的第一感觉做出选择。
    """
    )

    with st.form("my_form1"):
        st.session_state['pro18'] = st.radio(
            "安全第一",
            ('1–完全不同意', '2','3','4','5','6','7','8',	'9–完全同意'))
        st.session_state['pro19']= st.radio(
            "我不会用我的健康冒险。",
            ('1–完全不同意', '2','3','4','5','6','7','8',	'9–完全同意'))
        st.session_state['pro20'] = st.radio(
            "我倾向于规避风险。",
            ('1–完全不同意', '2','3','4','5','6','7','8',	'9–完全同意'))
        st.session_state['pro21'] = st.radio(
            "我经常冒险。",
            ('1–完全不同意', '2', '3', '4', '5', '6', '7', '8', '9–完全同意'))
        st.session_state['pro22'] = st.radio(
            "我讨厌“不知道将要发生什么”的感觉。",
            ('1–完全不同意', '2', '3', '4', '5', '6', '7', '8', '9–完全同意'))
        st.session_state['pro23' ] = st.radio(
            "我通常认为风险是一种挑战。",
            ('1–完全不同意', '2', '3', '4', '5', '6', '7', '8', '9–完全同意'))
        st.session_state['pro24'] = st.radio(
            "我认为自己是一个 . . .",
            ('1–风险规避者', '2', '3', '4', '5', '6', '7', '8', '9–风险寻求者'))

        submitted2 = st.form_submit_button("提交")
        if submitted2:
            dict1 = {}
            txt =''


            for i in st.session_state.keys():
                dict1[i] = st.session_state[i]
            dict1.pop('FormSubmitter:my_form1-提交')
            print("dict1", dict1)
            count = 1

            for i in dict1.keys():

                if count == len(dict1):
                    txt = txt + i + ',' + str(st.session_state[i]).strip('[]')
                else:
                    txt = txt + i + ',' + str(st.session_state[i]).strip('[]').replace('\'','')  +','
                    count = count +1
            # print("keys",keys)

            connect = pymysql.connect(**st.secrets["mysql"])  # 服务器名,账户,密码，数据库名称
            cur = connect.cursor()

            try:
                create_sqli = "create table survey (id int auto_increment PRIMARY KEY, answer longtext);"
                cur.execute(create_sqli)
            except Exception as e:
                print("创建数据表失败:", e)
            else:
                print("创建数据表成功;")

            # ---------------插入---------
            try:

                insert_sqli = "insert into survey (answer) values('{}');".format(txt)

                cur.execute(insert_sqli)
            except Exception as e:
                print("插入数据失败:", e)
            else:
                # 如果是插入数据， 一定要提交数据， 不然数据库中找不到要插入的数据;
                connect.commit()
                print("插入数据成功;")

            # try:
            #     create_sqli = "create table survey ( id int auto_increment PRIMARY KEY, answer longtext);"
            #     run_query(create_sqli)
            # except Exception as e:
            #     print("创建数据表失败:", e)
            # else:
            #     print("创建数据表成功;")
            #
            #     # ---------------插入---------
            # try:
            #
            #     # insert_sqli = "insert into survey (answer) values({keys1});".format(keys1="abc")
            #     insert_sqli = "insert into survey (answer) values('{}');".format(txt)
            #     print(insert_sqli)
            #
            #     run_query(insert_sqli)
            # except Exception as e:
            #     print("插入数据失败:", e)
            # else:
            #     # 如果是插入数据， 一定要提交数据， 不然数据库中找不到要插入的数据;
            #
            #     conn.commit()
            #     print("插入数据成功;")
            with open('./data/' + str(st.session_state['id']) + '_' + str(st.session_state['compe']) + '/result.json',
                      'w+') as f:
                json.dump(dict1, f, ensure_ascii=False)
            st.write('提交完成')

# 竞争可知，考虑成本
if index ==1:
    page_names_to_funcs = {
    "研究知情书": intro,
    "研究背景": background,
    "技能等信息填写": test_demo,
    "推荐工作":job_smallskill,
    "问题模块1": survey1_demo,
    "问题模块2": skill_compe,
    "问题模块3":skill_demo,
    "问题模块4":skill_compe1,
    "问题模块5":skill_demo1,
    "基本信息6":inform,
    "基本信息7":inform1
    }
    demo_name = st.sidebar.radio("模块选择", page_names_to_funcs.keys())
    # demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()

# 竞争力可知，不考虑成本
if index ==2:
    page_names_to_funcs = {
    "研究知情书": intro,
    "研究背景": background,
    "技能等信息填写": test_demo,
    "推荐工作":job_bigskill,
    "问题模块1": survey1_demo,
    "问题模块2": skill_compe,
    "问题模块3": skill_demo,
    "问题模块4": skill_compe1,
    "问题模块5": skill_demo1,
    "基本信息6": inform,
    "基本信息7": inform1
    }
    demo_name = st.sidebar.radio("模块选择", page_names_to_funcs.keys())
    # demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()

 # 竞争力不可知，不考虑成本
if index ==3:
    page_names_to_funcs = {
    "研究知情书": intro,
    "研究背景": background,
    "技能等信息填写": test_demo,
    "推荐工作":job_bigskill,
    "问题模块1": survey1_demo,
    "问题模块2": skill_nocompe,
    "问题模块3":skill_demo,
    "问题模块4":skill_nocompe1,
    "问题模块5":skill_demo1,
    "基本信息6":inform,
    "基本信息7":inform1
    }
    demo_name = st.sidebar.radio("模块选择", page_names_to_funcs.keys())
    # demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
 # 竞争力不可知，考虑成本
if index ==4:
    page_names_to_funcs = {
    "研究知情书": intro,
    "研究背景": background,
    "技能等信息填写": test_demo,
    "推荐工作":job_smallskill,
    "问题模块1": survey1_demo,
    "问题模块2": skill_nocompe,
    "问题模块3":skill_demo,
    "问题模块4":skill_nocompe1,
    "问题模块5":skill_demo1,
    "基本信息6":inform,
    "基本信息7":inform1
    }
    demo_name = st.sidebar.radio("模块选择", page_names_to_funcs.keys())
    # demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()




