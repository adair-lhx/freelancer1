import numpy as np
import re
import torch
import pandas as pd
def stack_string(dataframe_cols,clos_name):
    stack_string=""

    for i in range(len(dataframe_cols)):
        stack_string += str(dataframe_cols[clos_name][i])

    return stack_string

def skill_token(fl_cl,str_data):
    if fl_cl=="fl":
        skill_token_fl = re.findall(r"'([a-zA-Z0-9/\x20/]+)',",str_data)
        return skill_token_fl
    elif fl_cl == "cl":
        skill_token_cl = re.findall(r"'([a-zA-Z0-9/\x20/]+)',",str_data)

        return skill_token_cl
def skill_token1(fl_cl,str_data):
    if fl_cl == "fl":
        skill_token_fl = re.sub("[0-9\!\%\[\]\。\(\)\'\-\_\ ]", "", str_data)
        skill_token_fl = skill_token_fl.split(',')

        skill_token_fl = list(set(skill_token_fl))

        if '' in skill_token_fl:
            skill_token_fl.remove('')
        return skill_token_fl

    if fl_cl == "cl":
        skill_token_cl = re.findall(r"'([a-zA-Z0-9/\x20/]+)',",str_data)
        # print(skill_token_cl)
        #
        # skill_token_cl = skill_token_cl.split(',')

        skill_token_cl = list(set(skill_token_cl))
        if '' in skill_token_cl:
            skill_token_cl.remove('')
        return skill_token_cl

def skill_dict(all_data):

    fl_skill = all_data[["id_fl", "list_skill_fl"]]
    skill_fl_str = stack_string(fl_skill, "list_skill_fl")

    # list_fl_skill = list(set(skill_token("fl", skill_fl_str)))
    list_fl_skill = skill_token1( 'fl',skill_fl_str)

    cl_skill = all_data[["id_job", "list_skill_cl"]]
    skill_cl_str = stack_string(cl_skill, "list_skill_cl")
    list_cl_skill = skill_token1( 'cl',skill_cl_str)

    return list_fl_skill, list_cl_skill
def skill_dict1(list_fl_skill,list_cl_skill):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    skill_fl =[]
    skill_cl = []
    for i in list_fl_skill.keys():
        for j in list_fl_skill[i]:
            if j not in skill_fl:
                skill_fl.append(j)


    for i in list_cl_skill.keys():
        for j in list_cl_skill[i]:
            if j not in skill_cl:
                skill_cl.append(j)


    return skill_fl,skill_cl

def skill_list(list_fl_skill,list_cl_skill):
    list_fl_skill, list_cl_skill = skill_dict1(list_fl_skill,list_cl_skill)

    list_skill = list_fl_skill
    for i in list_cl_skill:
        if i not in list_skill:
            list_skill.append(i)
    return  list_skill

def skill_to_tensor3(all_data, skill_cl, skill_fl,skill_fl1):
    # list_fl_skill, list_cl_skill = skill_dict(all_data)
    list_fl_skill, list_cl_skill = skill_dict1(skill_fl,skill_cl)


    list_skill = list_fl_skill
    for i in list_cl_skill:
        if i not in list_skill:
            list_skill.append(i)
    # print("skill_after******",list_skill)
    t_fl, t_cl = torch.eye(len(list_skill)), torch.eye(len(list_skill))


    fl_tensors, cl_tensors = torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)


    for i in skill_fl1.keys():
        skill_fl2 = skill_fl1[i]
        tensor_fl_i = torch.zeros(1, len(list_skill))  # 创建初始tensor
        for value_fl in skill_fl2:


            index_fl_i = list_skill.index(value_fl)  # skill对应的向量

            tensor_fl_i = tensor_fl_i + t_fl[index_fl_i]
        fl_tensors = torch.cat((fl_tensors, tensor_fl_i), dim=0)

    for i in skill_cl.keys():
        skill_cl1 =  skill_cl[i]
        tensor_cl_i = torch.zeros(1, len(list_skill))
        for value_cl in skill_cl1:
            index_cl_i = list_skill.index(value_cl)

            tensor_cl_i = tensor_cl_i + t_cl[index_cl_i]
        cl_tensors = torch.cat((cl_tensors, tensor_cl_i), dim=0)



    return t_fl, list_skill,fl_tensors, cl_tensors

def skill_to_tensor2(all_data, skill_cl, skill_fl):
    # list_fl_skill, list_cl_skill = skill_dict(all_data)

    list_fl_skill, list_cl_skill = skill_dict1(skill_fl,skill_cl)


    list_skill = list_fl_skill
    for i in list_cl_skill:
        if i not in list_skill:
            list_skill.append(i)
    print("skill_after******",list_skill)

    t_fl, t_cl = torch.eye(len(list_skill)), torch.eye(len(list_skill))
    print("t_fl",t_fl.size())

    fl_tensors, cl_tensors = torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)


    for i in skill_fl.keys():

        skill_fl1 = skill_fl[i]
        tensor_fl_i = torch.zeros(1, len(list_skill))  # 创建初始tensor
        for value_fl in skill_fl1:

            index_fl_i = list_skill.index(value_fl)  # skill对应的向量
            tensor_fl_i = tensor_fl_i + t_fl[index_fl_i]


        fl_tensors = torch.cat((fl_tensors, tensor_fl_i), dim=0)

    for i in skill_cl.keys():
        skill_cl1 =  skill_cl[i]
        tensor_cl_i = torch.zeros(1, len(list_skill))

        for value_cl in skill_cl1:
            index_cl_i = list_skill.index(value_cl)
            tensor_cl_i = tensor_cl_i + t_cl[index_cl_i]
        cl_tensors = torch.cat((cl_tensors, tensor_cl_i), dim=0)
    num_nodes = len(list_skill)
    graph = np.zeros((num_nodes, num_nodes))

    for i in skill_cl.keys():
        index = []
        skill_cl1 = skill_cl[i]
        for value_cl in skill_cl1:
            index_cl_i = list_skill.index(value_cl)
            index.append(index_cl_i)
        for j in index:
            for q in index:
                if j != q:
                    graph[j, q] = 1
                    graph[q, j] = 1

    print("graph_non", len(graph.nonzero()[0]))

    return t_fl, t_cl,fl_tensors, cl_tensors,graph.nonzero()

def skill_to_tensor1(all_data,skill_cl,skill_fl):
    list_fl_skill, list_cl_skill, list_fl_cer= skill_dict(all_data)
    t_fl, t_cl = torch.eye(len(list_fl_skill)), np.eye(len(list_cl_skill))
    print(list_fl_skill)
    print(list_cl_skill)

    t_cer = torch.eye(len(list_fl_cer))

    fl_tensors, cl_tensors = torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)
    cer_tensors = torch.tensor([], dtype=torch.float)

    for i in skill_fl.keys():
        skill_fl1 = skill_token("fl", skill_fl[i])
        tensor_fl_i = torch.zeros(1, len(list_fl_skill))  # 创建初始tensor
        for value_fl in skill_fl1:
            index_fl_i = list_fl_skill.index(value_fl)  # skill对应的向量
            tensor_fl_i = tensor_fl_i + t_fl[index_fl_i]
        fl_tensors = torch.cat((fl_tensors, tensor_fl_i), dim=0)
    for i in skill_cl.keys():
        skill_cl1 = skill_token("cl", skill_cl[i])
        tensor_cl_i = torch.zeros(1, len(list_cl_skill))
        for value_cl in skill_cl1:
            index_cl_i = list_cl_skill.index(value_cl)
            tensor_cl_i = tensor_cl_i + t_cl[index_cl_i]
        cl_tensors = torch.cat((cl_tensors, tensor_cl_i), dim=0)

    return fl_tensors, cl_tensors

def skill_to_tensor(skill_cl, skill_fl):
    list_fl_skill, list_cl_skill = skill_dict1(skill_fl, skill_cl)

    list_skill = list_fl_skill
    for i in list_cl_skill:
        if i not in list_skill:
            list_skill.append(i)





    return list_skill

def skill_to_tensor4(skill_cl, skill_fl):
    # list_fl_skill, list_cl_skill = skill_dict(all_data)
    list_fl_skill, list_cl_skill = skill_dict1(skill_fl,skill_cl)


    list_skill = list_fl_skill
    for i in list_cl_skill:
        if i not in list_skill:
            list_skill.append(i)
    print("skill_after******",list_skill)

    '''crete graph'''
    print("create_graph")
    num_nodes = len(list_skill)
    graph = np.zeros((num_nodes, num_nodes))
    graph1 = np.zeros((num_nodes, num_nodes))
    for i in skill_cl.keys():
        index = []
        skill_cl1 =  skill_cl[i]
        for value_cl in skill_cl1:
            index_cl_i = list_skill.index(value_cl)
            index.append(index_cl_i)
        for j in index:
            for q in index:
                if j !=q:
                    graph[j,q] = 1
                    graph[q,j] = 1


    count = 0
    count1 = 0
    for i in skill_fl.keys():
        index = []
        skill_fl2 = skill_fl[i]
        for value_fl in skill_fl2:
            index_fl_i = list_skill.index(value_fl)  # skill对应的向量
            index.append(index_fl_i)
        for j in index:
            for q in index:
                if j !=q:
                    graph1[j,q] = graph1[j,q] +0.2
                    if graph1[j,q] >= 1:
                        if graph[j,q] ==  1:
                            count1 = count1 +1
                        graph[j,q] = 1
                        count = count +1
    print("count",count)
    print("count", count1)
    number = 0
    for i in range(185):
        if i in graph.nonzero()[0]:
            number =number +1
    print("number",number)
    print("graph_non", len(graph.nonzero()[0]))

    return graph
if __name__ == "__main__":
    path = "./data/bid_fl_job_review_pre1.csv"
    data = pd.read_csv(path)
    list_fl_skill, list_cl_skill = skill_dict(data)
    # print(list_fl_skill, list_cl_skill)
    fl_tensors,cl_tensors = skill_to_tensor(data,data.head(2))
    print(fl_tensors,cl_tensors)