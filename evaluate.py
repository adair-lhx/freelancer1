import numpy as np
import torch
import skill_deal1


def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0
# def precision(user, pred_items,bid_fl_pre):
# 	df = bid_fl_pre[bid_fl_pre['id_job']==int(user[0])]
# 	df1=list(df['id_fl'])
# 	count = 0
# 	for item in df1:
# 		if item in pred_items:
# 			count = count +1
# 	return count/len(pred_items)
def precision(ng_item, pred_items,topk):
	if ng_item in pred_items:
		return 1/topk
	return 0

def mrr(ng_item, pred_items):
	for index,i in enumerate(pred_items):
		if ng_item == i:
			return 1/(index+1)
	return 0

def map(ng_item, pred_items,topk):
	for index,i in enumerate(pred_items):
		if ng_item == i:
			return 1/(index+1)/topk
	return 0
def recall(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0
# def recall(user,pred_items,bid_fl_pre):
# 	df = bid_fl_pre[bid_fl_pre['id_job'] == int(user[0])]
# 	df1 = list(df['id_fl'])
# 	count = 0
# 	for item in df1:
# 		if item in pred_items:
# 			count = count + 1
# 	return count / len(df1)

def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []
	pre,rea = [],[]
	MRR,MAP =[],[]
	for user, item, label, corporate, country, reviews, earnings, preferred, verified, identity, payment, phone, facebook, client, paymentcl, profile, phonecl, deposit in test_loader:
		user = user.to(device)
		item = item.to(device)
		label = label.to(device)




		predictions = model(user, item,device)
		# print("user",user[0])
		# print("item",item)
		_, indices = torch.topk(predictions, top_k)
		tibu_value, tibu_indices = torch.topk(predictions, top_k+5)
		# print("value",_)
		# print("index",indices)
		# print("tibu_value",tibu_value)
		# print("tibu_indices",tibu_indices)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()
		# recommends_tibu = torch.take(
		# 	item,  tibu_indices).cpu().numpy().tolist()
		# print("recommends",recommends)
		# print("recommends_tibu",recommends_tibu)
		# print("id_job",list(bid_fl_pre[bid_fl_pre['id_job'] == int(user[0])]['list_skill_cl'])[0])
		# for fl in recommends_tibu:
		# 	print("fl_1", list(bid_fl_pre[bid_fl_pre['id_fl'] == int(fl)]['list_skill_fl'])[0])
		# if int(user[0])== 0:
		# 	exit()
		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		# print("ng_item",ng_item)
		pre.append(precision(ng_item, recommends,top_k))
		rea.append(recall(ng_item, recommends))
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))
		MRR.append(mrr(ng_item, recommends))
		MAP.append(map(ng_item, recommends,top_k))
	return np.mean(HR), np.mean(NDCG),np.mean(pre),np.mean(rea),np.mean(MRR),np.mean(MAP)

