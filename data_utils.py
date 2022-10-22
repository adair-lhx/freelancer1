import random
import numpy as np 
import pandas as pd 
import torch


class NCF_Data(object):
	"""
	Construct Dataset for NCF
	"""
	def __init__(self, args, ratings,feature_info):
		self.feature_info = feature_info
		print(self.feature_info)
		self.unchang_fea = self.feature_info['unchange_feas_fl']
		self.change_fea = self.feature_info['change_feas_fl']
		self.other_fea = self.feature_info['other_feas_cl']

		self.ratings = ratings
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size

		self.preprocess_ratings = self._reindex(self.ratings)

		self.user_pool = set(self.ratings['id_job'].unique())
		self.item_pool = set(self.ratings['id_fl'].unique())

		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
		self.negatives = self._negative_sampling(self.preprocess_ratings)
		random.seed(args.seed)
	
	def _reindex(self, ratings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(ratings['id_job'].drop_duplicates())

		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(ratings['id_fl'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		ratings['id_job'] = ratings['id_job'].apply(lambda x: user2id[x])
		ratings['id_fl'] = ratings['id_fl'].apply(lambda x: item2id[x])
		ratings['target_num_rating'] = ratings['target_num_rating'].apply(lambda x: float(x > 0))

		return ratings

	def _leave_one_out(self, ratings):
		"""
		leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
		"""
		ratings['rank_latest'] = ratings.groupby(['id_job'])['url'].rank(method='first', ascending=False)

		test = ratings.loc[ratings['rank_latest'] == 1]

		train = ratings.loc[ratings['rank_latest'] > 1]
		print("train",train['id_job'].nunique())
		# assert train['id_fl'].nunique()==test['id_fl'].nunique(), 'Not Match Train User with Test User'
		index = [ 'id_job','id_fl', 'target_num_rating']
		skill = ['list_skill_fl','list_skill_cl']
		bert_index = ['txt_title_fl','txt_description_fl','txt_title_cl','txt_description_cl']
		score = ['num_rating_fl','num_client_rating']
		for i in self.unchang_fea :
			index.append(i)
		for j in self.change_fea:
			index.append(j)
		for k in self.other_fea:
			index.append(k)
		for i in skill:
			index.append(i)
		for j in bert_index:
			index.append(j)
		for k in score:
			index.append(k)
		return train[index], test[index]



	def _negative_sampling(self, ratings):
		interact_status = (
			ratings.groupby('id_job')['id_fl']
			.apply(set)
			.reset_index()
			.rename(columns={'id_fl': 'interacted_items'}))

		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)

		interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))

		return interact_status[['id_job', 'negative_items', 'negative_samples']]

	def get_train_instance(self):
		users, items, ratings = [], [], []
		#unchang
		bool_corporate_fls,txt_country_fls,num_reviews_fls,num_earnings_fls = [],[],[],[]
		#chang_feature
		bool_preferred_fls,bool_verified_by_fls,bool_identity_verified_fls,bool_payment_verified_fls,bool_phone_verified_fls,bool_facebook_connected_fls =[],[],[],[],[],[]
		#other_feature
		num_client_reviewss,bool_payment_verified_cls,bool_profile_completed_cls,bool_phone_verified_cls,bool_deposit_made_cls = [],[],[],[],[]
		#skill
		list_skill_fls,list_skill_cls = [],[]
		#text
		txt_title_fls,txt_description_fls,txt_title_cls,txt_description_cls = [],[],[],[]
		#socre
		num_client_ratings,num_rating_fls = [],[]
		train_ratings = pd.merge(self.train_ratings, self.negatives[['id_job', 'negative_items']], on='id_job')
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))


		count = 0
		for row in train_ratings.itertuples():
			count = count +1

			users.append(int(row.id_job))
			items.append(int(row.id_fl))
			ratings.append(float(row.target_num_rating))
			# add unchange_fea
			bool_corporate_fls.append(int(row.bool_corporate_fl))
			txt_country_fls.append(int(row.txt_country_fl))
			num_reviews_fls.append(float(row.num_reviews_fl))
			num_earnings_fls.append(float(row.num_earnings_fl))

			# add change_fea
			bool_preferred_fls.append(int(row.bool_preferred_fl))
			bool_verified_by_fls.append(int(row.bool_verified_by_fl))
			bool_identity_verified_fls.append(int(row.bool_identity_verified_fl))
			bool_payment_verified_fls.append(int(row.bool_payment_verified_fl))
			bool_phone_verified_fls.append(int(row.bool_phone_verified_fl))
			bool_facebook_connected_fls.append(int(row.bool_facebook_connected_fl))

			#add other_fea
			num_client_reviewss.append(float(row.num_client_reviews))
			bool_payment_verified_cls.append(int(row.bool_payment_verified_cl))
			bool_profile_completed_cls.append(int(row.bool_profile_completed_cl))
			bool_phone_verified_cls.append(float(row.bool_phone_verified_cl))
			bool_deposit_made_cls.append(int(row.bool_deposit_made_cl))

			# #add skill
			# list_skill_fls.append(row.list_skill_fl)
			# list_skill_cls.append(row.list_skill_cl)
			# # add text
			# txt_title_fls.append(row.txt_title_fl)
			# txt_description_fls.append(row.txt_description_fl)
			# txt_title_cls.append(row.txt_title_cl)
			# txt_description_cls.append(row.txt_description_cl)


			for i in range(self.num_ng):
				users.append(int(row.id_job))
				items.append(int(row.negatives[i]))
				ratings.append(float(0))  # negative samples get 0 rating
				# add unchange_fea
				bool_corporate_fls.append(int(row.bool_corporate_fl))
				txt_country_fls.append(int(row.txt_country_fl))
				num_reviews_fls.append(float(row.num_reviews_fl))
				num_earnings_fls.append(float(row.num_earnings_fl))

				# add change_fea
				bool_preferred_fls.append(int(row.bool_preferred_fl))
				bool_verified_by_fls.append(int(row.bool_verified_by_fl))
				bool_identity_verified_fls.append(int(row.bool_identity_verified_fl))
				bool_payment_verified_fls.append(int(row.bool_payment_verified_fl))
				bool_phone_verified_fls.append(int(row.bool_phone_verified_fl))
				bool_facebook_connected_fls.append(int(row.bool_facebook_connected_fl))

				# add other_fea
				num_client_reviewss.append(float(row.num_client_reviews))
				bool_payment_verified_cls.append(int(row.bool_payment_verified_cl))
				bool_profile_completed_cls.append(int(row.bool_profile_completed_cl))
				bool_phone_verified_cls.append(float(row.bool_phone_verified_cl))
				bool_deposit_made_cls.append(int(row.bool_deposit_made_cl))
				# add skill
				# list_skill_fls.append(row.list_skill_fl)
				# list_skill_cls.append(row.list_skill_cl)
				# # add text
				# txt_title_fls.append(row.txt_title_fl)
				# txt_description_fls.append(row.txt_description_fl)
				# txt_title_cls.append(row.txt_title_cl)
				# txt_description_cls.append(row.txt_description_cl)



		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings,
			bool_corporate_fl_list =bool_corporate_fls,
			txt_country_fl_list=txt_country_fls,
			num_reviews_fl_list=num_reviews_fls,
			num_earnings_fl_list=num_earnings_fls,
			bool_preferred_fl_list = bool_preferred_fls,
			bool_verified_by_fl_list = bool_verified_by_fls,
			bool_identity_verified_fl_list =bool_identity_verified_fls,
			bool_payment_verified_fl_list = bool_payment_verified_fls,
			bool_phone_verified_fl_list = bool_phone_verified_fls,
			bool_facebook_connected_fl_list = bool_facebook_connected_fls,
			num_client_reviews_list =num_client_reviewss,
			bool_payment_verified_cl_list =bool_payment_verified_cls,
			bool_profile_completed_cl_list =bool_profile_completed_cls,
			bool_phone_verified_cl_list =bool_phone_verified_cls,
			bool_deposit_made_cl_list =bool_deposit_made_cls


		)
		return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

	def get_test_instance(self):
		users, items, ratings = [], [], []
		# unchang
		bool_corporate_fls, txt_country_fls, num_reviews_fls, num_earnings_fls = [], [], [], []
		# chang_feature
		bool_preferred_fls, bool_verified_by_fls, bool_identity_verified_fls, bool_payment_verified_fls, bool_phone_verified_fls, bool_facebook_connected_fls = [], [], [], [], [], []
		# other_feature
		num_client_reviewss, bool_payment_verified_cls, bool_profile_completed_cls, bool_phone_verified_cls, bool_deposit_made_cls = [], [], [], [], []
		#skill
		list_skill_fls,list_skill_cls=[],[]
		# text
		txt_title_fls, txt_description_fls, txt_title_cls, txt_description_cls = [], [], [], []
		# socre
		num_client_ratings, num_rating_fls = [], []

		test_ratings = pd.merge(self.test_ratings, self.negatives[['id_job', 'negative_samples']], on='id_job')
		for row in test_ratings.itertuples():
			users.append(int(row.id_job))
			items.append(int(row.id_fl))
			ratings.append(float(row.target_num_rating))
			# add unchange_fea
			bool_corporate_fls.append(int(row.bool_corporate_fl))
			txt_country_fls.append(int(row.txt_country_fl))
			num_reviews_fls.append(float(row.num_reviews_fl))
			num_earnings_fls.append(float(row.num_earnings_fl))

			# add change_fea
			bool_preferred_fls.append(int(row.bool_preferred_fl))
			bool_verified_by_fls.append(int(row.bool_verified_by_fl))
			bool_identity_verified_fls.append(int(row.bool_identity_verified_fl))
			bool_payment_verified_fls.append(int(row.bool_payment_verified_fl))
			bool_phone_verified_fls.append(int(row.bool_phone_verified_fl))
			bool_facebook_connected_fls.append(int(row.bool_facebook_connected_fl))

			# add other_fea
			num_client_reviewss.append(float(row.num_client_reviews))
			bool_payment_verified_cls.append(int(row.bool_payment_verified_cl))
			bool_profile_completed_cls.append(int(row.bool_profile_completed_cl))
			bool_phone_verified_cls.append(float(row.bool_phone_verified_cl))
			bool_deposit_made_cls.append(int(row.bool_deposit_made_cl))

			# # add skill
			# list_skill_fls.append(row.list_skill_fl)
			# list_skill_cls.append(row.list_skill_cl)
			# # add text
			# txt_title_fls.append(row.txt_title_fl)
			# txt_description_fls.append(row.txt_description_fl)
			# txt_title_cls.append(row.txt_title_cl)
			# txt_description_cls.append(row.txt_description_cl)



			for i in getattr(row, 'negative_samples'):
				users.append(int(row.id_job))
				items.append(int(i))
				ratings.append(float(0))
				# add unchange_fea
				bool_corporate_fls.append(int(row.bool_corporate_fl))
				txt_country_fls.append(int(row.txt_country_fl))
				num_reviews_fls.append(float(row.num_reviews_fl))
				num_earnings_fls.append(float(row.num_earnings_fl))

				# add change_fea
				bool_preferred_fls.append(int(row.bool_preferred_fl))
				bool_verified_by_fls.append(int(row.bool_verified_by_fl))
				bool_identity_verified_fls.append(int(row.bool_identity_verified_fl))
				bool_payment_verified_fls.append(int(row.bool_payment_verified_fl))
				bool_phone_verified_fls.append(int(row.bool_phone_verified_fl))
				bool_facebook_connected_fls.append(int(row.bool_facebook_connected_fl))

				# add other_fea
				num_client_reviewss.append(float(row.num_client_reviews))
				bool_payment_verified_cls.append(int(row.bool_payment_verified_cl))
				bool_profile_completed_cls.append(int(row.bool_profile_completed_cl))
				bool_phone_verified_cls.append(float(row.bool_phone_verified_cl))
				bool_deposit_made_cls.append(int(row.bool_deposit_made_cl))

				# add skill
				# list_skill_fls.append(row.list_skill_fl)
				# list_skill_cls.append(row.list_skill_cl)
				# # add text
				# txt_title_fls.append(row.txt_title_fl)
				# txt_description_fls.append(row.txt_description_fl)
				# txt_title_cls.append(row.txt_title_cl)
				# txt_description_cls.append(row.txt_description_cl)


		
		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings,
			bool_corporate_fl_list=bool_corporate_fls,
			txt_country_fl_list=txt_country_fls,
			num_reviews_fl_list=num_reviews_fls,
			num_earnings_fl_list=num_earnings_fls,
			bool_preferred_fl_list=bool_preferred_fls,
			bool_verified_by_fl_list=bool_verified_by_fls,
			bool_identity_verified_fl_list=bool_identity_verified_fls,
			bool_payment_verified_fl_list=bool_payment_verified_fls,
			bool_phone_verified_fl_list=bool_phone_verified_fls,
			bool_facebook_connected_fl_list=bool_facebook_connected_fls,
			num_client_reviews_list=num_client_reviewss,
			bool_payment_verified_cl_list=bool_payment_verified_cls,
			bool_profile_completed_cl_list=bool_profile_completed_cls,
			bool_phone_verified_cl_list=bool_phone_verified_cls,
			bool_deposit_made_cl_list=bool_deposit_made_cls

		)

		return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=4)


	def get_test(self,item_list):
		users, items, ratings = [], [], []
		# unchang
		bool_corporate_fls, txt_country_fls, num_reviews_fls, num_earnings_fls = [], [], [], []
		# chang_feature
		bool_preferred_fls, bool_verified_by_fls, bool_identity_verified_fls, bool_payment_verified_fls, bool_phone_verified_fls, bool_facebook_connected_fls = [], [], [], [], [], []
		# other_feature
		num_client_reviewss, bool_payment_verified_cls, bool_profile_completed_cls, bool_phone_verified_cls, bool_deposit_made_cls = [], [], [], [], []
		#skill
		list_skill_fls,list_skill_cls=[],[]
		# text
		txt_title_fls, txt_description_fls, txt_title_cls, txt_description_cls = [], [], [], []
		# socre
		num_client_ratings, num_rating_fls = [], []

		test_ratings = pd.merge(self.test_ratings, self.negatives[['id_job', 'negative_samples']], on='id_job')
		for row in test_ratings.itertuples():
			users.append(int(row.id_job))
			items.append(int(row.id_fl))
			ratings.append(float(row.target_num_rating))
			# add unchange_fea
			bool_corporate_fls.append(int(row.bool_corporate_fl))
			txt_country_fls.append(int(row.txt_country_fl))
			num_reviews_fls.append(float(row.num_reviews_fl))
			num_earnings_fls.append(float(row.num_earnings_fl))

			# add change_fea
			bool_preferred_fls.append(int(row.bool_preferred_fl))
			bool_verified_by_fls.append(int(row.bool_verified_by_fl))
			bool_identity_verified_fls.append(int(row.bool_identity_verified_fl))
			bool_payment_verified_fls.append(int(row.bool_payment_verified_fl))
			bool_phone_verified_fls.append(int(row.bool_phone_verified_fl))
			bool_facebook_connected_fls.append(int(row.bool_facebook_connected_fl))

			# add other_fea
			num_client_reviewss.append(float(row.num_client_reviews))
			bool_payment_verified_cls.append(int(row.bool_payment_verified_cl))
			bool_profile_completed_cls.append(int(row.bool_profile_completed_cl))
			bool_phone_verified_cls.append(float(row.bool_phone_verified_cl))
			bool_deposit_made_cls.append(int(row.bool_deposit_made_cl))

			# # add skill
			# list_skill_fls.append(row.list_skill_fl)
			# list_skill_cls.append(row.list_skill_cl)
			# # add text
			# txt_title_fls.append(row.txt_title_fl)
			# txt_description_fls.append(row.txt_description_fl)
			# txt_title_cls.append(row.txt_title_cl)
			# txt_description_cls.append(row.txt_description_cl)




		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings,
			bool_corporate_fl_list=bool_corporate_fls,
			txt_country_fl_list=txt_country_fls,
			num_reviews_fl_list=num_reviews_fls,
			num_earnings_fl_list=num_earnings_fls,
			bool_preferred_fl_list=bool_preferred_fls,
			bool_verified_by_fl_list=bool_verified_by_fls,
			bool_identity_verified_fl_list=bool_identity_verified_fls,
			bool_payment_verified_fl_list=bool_payment_verified_fls,
			bool_phone_verified_fl_list=bool_phone_verified_fls,
			bool_facebook_connected_fl_list=bool_facebook_connected_fls,
			num_client_reviews_list=num_client_reviewss,
			bool_payment_verified_cl_list=bool_payment_verified_cls,
			bool_profile_completed_cl_list=bool_profile_completed_cls,
			bool_phone_verified_cl_list=bool_phone_verified_cls,
			bool_deposit_made_cl_list=bool_deposit_made_cls
		)

		# return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=4)
		return torch.utils.data.DataLoader(dataset, batch_size=len(item_list), shuffle=False, num_workers=4)

class Rating_Datset(torch.utils.data.Dataset):
	def __init__(self, user_list, item_list, rating_list,bool_corporate_fl_list ,
			txt_country_fl_list,
			num_reviews_fl_list,
			num_earnings_fl_list,
			bool_preferred_fl_list ,
			bool_verified_by_fl_list ,
			bool_identity_verified_fl_list,
			bool_payment_verified_fl_list ,
			bool_phone_verified_fl_list ,
			bool_facebook_connected_fl_list ,
			num_client_reviews_list ,
			bool_payment_verified_cl_list ,
			bool_profile_completed_cl_list ,
			bool_phone_verified_cl_list ,
			bool_deposit_made_cl_list
				 ):
		super(Rating_Datset, self).__init__()
		self.user_list = user_list
		self.item_list = item_list
		self.rating_list = rating_list
		self.bool_corporate_fl_list  = bool_corporate_fl_list
		self.txt_country_fl_list = txt_country_fl_list
		self.num_reviews_fl_list = num_reviews_fl_list
		self.num_earnings_fl_list = num_earnings_fl_list
		self.bool_preferred_fl_list = bool_preferred_fl_list
		self.bool_verified_by_fl_list = bool_verified_by_fl_list
		self.bool_identity_verified_fl_list = bool_identity_verified_fl_list
		self.bool_payment_verified_fl_list = bool_payment_verified_fl_list
		self.bool_phone_verified_fl_list = bool_phone_verified_fl_list
		self.bool_facebook_connected_fl_list = bool_facebook_connected_fl_list
		self.num_client_reviews_list = num_client_reviews_list
		self.bool_payment_verified_cl_list = bool_payment_verified_cl_list
		self.bool_profile_completed_cl_list = bool_profile_completed_cl_list
		self.bool_phone_verified_cl_list = bool_phone_verified_cl_list
		self.bool_deposit_made_cl_list = bool_deposit_made_cl_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):

		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		bool_corporate_fl= self.bool_corporate_fl_list[idx]
		txt_country_fl= self.txt_country_fl_list[idx]
		num_reviews_fl= self.num_reviews_fl_list[idx]
		num_earnings_fl= self.num_earnings_fl_list[idx]
		bool_preferred_fl= self.bool_preferred_fl_list[idx]
		bool_verified_by_fl= self.bool_verified_by_fl_list[idx]
		bool_identity_verified_fl= self.bool_identity_verified_fl_list[idx]
		bool_payment_verified_fl= self.bool_payment_verified_fl_list[idx]
		bool_phone_verified_fl= self.bool_phone_verified_fl_list[idx]
		bool_facebook_connected_fl= self.bool_facebook_connected_fl_list[idx]
		num_client_reviews= self.num_client_reviews_list[idx]
		bool_payment_verified_cl= self.bool_payment_verified_cl_list[idx]
		bool_profile_completed_cl= self.bool_profile_completed_cl_list[idx]
		bool_phone_verified_cl= self.bool_phone_verified_cl_list[idx]
		bool_deposit_made_cl= self.bool_deposit_made_cl_list[idx]


		return (
			torch.tensor(user, dtype=torch.long),
			torch.tensor(item, dtype=torch.long),
			torch.tensor(rating, dtype=torch.float),

			torch.tensor(bool_corporate_fl, dtype=torch.int32),
			torch.tensor(txt_country_fl, dtype=torch.int32),
			torch.tensor(num_reviews_fl, dtype=torch.int32),
			torch.tensor(num_earnings_fl, dtype=torch.int32),
			torch.tensor(bool_preferred_fl, dtype=torch.int32),
			torch.tensor(bool_verified_by_fl, dtype=torch.int32),

		    torch.tensor(bool_identity_verified_fl, dtype=torch.int32),
		    torch.tensor(bool_payment_verified_fl, dtype=torch.int32),
		    torch.tensor(bool_phone_verified_fl, dtype=torch.int32),
		    torch.tensor(bool_facebook_connected_fl, dtype=torch.int32),
		    torch.tensor(num_client_reviews, dtype=torch.int32),

		    torch.tensor(bool_payment_verified_cl, dtype=torch.int32),
			torch.tensor(bool_profile_completed_cl, dtype=torch.int32),
			torch.tensor(bool_phone_verified_cl, dtype=torch.int32),
			torch.tensor(bool_deposit_made_cl, dtype=torch.int32)
			)