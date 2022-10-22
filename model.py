import torch
import torch.nn as nn
import numpy as np
class Generalized_Matrix_Factorization(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.affine_output = nn.Linear(in_features=self.factor_num, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Multi_Layer_Perceptron, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num
        self.layers = args.layers

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
            # vector = nn.BatchNorm1d()(vector)
            # vector = nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass



class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items,num_list):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_list = num_list
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)
        # unchange embedding
        self.embedding_corporate_mlp = nn.Embedding(num_embeddings=self.num_list[0], embedding_dim=4)
        self.embedding_country_mlp = nn.Embedding(num_embeddings=self.num_list[1], embedding_dim=16)

        self.embedding_reviews_mlp = nn.Embedding(num_embeddings=self.num_list[2], embedding_dim=55)
        self.embedding_earnings_mlp = nn.Embedding(num_embeddings=self.num_list[3], embedding_dim=25)

        self.unchang_output = nn.Linear(in_features=100 , out_features=26)
        # change embedding
        self.embedding_preferred_mlp = nn.Embedding(num_embeddings=self.num_list[4], embedding_dim=4)
        self.embedding_verified_mlp = nn.Embedding(num_embeddings=self.num_list[5], embedding_dim=4)

        self.embedding_identity_mlp = nn.Embedding(num_embeddings=self.num_list[6], embedding_dim=4)
        self.embedding_payment_mlp = nn.Embedding(num_embeddings=self.num_list[7], embedding_dim=4)

        self.embedding_phone_mlp = nn.Embedding(num_embeddings=self.num_list[8], embedding_dim=4)
        self.embedding_facebook_mlp = nn.Embedding(num_embeddings=self.num_list[9], embedding_dim=4)

        self.chang_output = nn.Linear(in_features=24, out_features=12)
        #  other_feas
        self.embedding_client_mlp = nn.Embedding(num_embeddings=self.num_list[10], embedding_dim=180)
        self.embedding_paymentcl_mlp = nn.Embedding(num_embeddings=self.num_list[11], embedding_dim=4)

        self.embedding_profile_mlp = nn.Embedding(num_embeddings=self.num_list[12], embedding_dim=4)
        self.embedding_phonecl_mlp = nn.Embedding(num_embeddings=self.num_list[13], embedding_dim=4)

        self.embedding_deposit_mlp = nn.Embedding(num_embeddings=self.num_list[14], embedding_dim=4)
        self.other_output = nn.Linear(in_features=196, out_features=26)



        self.feature = nn.Linear(in_features=8, out_features=1)

        #skill layer
        # self.linear_fl_skill = nn.Linear(in_features=193, out_features=32) #123.csv
        # self.linear_fl_skill = nn.Linear(in_features=964, out_features=128)
        # self.linear_fl_skill = nn.Linear(in_features=837, out_features=128) #20w

        self.linear_fl_skill = nn.Linear(in_features=892, out_features=128)  # 50w
        self.S = nn.ReLU()

        # # self.linear_cl_skill = nn.Linear(in_features=201, out_features=32) #123.csv
        # self.linear_fl_skill = nn.Linear(in_features=185, out_features=128)

        self.linear_skill = nn.Linear(in_features=1, out_features=1)
        # feature layers
        self.feature_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.feature_layers.append(torch.nn.Linear(in_size, out_size))
            self.feature_layers.append(nn.ReLU())


        # base layers
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        # output
        # self.affine_output = nn.Linear(in_features=5*args.layers[-1] + self.factor_num_mf + 16, out_features=1)
        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf , out_features=1)

        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        # unchange embedding
        nn.init.normal_(self.embedding_corporate_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_country_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_reviews_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_earnings_mlp.weight, std=0.01)
        # change embedding
        nn.init.normal_(self.embedding_preferred_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_verified_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_identity_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_phone_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_facebook_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_payment_mlp.weight, std=0.01)
        #other
        nn.init.normal_(self.embedding_client_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_paymentcl_mlp.weight, std=0.01)
        nn.init.normal_( self.embedding_profile_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_phonecl_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_deposit_mlp.weight, std=0.01)


        for m in self.feature_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        #feature
        nn.init.xavier_uniform_(self.unchang_output.weight)
        nn.init.xavier_uniform_(self.chang_output.weight)
        nn.init.xavier_uniform_(self.other_output.weight)

        #skill
        nn.init.xavier_uniform_(self.linear_fl_skill.weight)
        # nn.init.xavier_uniform_(self.linear_cl_skill.weight)
        nn.init.xavier_uniform_(self.linear_skill.weight)

        nn.init.xavier_uniform_(self.feature.weight)



        #base
        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices,freelancers_fal,feature1,fl_tensor_skill,cl_tensor_skill,skill,web):
        # print("user_indices",user_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)


        """---------------skill's embedding------------------------------------"""


        Embedding_skill = self.linear_fl_skill(skill)  # freelancer  skill_embded
        # Embedding_skill = nn.functional.normalize(Embedding_skill, p=2, dim=1)
        # Embedding_skill = 0.5*Embedding_skill
        Embedding_skill  = self.S(Embedding_skill )
        # Embedding_skill.clamp_min_(0)
        # print("Embedding_skill", Embedding_skill)
        # Embedding_skill_fl = torch.tensor([],device = device)
        # Embedding_skill_cl = torch.tensor([],device = device)
        Embedding_skill_fl = torch.tensor([])
        Embedding_skill_cl = torch.tensor([])
        #
        #
        if web ==1:
            Embedding_skill_fl = torch.load('Embedding_skill_fl.pth')
            # print(Embedding_skill_fl.size())

            Embedding_skill_fl = Embedding_skill_fl[freelancers_fal[:-1]]

            a = torch.sum(Embedding_skill[fl_tensor_skill[-1]],dim = 0 )
            a = a.unsqueeze(0)

            Embedding_skill_cl1 = torch.sum(Embedding_skill[cl_tensor_skill[-1]],dim = 0 )

            Embedding_skill_cl = Embedding_skill_cl1.expand( Embedding_skill_fl.size(0),Embedding_skill_cl1.size(0))

            b = torch.sum(Embedding_skill[cl_tensor_skill[-1]], dim=0)
            b = b.unsqueeze(0)
            Embedding_skill_fl = torch.cat((Embedding_skill_fl, a), 0)
            Embedding_skill_cl = torch.cat((Embedding_skill_cl, b),0)
        if web ==0:
            # Embedding_skill_fl = torch.load('Embedding_skill_fl.pth')
            # Embedding_skill_fl = Embedding_skill_fl[freelancers_fal]
            # Embedding_skill_cl1 = torch.sum(Embedding_skill[cl_tensor_skill[-1]], dim=0)

            for i in range(len(fl_tensor_skill)):
                a = torch.sum(Embedding_skill[fl_tensor_skill[i]], dim=0)
                a = a.unsqueeze(0)
                b = torch.sum(Embedding_skill[cl_tensor_skill[i]], dim=0)
                b = b.unsqueeze(0)
                Embedding_skill_fl = torch.cat((Embedding_skill_fl, a), 0)
                Embedding_skill_cl = torch.cat((Embedding_skill_cl, b), 0)
            # Embedding_skill_cl = Embedding_skill_cl1.expand(Embedding_skill_fl.size(0), Embedding_skill_cl1.size(0))

        # torch.save(Embedding_skill_fl,'Embedding_skill_fl.pth')
        # print("Embedding_skill_fl",Embedding_skill_fl.size())
        # torch.save(Embedding_skill_cl, 'Embedding_skill_cl.pth')


        mul_skill = torch.mul(Embedding_skill_fl, Embedding_skill_cl)

        mul_skill = torch.sum(mul_skill,dim=-1)
        # print("mul_skill2", mul_skill)
        # print("self.linear_skill",self.linear_skill.weight)
        mul_skill = mul_skill.view(mul_skill.size(0),1)
        mul_skill = self.linear_skill(mul_skill)

        # unchange embedding
        corporate_embedding_mlp = self.embedding_corporate_mlp(feature1['corporate'][freelancers_fal])

        country_embedding_mlp = self.embedding_country_mlp(feature1['country'][freelancers_fal])

        reviews_embedding_mlp= self.embedding_reviews_mlp(feature1['reviews'][freelancers_fal])
        earnings_membedding_lp= self.embedding_earnings_mlp(feature1['earnings'][freelancers_fal])

        # change embedding
        preferred_embedding_mlp= self.embedding_preferred_mlp(feature1['preferred'][freelancers_fal])
        verified_embedding_mlp= self.embedding_verified_mlp(feature1['verified'][freelancers_fal])

        identity_embedding_mlp = self.embedding_identity_mlp(feature1['identity'][freelancers_fal])
        payment_embedding_mlp= self.embedding_payment_mlp(feature1['payment'][freelancers_fal])

        phone_embedding_mlp = self.embedding_phone_mlp(feature1['phone'][freelancers_fal])
        facebook_embedding_mlp=  self.embedding_facebook_mlp(feature1['facebook'][freelancers_fal])

        #  other_feas
        client_embedding_mlp= self.embedding_client_mlp(feature1['client'][freelancers_fal])
        paymentcl_embedding_mlp= self.embedding_paymentcl_mlp(feature1['paymentcl'][freelancers_fal])
        profile_embedding_mlp= self.embedding_profile_mlp(feature1['profile'][freelancers_fal])
        phonecl_embedding_mlp= self.embedding_phonecl_mlp(feature1['phonecl'][user_indices])

        deposit_embedding_mlp = self.embedding_deposit_mlp(feature1['deposit'][user_indices])
        fl_unchange_feature = torch.cat([corporate_embedding_mlp, country_embedding_mlp,reviews_embedding_mlp,earnings_membedding_lp],dim=-1)
        fl_change_feature = torch.cat([preferred_embedding_mlp,verified_embedding_mlp,identity_embedding_mlp,payment_embedding_mlp,phone_embedding_mlp,facebook_embedding_mlp],dim=-1)
        cl_feature = torch.cat([client_embedding_mlp,paymentcl_embedding_mlp,profile_embedding_mlp, phonecl_embedding_mlp,deposit_embedding_mlp],dim=-1)

        fl_unchange_feature1 = self.unchang_output(fl_unchange_feature)
        fl_change_feature1 = self.chang_output(fl_change_feature)

        cl_feature1 = self.other_output(cl_feature)

        feature_vector = torch.cat([ fl_unchange_feature1,fl_change_feature1,cl_feature1],dim=-1)

        for idx, _ in enumerate(range(len(self.feature_layers))):
            feature_vector = self.feature_layers[idx](feature_vector)

        feature_vector = self.feature(feature_vector)

        # job & freelancer
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        # vector = torch.cat([mlp_vector, mf_vector,feature_vector,mul_skill,score_feature], dim=-1)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)
        # print("logits",logits)
        # # print("feature_vector",feature_vector)
        # print("mul_skill",mul_skill)

        rating = self.logistic(logits + feature_vector + mul_skill)
        # rating = logits + feature_vector + mul_skill
        return rating.squeeze(),Embedding_skill

    def get_fea(self, feature1):

        # unchange embedding
        corporate_embedding_mlp = self.embedding_corporate_mlp(feature1['bool_corporate_fl'])
        country_embedding_mlp = self.embedding_country_mlp(feature1['txt_country_fl'])

        reviews_embedding_mlp= self.embedding_reviews_mlp(feature1['num_reviews_fl'])
        earnings_membedding_lp= self.embedding_earnings_mlp(feature1['num_earnings_fl'])

        # change embedding
        preferred_embedding_mlp= self.embedding_preferred_mlp(feature1['bool_preferred_fl'])
        verified_embedding_mlp= self.embedding_verified_mlp(feature1['bool_verified_by_fl'])

        identity_embedding_mlp = self.embedding_identity_mlp(feature1['bool_identity_verified_fl'])
        payment_embedding_mlp= self.embedding_payment_mlp(feature1['bool_payment_verified_fl'])

        phone_embedding_mlp = self.embedding_phone_mlp(feature1['bool_phone_verified_fl'])
        facebook_embedding_mlp=  self.embedding_facebook_mlp(feature1['bool_facebook_connected_fl'])

        #  other_feas
        client_embedding_mlp= self.embedding_client_mlp(feature1['num_client_reviews'])
        paymentcl_embedding_mlp= self.embedding_paymentcl_mlp(feature1['bool_payment_verified_cl'])

        profile_embedding_mlp= self.embedding_profile_mlp(feature1['bool_profile_completed_cl'])
        phonecl_embedding_mlp= self.embedding_phonecl_mlp(feature1['bool_phone_verified_cl'])

        deposit_embedding_mlp = self.embedding_deposit_mlp(feature1['bool_deposit_made_cl'])
        fl_unchange_feature = torch.cat([corporate_embedding_mlp, country_embedding_mlp,reviews_embedding_mlp,earnings_membedding_lp],dim=-1)
        fl_change_feature = torch.cat([preferred_embedding_mlp,verified_embedding_mlp,identity_embedding_mlp,payment_embedding_mlp,phone_embedding_mlp,facebook_embedding_mlp],dim=-1)
        cl_feature = torch.cat([client_embedding_mlp,paymentcl_embedding_mlp,profile_embedding_mlp, phonecl_embedding_mlp,deposit_embedding_mlp],dim=-1)

        fl_unchange_feature1 = self.unchang_output(fl_unchange_feature)
        fl_change_feature1 = self.chang_output(fl_change_feature)

        cl_feature1 = self.other_output(cl_feature)

        feature_vector = torch.cat([ fl_unchange_feature1,fl_change_feature1,cl_feature1],dim=-1)

        for idx, _ in enumerate(range(len(self.feature_layers))):
            feature_vector = self.feature_layers[idx](feature_vector)

        feature = self.feature(feature_vector)
        return feature_vector,feature



    def get_rate(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # job & freelancer
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        # vector = torch.cat([mlp_vector, mf_vector,feature_vector,mul_skill,score_feature], dim=-1)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)

        return logits,vector
    def get_skill(self,skill, fl_tensor_skill):

        # Embedding_skill = self.linear_fl_skill(skill)  # freelancer  skill_embded
        Embedding_skill = self.S(self.linear_fl_skill(skill))
        # print(Embedding_skill)
        # Embedding_skill = nn.functional.normalize(Embedding_skill, p=2, dim=1)
        # Embedding_skill = Embedding_skill*0.5

        a = torch.sum(Embedding_skill[fl_tensor_skill], dim=0)
        a = a.unsqueeze(0)
        return a,Embedding_skill