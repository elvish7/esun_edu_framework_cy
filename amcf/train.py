from pandas.io import pickle
from amcf import AMCF
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
import numpy as np
from utils_amcf import get_data, item_to_genre
from evaluate import XEval
from scipy.stats import ttest_1samp
import pickle
from datetime import datetime

class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self, epoch, asp_n):
        self.path = 'Data/'
        self.dataset = 'fund' 
        self.epochs = epoch
        self.batch_size = 256
        self.num_asp = asp_n #13 #14 #2 #6 #18 # ml:18
        self.e_dim = 128 #120
        # self.mlp_dim = [64, 32, 16]
        self.reg = 1e-1
        self.bias_reg = 3e-3
        self.asp_reg = 5e-3 #5e-3
        # self.num_neg = 4
        self.lr = 7e-4 # 4e-3(score1-11/nan) 7e-3
        self.bias_lr = 7e-3
        self.asp_lr = 7e-2
        self.lambda1 = 5e-2 # 5e-2
        # self.loss_weights = [1, 1, 1]


def train(model, trainloader, testloader, evaluator, optimizer, criterion, device, args, data_fund):
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_mse_loss = 0.0
        running_sim_loss = 0.0
        epoch_size = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # get genre information from item id
            item_asp = item_to_genre(inputs[:, 1], data_fund).values
            item_asp = torch.Tensor(item_asp).to(device)
            # inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            epoch_size += len(user_inputs) # calculate the total number of samples in an epoch

            optimizer.zero_grad()
            outputs, cos_sim, pref = model(user_inputs, item_inputs, item_asp) # cos_sim stands for a distance measure.
            outputs = outputs.flatten()
            cos_sim = cos_sim.flatten()

            mse_loss = criterion(outputs, labels.to(torch.float)) # to float
            sim_loss = cos_sim.sum()
            # combined loss
            loss = mse_loss + (args.lambda1 * sim_loss)  
            #loss = mse_loss # no explaination
            loss.backward()
            optimizer.step()
            # collect running losses
            running_loss += (mse_loss + (args.lambda1 * sim_loss)).data
            running_mse_loss += mse_loss.data
            running_sim_loss += sim_loss.data

        # total loss
        epoch_loss = running_loss / epoch_size
        # rmse loss
        epoch_mse_loss = running_mse_loss / epoch_size
        rmse = np.sqrt(epoch_mse_loss.cpu().numpy())
        # sim loss
        epoch_sim_loss = running_sim_loss / epoch_size
        print("Epoch {:d}: the training RMSE loss: {:.4f}, item embedding similarity: {:.4f}".format(epoch, rmse, epoch_sim_loss))
        print("Total loss: {:.4f}".format(epoch_loss))
    
        vrmse, vmae, cos_sim = test(model, testloader, evaluator, criterion, device, args, data_fund)
        model.train()
    print(30*'+' + 'training completed!' + 30*'+')
    
    return model


def test(model, testloader, evaluator, criterion, device, args, data_fund):
    model.eval()
    for epoch in range(1):
        running_loss = 0.0
        running_l1loss = 0.0
        running_cos_sim = 0.0
        epoch_size = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            # get genre information from item id
            item_asp = item_to_genre(inputs[:, 1], data_fund).values
            item_asp = torch.Tensor(item_asp).to(device)

            # get user list tensor in CPU
            user_in_batch = inputs[:, 0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            epoch_size += len(user_inputs) # calculate the total number of samples in an epoch

            # optimizer.zero_grad()
            outputs, cos_sim, pref = model(user_inputs, item_inputs, item_asp)
            outputs = outputs.flatten()

            # loss = get_sse(outputs.data.cpu().numpy(), labels.data.cpu().numpy())
            loss = criterion(outputs, labels.to(torch.float)) # to float
            l1loss = nn.L1Loss(reduction='sum')
            total_l1 = l1loss(outputs, labels.to(torch.float))

            running_loss += loss.data
            running_l1loss += total_l1


        epoch_loss = running_loss / epoch_size # get the average loss
        mae = running_l1loss / epoch_size
        rmse = np.sqrt(epoch_loss.cpu().numpy()) # get RMSE by sqrt

        print("The validation RMSE loss: {:.4f}".format(rmse))
        print("The validation MAE loss: {:.4f}".format(mae))

        users = torch.tensor(range(len(user_inputs)), dtype=torch.long).to(device)
        u_pred = model.predict_pref(users)

        cos_sim = evaluator.get_cos_sim(users, u_pred, data_fund)#.mean()
        print("average cos_sim is: {:.4f}".format(cos_sim.mean()))
    
    return rmse, mae, cos_sim.mean().item()



def model_training(user_n, item_n, asp_n, data_rating, data_fund, epoch, weights):
    print(30*'+' + 'Start training' + 30*'+', datetime.now())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = Args(epoch, asp_n)
    # determine data size
    num_users = user_n
    num_items = item_n  

    # data_loaders contains all K-Fold train_loaders and test_loaders
    data_loaders = get_data(data_rating, batch_size=args.batch_size)
    evaluator = XEval(data_rating, data_fund, dataset=args.dataset)
    # load datasets
    for trainloader, testloader in data_loaders:
        # Build model
        model = AMCF(num_user=num_users, num_item=num_items, weights=weights, num_asp=args.num_asp, e_dim=args.e_dim, ave=evaluator.get_all_ave())
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        
        # set parameter learning rates and regularizations
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key == 'i_bias' or key == 'u_bias':
                params += [{'params':[value],'lr':args.bias_lr, 'weight_decay':args.bias_reg}]
            elif key =='asp_emb.W':
                params += [{'params':[value],'lr':args.asp_lr, 'weight_decay':args.asp_reg}]
            else:
                params += [{'params':[value],'lr':args.lr, 'weight_decay':args.reg}]
        optimizer = SGD(params, lr=args.lr, weight_decay=args.reg)
        
        criterion = nn.MSELoss(reduction='sum')

        # calculate validation losses
        fitted_model = train(model, trainloader, testloader, evaluator, optimizer, criterion, device, args, data_fund)
        val_rmse, val_mae, cos_sim = test(fitted_model, testloader, evaluator, criterion, device, args, data_fund)

    model_path = 'AMCF_model.pt'
    torch.save(fitted_model, model_path)
    print('Model saved at: ', model_path)

    print(30*'+' + 'Finish!' + 30*'+', datetime.now())
    return fitted_model, val_rmse, cos_sim

#if __name__ == '__main__':
#    main()