#lstm data-stream
#2023-3-8
#encoding=utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader,TensorDataset, Dataset
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from scipy.spatial import distance
from pykalman import KalmanFilter
import random
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers_num, drop_rate,CHOOSE):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size= input_dim, 
            hidden_size= hidden_dim,  
            num_layers= layers_num ,  
            bias=True,  
            dropout=drop_rate,  
            batch_first=True,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),

        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            # nn.Dropout(p=0.1),
        )
        self.choose = CHOOSE

    def forward(self,x):
        r_out, (h_t, c_t) = self.lstm(x)
        batch, time_step, hidden_size = r_out.shape
        out = r_out[:, -1, :].view(batch, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        if self.choose == 'True':
            print(out)
            print(h_t,c_t)
            print(h_t.shape,c_t.shape)
            print(h_t[-1,:],c_t[-1,:])
            sys.exit(0)
            return out, h_t[-1,:], c_t[-1,:]
        else:
            return out
        
class Data_set(Dataset):
    def __init__(self,x,y):

        self.x = x
        self.y = y

    def __getitem__(self, id):
        data = (self.x[id],
                self.y[id])
        return data

    def __len__(self):
        return len(self.y)
    
def sliding_window(data, sw_width, in_start=0):
    X = []
    data = np.array(data, type(float))
    for _ in range(data.shape[0]):
        in_end = in_start + sw_width
        if (in_end>data.shape[0]):
            break
        else:
            train_seq = data[in_start:in_end, :]
            X.append(train_seq)
            in_start += 1
    return np.array(X)

def chunk_process(data):
    x = torch.tensor(data[:, 0:-1].astype(float), dtype=torch.float).unsqueeze(0)
    y = torch.tensor(data[-1:, -1:].astype(float), dtype=torch.float).squeeze()
    return x,y

def model_train(ft_model,buffer,max_ft_epoch,ft_initial_lr,ft_batch_size,patiences):
    model = ft_model.to(device)
    criterion_ft = nn.MSELoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=ft_initial_lr)
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10)
    patiences = patiences 
    best_loss = float('inf')
    epochs_no_improve = 0

    fine_tune_x = torch.tensor(buffer[:,:,0:-1].astype(np.float32))
    fine_tune_y = torch.tensor(buffer[:,-1,-1:].astype(np.float32))
    data_fine_tune = Data_set(fine_tune_x, fine_tune_y)
    loader_fine_tune = DataLoader(data_fine_tune, batch_size=ft_batch_size, shuffle=False, drop_last=True)
    for epoch in range(max_ft_epoch):
        model.train()
        fine_tune_loss = 0.0
        for _ in loader_fine_tune:
            x, y = _
            x = x.to(device)
            y = y.to(device)
            optimizer_ft.zero_grad()
            pred,htt,ctt = model.forward(x)
            loss = criterion_ft(pred.squeeze(), y.squeeze())
            loss.backward(retain_graph=True)
            optimizer_ft.step()

            fine_tune_loss += loss.item()

        avg_tune_loss = fine_tune_loss / len(loader_fine_tune)

        if avg_tune_loss < best_loss:
            best_loss = avg_tune_loss
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patiences:
                break
        sheduler.step(loss)
    return model

def cosine_decay_array(start_lr, min_lr, decay_steps, total_steps):
    factor = (math.pi / decay_steps)
    array_zeros = np.zeros((total_steps))
    for i in range(total_steps):
        if i < decay_steps:
            array_zeros[i] = min_lr + 0.5 * (start_lr - min_lr) * (1.0 + math.cos(factor * i))
        else:
            array_zeros[i] = min_lr
    return array_zeros

if __name__ == '__main__':
    #----------------------------------------------------------
    # lstm param
    INPUT_SIZE = 8 
    HIDDEN_SIZE = 32
    LAYER_NUM = 3
    OUTPUT_SIZE = 1 
    TIME_STEP = 31 
    DROP_RATE = 0.1

    threshold_initial = 30 
    BUFFER_LEN = 8 
    fine_tune_batch_size = 3 
    max_fine_tune_epoch = 250
    fine_tune_initial_lr = 1e-3 

    kc = 30 
    alpha = 5
    beta = 1
    gamma = 0.1
    #----------------------------------------------------------

    data_set_1977_1997 = np.array(pd.read_excel(r'./data_period1.xlsx'))
    data_set_2007_2014 = np.array(pd.read_excel(r'./data_period2.xlsx'))

    train_set = data_set_1977_1997[:3652,3:]
    train_x = torch.tensor(sliding_window((train_set[:, 0:-1]), TIME_STEP).astype(np.float32))
    train_y = torch.tensor(sliding_window((train_set[:, -1:]), TIME_STEP)[:, -1, :].astype(np.float32))
    loader_lstm = DataLoader(Data_set(train_x, train_y), batch_size=128, shuffle=False, drop_last=True)

    stream_set_one = data_set_1977_1997[3652:, 3:]
    stream_set_two = data_set_2007_2014[:, 3:-1]
    chunks_one = sliding_window(stream_set_one,TIME_STEP)
    chunks_two = sliding_window(stream_set_two,TIME_STEP) 

    chunks = np.concatenate((chunks_one,chunks_two),axis=0) 

    test_y_one = sliding_window((stream_set_one[:, -1:]), TIME_STEP)[:, -1, :].astype(np.float32)
    test_y_two = sliding_window((stream_set_two[:, -1:]), TIME_STEP)[:, -1, :].astype(np.float32)
    test_y = np.concatenate((test_y_one,test_y_two),axis=0)

    # -------------------------------------------------------
    model_num = 5
    prediction_all = np.zeros((model_num,test_y.shape[0]))
    for seed_num in tqdm(range(0,model_num)):
        seed = random.randint(0, 4294967295)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_default_tensor_type(torch.FloatTensor)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        model_lstm = RNN(input_dim= INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layers_num=LAYER_NUM, drop_rate=DROP_RATE,CHOOSE=False).to(device)
        optimizer = optim.Adam(model_lstm.parameters(),lr = 5e-3)
        criterion = nn.MSELoss()
        model_lstm.train()
        lowest_loss = 10000000
        for epoch in range(250):
            for data in loader_lstm:
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model_lstm.forward(x).squeeze()
                loss = criterion(pred, y.squeeze())
                loss.backward()
                optimizer.step()
        
            checkpoint = {"model_state_dict": model_lstm.state_dict(),
                    "optimizer_state_dic": optimizer.state_dict(),
                    "loss": loss,
                    "epoch": epoch}
            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                save_model_path = r'.\temp'
                path_checkpoint = save_model_path+'/'+'best_model.pkl'
                torch.save(checkpoint, path_checkpoint)
                # print('best model')
        
            nse = metrics.r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy()) 
            # print('Epoch:', '%04d' % (epoch), 'loss:', loss.item(), 'NSE:', nse)
        # # print('train finish:',model_num)
        
        rnn = RNN(input_dim= INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layers_num=LAYER_NUM, drop_rate=DROP_RATE,CHOOSE='True').to(device)
        rnn.load_state_dict(torch.load(r'.\temp\best_model.pkl')['model_state_dict'])


        prediction = []
        targets = []
        err_list = [] 
        # hidden_dist = []
        ht_dist = []
        ct_dist = []

        buffer = np.empty((1,TIME_STEP, INPUT_SIZE+1))
        time = 0
        # drift_num = 0
        fine_tuning = False
        mark_time = 0
        temp_thresh = []

        # threshold = threshold_initial ## ## stream delete

        ht_diff= []
        ct_diff= []
        err_diff= []
        loss_com_all = []

        #stream
        for chunk in chunks:
            if time > 0 :
                target_last = targets[-1]
                pred_last = prediction[-1]
                error = target_last - pred_last
                err_list.append(error.squeeze()) # all error
                if len(err_list) > kc-1:
                    ht_mean = torch.mean(ht_temp[-kc-1:-1],dim=0)
                    ct_mean = torch.mean(ct_temp[-kc-1:-1],dim=0)
                    err_mean = np.mean(err_list[-kc-1:-1])

                    ht_dis_val = torch.norm(ht_mean - ht_temp[-1]).detach().numpy() 
                    ct_dis_val = torch.norm(ct_mean - ht_temp[-1]).detach().numpy()
                    err_val = np.absolute(error-err_mean).squeeze()

                    ht_diff.append(ht_dis_val)
                    ct_diff.append(ct_dis_val)
                    err_diff.append(err_val)
                    # print(ht_dis_val,ct_dis_val,err_val)

                    loss_com = alpha * ht_dis_val + beta * ct_dis_val + gamma * err_val
                    loss_com_all.append(loss_com)

                    time_point = time - mark_time
                    if mark_time == 0:
                        threshold = threshold_initial
                    else:
                        threshold = threshold_array[time_point]

                    temp_thresh.append(threshold)

                    # # # threshold = 20

                    if loss_com > threshold:
                        # print('drift occurs.')
                        fine_tuning = True

                        if fine_tuning == True:
                
                            for param in rnn.lstm.parameters():
                                param.requires_grad = False
                            # rnn.fc1[0].reset_parameters()
                            rnn.fc2[0].reset_parameters()
                            rnn.fc2[2].reset_parameters()
                            # for param in rnn.fc1.parameters():
                            #     param.requires_grad = False
                            # for param in rnn.fc2.parameters():
                            #     param.requires_grad = False

                            rnn.train()
                            rnn = model_train(rnn, buffer, max_fine_tune_epoch, fine_tune_initial_lr, fine_tune_batch_size, patiences=10)

                            fine_tuning = False

                            threshold_array = cosine_decay_array(loss_com, 10, 70, 200) #start_lr, min_lr, decay_steps, total_steps
                            mark_time = time


            x, y = chunk_process(chunk) #tensor bz,ts,fea  1
            targets.append(np.array(y)) #numpy
            rnn.eval()

            pred_temp, ht, ct = rnn.forward(x.to(device)) #tensor  1  hidden size:[1,64]

            prediction.append(pred_temp.cpu().detach().numpy().squeeze())

            if time == 0: # hidden cell save
                ht_temp = ht.cpu()
                ct_temp = ct.cpu()
            else:
                ht_temp = torch.cat((ht_temp,ht.cpu())) # [k,hidden,size]
                ct_temp = torch.cat((ct_temp,ct.cpu())) # [k,hidden,size]

            buffer = np.concatenate((buffer,np.expand_dims(chunk,axis=0)), axis=0) # [ 1 ts fea+1 ]
            if buffer.shape[0] > (BUFFER_LEN-1) :
                buffer = np.delete(buffer, 0, axis=0)

            time += 1

            #time.sleep(0.01)

        prediction_all[seed_num,:] = prediction


    prediction_stream = np.mean(prediction_all[:,:],axis=0).reshape(-1,1)
    test_y = test_y.reshape(-1,1)
    R2_lstm = metrics.r2_score(test_y, prediction_stream)
    print('R2_lstm:', R2_lstm)
    