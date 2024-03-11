import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterSampler, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter


class BRNNModel:
    def __init__(self,look_forward, look_back, slidingWindow, epoch=10, batch_size=128):
        self.look_forward = look_forward
        self.look_back = look_back
        self.slidingWindow = slidingWindow
        self.epoch = epoch
        self.batch_size = batch_size

    def processData(self, thanhly=None, luuluong=None, evaluate=True):
        # xóa Nan
        luuluong=luuluong.dropna()
        # Chuyển đổi cột NGAY sang định dạng datetime
        try:
            luuluong.loc[:,'NGAY'] = pd.to_datetime(luuluong['NGAY'], format='%d-%b-%y')
        except:
            luuluong.loc[:,'NGAY'] = pd.to_datetime(luuluong['NGAY'], format='%m/%d/%Y')

        # xóa Khách hàng có lích sử ít hơn 30 ngày
        # # Tính tổng UP và DOWN theo từng KHACHHANG_ID và từng NGAY
        luuluong = luuluong.groupby(['KHACHHANG_ID', 'NGAY']).agg({'UP': 'sum', 'DOWN': 'sum'}).reset_index()
        count_days = luuluong.groupby("KHACHHANG_ID")['NGAY'].count()
        Threshold = count_days >=30
        luuluong = luuluong[luuluong["KHACHHANG_ID"].isin(count_days[Threshold].index)]
        # print(len(luuluong["KHACHHANG_ID"].unique()))
        # Sắp xếp dữ liệu 
        luuluong = luuluong.sort_values(by=['KHACHHANG_ID','NGAY'])
        # làm đầy dữ liệu.
        new_data = pd.DataFrame()
        for id in luuluong['KHACHHANG_ID'].unique():
            customer = luuluong[luuluong['KHACHHANG_ID']==id]
            date_list = customer["NGAY"].unique()
            start = date_list.min()
            end = date_list.max()
            # print(date_max, date_tl)
            date_range = pd.date_range(start, end)
            if(new_data.empty):
                new_data = pd.DataFrame([(id, date) for date in date_range], columns=['KHACHHANG_ID','NGAY'])
                continue
            current = pd.DataFrame([(id, date) for date in date_range], columns=['KHACHHANG_ID','NGAY'])
            new_data = pd.concat([new_data, current], ignore_index=True)
        
        data = luuluong.merge(new_data[['KHACHHANG_ID','NGAY']], on=['KHACHHANG_ID', 'NGAY'], how='right').fillna({'UP':0,'DOWN':0})
        # # print(data['KHACHHANG_ID'].value_counts())
        # gộp 2 file csv 
        if(evaluate==True): 
            try:
                thanhly['NGAY'] = pd.to_datetime(thanhly['NGAY'], format='%d-%b-%y')
            except:
                thanhly['NGAY'] = pd.to_datetime(thanhly['NGAY'], format='%m/%d/%Y')

            thanhly = thanhly.sort_values(by=['KHACHHANG_ID'])
            # loại bỏ đối tượng trùng lặp.
            thanhly = thanhly.drop_duplicates(subset="KHACHHANG_ID", keep='first')
            data = data.merge(thanhly, on='KHACHHANG_ID', how='left')
            data = data.rename(columns = {'KHACHHANG_ID':'KHACHHANG_ID', 'NGAY_x':'NGAY', 'UP':'UP', 'DOWN':'DOWN','NGAY_y':'NGAYTHANHLY','THANHLY':'THANHLY'})
            data['UP'] = data['UP'].apply(lambda x: x / (1024 ** 3) if x != 0 else 0)
            data['DOWN'] = data['DOWN'].apply(lambda x: x / (1024 ** 3) if x != 0 else 0)

        return data
    

    def createSlidingWindow(self, customer, key, evaluate=True):
        sequences = []
        targets = []
        if(evaluate==True):
            if(len(customer) != (self.look_back+self.look_forward)):
                for i in range(0,(len(customer)-self.look_back-self.look_forward), self.slidingWindow):
                    sequences.append(customer[key].values[i:i+self.look_back])
                    targets.append(customer[key].values[i+self.look_back:i+self.look_back+self.look_forward])
            else:
                sequences.append(customer[key].values[:self.look_back])
                targets.append(customer[key].values[self.look_back:self.look_back+self.look_forward])
            sequences = np.array(sequences)
            targets = np.array(targets)
            sequences = sequences.reshape((sequences.shape[0], sequences.shape[1],1))
            targets = targets.reshape((targets.shape[0],targets.shape[1],1))
            return sequences, targets
        
        else:
            if(len(customer) != (self.look_back)):
                for i in range(0,(len(customer)-self.look_back-self.look_forward), self.slidingWindow):
                    sequences.append(customer[key].values[i:i+self.look_back])
                else:
                    sequences.append(customer[key].values[:self.look_back])
                sequences = np.array(sequences)
                sequences = sequences.reshape((sequences.shape[0], sequences.shape[1],1))
                return sequences


    def buildModel(self, model=None):
        if (model is None):
            model = keras.Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.look_back, 1)),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.35),
                Bidirectional(LSTM(128, return_sequences=True)),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.35),
                Bidirectional(LSTM(32)),
                Dropout(0.35),
                Dense(self.look_forward, activation='relu')
            ])

            # keras.utils.plot_model(model,to_file="BRNN1.png",show_shapes=True, show_layer_names=True, show_layer_activations=True)
            model.compile(optimizer='adam', loss='mae')
        return model

    def train_split(self, data, key):
        list_id = data['KHACHHANG_ID'].unique()
        sequences = np.empty((0,self.look_back,1))
        targets = np.empty((0,self.look_forward,1))
        for id in list_id:
            customer = data[data['KHACHHANG_ID']==id]
            sequence, target = self.createSlidingWindow(customer, key=key)
            sequences = np.concatenate((sequences, sequence), axis=0)
            targets = np.concatenate((targets, target), axis=0)
        return sequences, targets
    
    def fit(self, sequences, targets, model=None):
        model = self.buildModel(model)
        model.fit(sequences, targets, epochs=self.epoch, batch_size=self.batch_size, validation_split=0.2)
        return model
    
    def Threshold(self, data, key):
        customer_thanhly = data[data['THANHLY']==1]
        # customer_thanhly = customer_thanhly.groupby(['KHACHHANG_ID']).agg({key: 'sum'})
        Threshold = []
        for id in customer_thanhly["KHACHHANG_ID"].unique():
            customer = customer_thanhly[customer_thanhly["KHACHHANG_ID"]==id]
            customer = customer.tail(30)
            Threshold.append(np.sum(customer[key]))
        return np.mean(Threshold)
    
    def smean_absolute_error(self, model_up, model_down, data):
            sequences_up, targets_up = self.train_split(data, "UP")
            sequences_down, targets_down = self.train_split(data, "DOWN")
            loss_up = model_up.evaluate(sequences_up, targets_up)
            loss_down = model_down.evaluate(sequences_down, targets_down)
            return np.round(loss_up,4), np.round(loss_down,4)
    
    def predict_labels(self, model_up, model_down, data, Threshold_value_up, Threshold_value_down):

        list_id = data['KHACHHANG_ID'].unique()
        labels = []
        count = 0
        for id in list_id:
            customer = data[data['KHACHHANG_ID']==id]
            sequences_up = self.createSlidingWindow(customer, 'UP', evaluate=False)
            sequences_down= self.createSlidingWindow(customer, 'DOWN', evaluate=False)
            ##########################################################################
            sequence_up = sequences_up[-1]
            sequence_down = sequences_down[-1]
            ###########################################################################
            predict_up=model_up.predict(np.array([sequence_up]))
            predict_down=model_down.predict(np.array([sequence_down]))   

            if(len(predict_up[0])>7):
                temp = False
                for i in range(0,(len(predict_up[0]) - 7),7):
                    UP = predict_up[0]
                    DOWN = predict_down[0]
                    UP = UP[i:7+i]
                    DOWN = DOWN[i:7+i]
                    # print(UP)     
                    if((np.sum(UP) < Threshold_value_up) and (np.sum(DOWN) < Threshold_value_down)):
                        labels.append(1)
                        temp = True
                        break
                if(temp == False):
                    labels.append(0)
                
            else: 
                if((np.sum(predict_up) < Threshold_value_up) and (np.sum(predict_down) < Threshold_value_down)):
                    labels.append(1)
                else:
                    labels.append(0)
            
            count+=1
            if(count>=10):
                break

        return np.array(labels)
    
    def accuaracy(self, predict, test):
        labels = [1,0]
        acc = accuracy_score(test, predict)
        precision = precision_score(test, predict, labels=labels)
        recal = recall_score(test, predict, labels=labels)
        f1= f1_score(test, predict, labels=labels)
        result = [round(acc, 4), round(precision, 4), round(recal, 4), round(f1, 4)]
        c_matrix = confusion_matrix(test, predict, labels=labels)
        return result, c_matrix
    
    def to_csv(self, List, csv_file, list_name):
            if not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(list_name)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(List)

    def filter(self, data):
        def evaluate_performance(data, window_length, polyorder):
            smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
            mse = mean_squared_error(data, smoothed_data)
            return mse

        # Tạo lưới giá trị để thử nghiệm
        param_grid = {
            'window_length': range(30, 300, 2),  # Các giá trị lẻ từ 5 đến 21
            'polyorder': range(1, 10)  # Bậc đa thức từ 1 đến 4
        }

        # Sử dụng tối ưu hóa di động để chọn tham số tối ưu
        param_list = list(ParameterSampler(param_grid, n_iter=90, random_state=42))
        best_params = None
        best_mse = float('inf')

        for params in param_list:
            mse = evaluate_performance(data, window_length=params['window_length'], polyorder=params['polyorder'])
            if mse < best_mse:
                best_mse = mse
                best_params = params

        smoothed_data = savgol_filter(data, window_length=best_params['window_length'], polyorder=best_params['polyorder'])
        # plt.plot(data, label='Dữ liệu gốc', linestyle='dashed')
        # plt.plot(smoothed_data, label='Dữ liệu sau khi được làm mịn', linestyle='solid')
        # plt.legend()
        # plt.show()
        return smoothed_data
    

#################################################################################################################################################################################################################################################################################################################################
class LSTMModel:
    def __init__(self,look_forward, look_back, slidingWindow, epoch=10, batch_size=128):
        self.look_forward = look_forward
        self.look_back = look_back
        self.slidingWindow = slidingWindow
        self.epoch = epoch
        self.batch_size = batch_size

    def processData(self, thanhly=None, luuluong=None, evaluate=True):
        # xóa Nan
        luuluong=luuluong.dropna()
        # Chuyển đổi cột NGAY sang định dạng datetime
        try:
            luuluong.loc[:,'NGAY'] = pd.to_datetime(luuluong['NGAY'], format='%d-%b-%y')
        except:
            luuluong.loc[:,'NGAY'] = pd.to_datetime(luuluong['NGAY'], format='%m/%d/%Y')

        # xóa Khách hàng có lích sử ít hơn 30 ngày
        # # Tính tổng UP và DOWN theo từng KHACHHANG_ID và từng NGAY
        luuluong = luuluong.groupby(['KHACHHANG_ID', 'NGAY']).agg({'UP': 'sum', 'DOWN': 'sum'}).reset_index()
        count_days = luuluong.groupby("KHACHHANG_ID")['NGAY'].count()
        Threshold = count_days >=30
        luuluong = luuluong[luuluong["KHACHHANG_ID"].isin(count_days[Threshold].index)]
        # print(len(luuluong["KHACHHANG_ID"].unique()))
        # Sắp xếp dữ liệu 
        luuluong = luuluong.sort_values(by=['KHACHHANG_ID','NGAY'])
        # làm đầy dữ liệu.
        new_data = pd.DataFrame()
        for id in luuluong['KHACHHANG_ID'].unique():
            customer = luuluong[luuluong['KHACHHANG_ID']==id]
            date_list = customer["NGAY"].unique()
            start = date_list.min()
            end = date_list.max()
            date_range = pd.date_range(start, end)
            if(new_data.empty):
                new_data = pd.DataFrame([(id, date) for date in date_range], columns=['KHACHHANG_ID','NGAY'])
                continue
            current = pd.DataFrame([(id, date) for date in date_range], columns=['KHACHHANG_ID','NGAY'])
            new_data = pd.concat([new_data, current], ignore_index=True)
        
        data = luuluong.merge(new_data[['KHACHHANG_ID','NGAY']], on=['KHACHHANG_ID', 'NGAY'], how='right').fillna({'UP':0,'DOWN':0})
        # # print(data['KHACHHANG_ID'].value_counts())
        # gộp 2 file csv 
        if(evaluate==True): 
            try:
                thanhly['NGAY'] = pd.to_datetime(thanhly['NGAY'], format='%d-%b-%y')
            except:
                thanhly['NGAY'] = pd.to_datetime(thanhly['NGAY'], format='%m/%d/%Y')

            thanhly = thanhly.sort_values(by=['KHACHHANG_ID'])
            # loại bỏ đối tượng trùng lặp.
            thanhly = thanhly.drop_duplicates(subset="KHACHHANG_ID", keep='first')
            data = data.merge(thanhly, on='KHACHHANG_ID', how='left')
            data = data.rename(columns = {'KHACHHANG_ID':'KHACHHANG_ID', 'NGAY_x':'NGAY', 'UP':'UP', 'DOWN':'DOWN','NGAY_y':'NGAYTHANHLY','THANHLY':'THANHLY'})
            data['UP'] = data['UP'].apply(lambda x: x / (1024 ** 3) if x != 0 else 0)
            data['DOWN'] = data['DOWN'].apply(lambda x: x / (1024 ** 3) if x != 0 else 0)

        return data
    

    def createSlidingWindow(self, customer, key, evaluate=True):
        sequences = []
        targets = []
        if(evaluate==True):
            if(len(customer) != (self.look_back+self.look_forward)):
                for i in range(0,(len(customer)-self.look_back-self.look_forward), self.slidingWindow):
                    sequences.append(customer[key].values[i:i+self.look_back])
                    targets.append(customer[key].values[i+self.look_back:i+self.look_back+self.look_forward])
            else:
                sequences.append(customer[key].values[:self.look_back])
                targets.append(customer[key].values[self.look_back:self.look_back+self.look_forward])
            sequences = np.array(sequences)
            targets = np.array(targets)
            sequences = sequences.reshape((sequences.shape[0], sequences.shape[1],1))
            targets = targets.reshape((targets.shape[0],targets.shape[1],1))
            return sequences, targets
        
        else:
            if(len(customer) != (self.look_back)):
                for i in range(0,(len(customer)-self.look_back-self.look_forward), self.slidingWindow):
                    sequences.append(customer[key].values[i:i+self.look_back])
                else:
                    sequences.append(customer[key].values[:self.look_back])
                sequences = np.array(sequences)
                sequences = sequences.reshape((sequences.shape[0], sequences.shape[1],1))
                return sequences


    def buildModel(self, model=None):
        if (model is None):
            model = keras.Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.look_back, 1)),
                LSTM(64, return_sequences=True),
                Dropout(0.35),
                LSTM(128, return_sequences=True),
                LSTM(64, return_sequences=True),
                Dropout(0.35),
                LSTM(32),
                Dropout(0.35),
                Dense(self.look_forward, activation='relu')
            ])

            # keras.utils.plot_model(model,to_file="BRNN1.png",show_shapes=True, show_layer_names=True, show_layer_activations=True)
            model.compile(optimizer='adam', loss='mae')
        return model

    def train_split(self, data, key):
        list_id = data['KHACHHANG_ID'].unique()
        sequences = np.empty((0,self.look_back,1))
        targets = np.empty((0,self.look_forward,1))
        for id in list_id:
            customer = data[data['KHACHHANG_ID']==id]
            sequence, target = self.createSlidingWindow(customer, key=key)
            sequences = np.concatenate((sequences, sequence), axis=0)
            targets = np.concatenate((targets, target), axis=0)
        return sequences, targets
    
    def fit(self, sequences, targets, model=None):
        model = self.buildModel(model)
        model.fit(sequences, targets, epochs=self.epoch, batch_size=self.batch_size, validation_split=0.2)
        return model
    
    def Threshold(self, data, key):
        customer_thanhly = data[data['THANHLY']==1]
        # customer_thanhly = customer_thanhly.groupby(['KHACHHANG_ID']).agg({key: 'sum'})
        Threshold = []
        for id in customer_thanhly["KHACHHANG_ID"].unique():
            customer = customer_thanhly[customer_thanhly["KHACHHANG_ID"]==id]
            customer = customer.tail(30)
            Threshold.append(np.sum(customer[key]))
        return np.mean(Threshold)
    
    def smean_absolute_error(self, model_up, model_down, data):
            sequences_up, targets_up = self.train_split(data, "UP")
            sequences_down, targets_down = self.train_split(data, "DOWN")
            loss_up = model_up.evaluate(sequences_up, targets_up)
            loss_down = model_down.evaluate(sequences_down, targets_down)
            return np.round(loss_up,4), np.round(loss_down,4)
    
    def predict_labels(self, model_up, model_down, data, Threshold_value_up, Threshold_value_down):

        list_id = data['KHACHHANG_ID'].unique()
        labels = []
        count = 0
        for id in list_id:
            customer = data[data['KHACHHANG_ID']==id]
            sequences_up = self.createSlidingWindow(customer, 'UP', evaluate=False)
            sequences_down= self.createSlidingWindow(customer, 'DOWN', evaluate=False)
            ##########################################################################
            sequence_up = sequences_up[-1]
            sequence_down = sequences_down[-1]
            ###########################################################################
            predict_up=model_up.predict(np.array([sequence_up]))
            predict_down=model_down.predict(np.array([sequence_down]))   
            for i in range(len(predict_up[0])):
                UP = predict_up[0]
                DOWN = predict_down[0]
                UP = UP[i:self.look_forward+i]
                DOWN = DOWN[i:self.look_forward+i]
                print(UP)     
                if((np.sum(UP) < Threshold_value_up) and (np.sum(DOWN) < Threshold_value_down)):
                    labels.append(1)
                    break
            else: 
                labels.append(0)
            
            count+=1
            if(count>=10):
                break

        return np.array(labels)
    
    def accuaracy(self, predict, test):
        labels = [1,0]
        acc = accuracy_score(test, predict)
        precision = precision_score(test, predict, labels=labels)
        recal = recall_score(test, predict, labels=labels)
        f1= f1_score(test, predict, labels=labels)
        result = [round(acc, 4), round(precision, 4), round(recal, 4), round(f1, 4)]
        c_matrix = confusion_matrix(test, predict, labels=labels)
        return result, c_matrix
    
    def to_csv(self, List, csv_file, list_name):
            if not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(list_name)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(List)

    def filter(self, data):
        def evaluate_performance(data, window_length, polyorder):
            smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
            mse = mean_squared_error(data, smoothed_data)
            return mse

        # Tạo lưới giá trị để thử nghiệm
        param_grid = {
            'window_length': range(30, 300, 2),  # Các giá trị lẻ từ 5 đến 21
            'polyorder': range(1, 10)  # Bậc đa thức từ 1 đến 4
        }

        # Sử dụng tối ưu hóa di động để chọn tham số tối ưu
        param_list = list(ParameterSampler(param_grid, n_iter=90, random_state=42))
        best_params = None
        best_mse = float('inf')

        for params in param_list:
            mse = evaluate_performance(data, window_length=params['window_length'], polyorder=params['polyorder'])
            if mse < best_mse:
                best_mse = mse
                best_params = params

        smoothed_data = savgol_filter(data, window_length=best_params['window_length'], polyorder=best_params['polyorder'])
        # plt.plot(data, label='Dữ liệu gốc', linestyle='dashed')
        # plt.plot(smoothed_data, label='Dữ liệu sau khi được làm mịn', linestyle='solid')
        # plt.legend()
        # plt.show()
        return smoothed_data


#############################################################################################################

class DecisionTree:
    # def __init__(self):

    def processData(self, data, evaluate=True):
        
        data.drop('MAKHACHHANG', axis = 'columns', inplace=True)
        data.drop('PHUONGXA', axis = 'columns', inplace=True)
        data['LOAIDICHVU'].replace({'Fiber': 1, 'MyTV': 0}, inplace = True)
        # data.corr()['THANHLY'].sort_values(ascending=False)
        data = pd.get_dummies(data=data, columns=['QUANHUYEN'])
        ################################################
        cols_to_scale = ['GIADICHVU']
        scaler = MinMaxScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        #######################################
        if(evaluate==True):
            sm = SMOTE()
            dataset_dummy = pd.get_dummies(data, drop_first=True)

            X = dataset_dummy.drop(["THANHLY"],axis=1)
            y = dataset_dummy['THANHLY']

            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        else:
            try:
                data.drop("THANHLY", axis='columns', inplace=True)
                return data
            except:
                return data
    
    def buildModel(self, model=None):
        if (model is None):
            model = DecisionTreeClassifier()   
        return model

    def fit(self, X, y, model = None):   
        model = self.buildModel(model)
        model.fit(X, y)
        return model

    def evaluate(self, model, X, y):
        total_test_score = 0.0;
        total_precision_score = 0.0;
        total_recal_score = 0.0;
        total_f1_score = 0.0;
        y_pred = model.predict(X)
        total_test_score += model.score(X, y)
        total_precision_score += precision_score(y, y_pred)
        total_recal_score += recall_score(y, y_pred)
        total_f1_score += f1_score(y, y_pred, labels=[1,0])
        result = [round(total_test_score, 4), round(total_precision_score, 4), round(total_recal_score, 4), round(total_f1_score, 4)]
        c_matrix = confusion_matrix(y_pred, y, labels=[1, 0])

        # x,y = np.unique(y_pred, return_counts=True)
        # print(x,y)
        # print(c_matrix)
        return result, c_matrix

    def predict(self, model ,data):
        y_pred = model.predict(data)
        return y_pred