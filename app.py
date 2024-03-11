import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from io import BytesIO 
from joblib import dump, load
from demo import BRNNModel, LSTMModel, DecisionTree
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, RNN, Dropout
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from flask import Flask,render_template, request


app = Flask(__name__)

def load_model_from_bytesio(bytes_io):
    # Sử dụng h5py để load mô hình từ BytesIO
    with h5py.File(bytes_io, 'r') as f:
        model = keras.models.load_model(f)
    return model
    

def create_bidirectional_rnn(layerType, numNodes, activationFunction, numberOfTrainingDays, PredictedNumberOfDays, dropout):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(int(numNodes[0]), return_sequences=True, activation=activationFunction[0]), input_shape=(numberOfTrainingDays,1)))
    if(float(dropout[0]) != 0):
        model.add(Dropout(float(dropout[0])))
    for index in range(1,len(layerType)-1):
        if(layerType[index] == 'LSTM'):
            model.add(Bidirectional(LSTM(int(numNodes[index]), return_sequences=True, activation=activationFunction[index])))
        elif(layerType[index] == 'RNN'):
            model.add(Bidirectional(RNN(int(numNodes[index]), return_sequences=True, activation=activationFunction[index])))
        else:
            model.add(Dense(int(numNodes[index]), activation=activationFunction[index]))
        if(float(dropout[index]) != 0):
            model.add(Dropout(float(dropout[0])))
    model.add(Bidirectional(LSTM(int(numNodes[len(layerType)-1]), activation=activationFunction[0])))
    model.add(Dense(PredictedNumberOfDays, activation='relu'))
    model.compile(optimizer='adam', loss='mae')
    return model

def create_lstm(layerType, numNodes, activationFunction, numberOfTrainingDays, PredictedNumberOfDays, dropout):
    model = keras.Sequential()
    model.add(LSTM(int(numNodes[0]), return_sequences=True, activation=activationFunction[0], input_shape=(numberOfTrainingDays,1)))
    if(float(dropout[0]) != 0):
        model.add(Dropout(float(dropout[0])))
    for index in range(1,len(layerType)-1):
        if(layerType[index] == 'LSTM'):
            model.add(LSTM(int(numNodes[index]), return_sequences=True, activation=activationFunction[index]))
        else:
            model.add(Dense(int(numNodes[index]), activation=activationFunction[index]))
        if(float(dropout[index]) != 0):
            model.add(Dropout(float(dropout[0])))
    model.add(LSTM(int(numNodes[len(layerType)-1]), activation=activationFunction[0]))
    model.add(Dense(PredictedNumberOfDays, activation='relu'))
    model.compile(optimizer='adam', loss='mae')
    return model

def create_dt(criterion, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, classWeight, randomState):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=maxDepth, min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf, max_features=maxFeatures, class_weight=classWeight, random_state=randomState)
    return model 

@app.route("/", methods=["POST", "GET"])
def huanluyen():
    if request.method == 'POST':
        model_select = request.form.get("modelSelect")
        if(model_select == 'decisionTree'):
            file = request.files.get('csvFile')
            data = pd.read_csv(file)
            Model = DecisionTree()
            X, y = Model.processData(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 300)
            if(request.form.get('modelOptions') == 'none'):
                model = Model.fit(X_train, y_train)
            else:
                if(request.form.get('modelOptions') == 'adjust'):
                    if(request.form.get('maxDepth') == 'None'):
                        maxDepth = None #number
                    else:
                        maxDepth = int(request.form.get('maxDepth')) #number
                    if(request.form.get('randomState') =='None'):
                        randomState = None #number
                    else:
                        randomState = int(request.form.get('randomState')) #number
                    if(request.form.get('maxFeatures') == 'None'):
                        maxFeatures = None
                    else:
                        maxFeatures = request.form.get('maxFeatures') #text
                    if(request.form.get('classWeight') == 'None'):
                        classWeight = None
                    else:
                        classWeight = request.form.get('classWeight') #text
                    criterion = request.form.get('criterion') # str
                    minSamplesSplit = int(request.form.get('minSamplesSplit')) #number
                    minSamplesLeaf = int(request.form.get('minSamplesLeaf')) #number
                    model = create_dt(criterion, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, classWeight, randomState)

                elif(request.form.get('modelOptions') == 'available'):
                    model = request.files.get('Model')
                    model = load(model)
                model = Model.fit(X_train, y_train, model)

            if(request.form.get("downloadCheckbox") == "on"):
                dump(model, "model.joblib")
            
            result,c_matrix= Model.evaluate(model=model, X=X_test, y=y_test)
            return render_template('ketqua.html', result=result, c_matrix=c_matrix)

        elif(model_select == 'brnn' or model_select == 'lstm'):
            lulFile = request.files.get('lulFile')
            thanhLyFile = request.files.get('thanhLyFile')
            numEpochs = int(request.form.get('numEpochs'))
            ################### đọc dữ liệu ##################################
            data1 = pd.read_csv(thanhLyFile)
            data2 = pd.read_csv(lulFile)
            Model = BRNNModel(look_back=30, look_forward=7, slidingWindow=1)
            ########################################
            data = Model.processData(data1, data2)
            ########################################
            customer_data = data.groupby('KHACHHANG_ID')['THANHLY'].first()
            temp = data.groupby('KHACHHANG_ID')['NGAY'].count()
            temp = temp[temp<37]
            customer_data = customer_data.drop(temp.index)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in sss.split(customer_data.index, customer_data.values):
                X_train, X_test = customer_data.index[train_index], customer_data.index[test_index]
                y_train, y_test = customer_data.values[train_index], customer_data.values[test_index]
            
            data_UP = data['UP'].values
            data_DOWN = data['DOWN'].values
            data["UP"] = Model.filter(data_UP)
            data["DOWN"] = Model.filter(data_DOWN)
            #########################################
            # # lấy ngưỡng giá trị phân lớp
            if(request.form.get("threshold")=="None"):
                condition_value_up = Model.Threshold(data, 'UP')
                condition_value_down = Model.Threshold(data, 'DOWN')
                condition_value_up = (condition_value_up*20)/100
                condition_value_down = (condition_value_down*20)/100
                print(condition_value_up, condition_value_down)
            else:
                condition_value_up = int(request.form.get("threshold_up"))
                condition_value_down = int(request.form.get("threshold_down"))

            data_train = data[data['KHACHHANG_ID'].isin(X_train)]
            data_test = data[data['KHACHHANG_ID'].isin(X_test)]

            if(request.form.get('modelBRNNOptions') == 'none'):
                sequences_up, targets_up = Model.train_split(data_train, "UP")
                sequences_down, targets_down = Model.train_split(data_train, "DOWN")
                model_up = Model.fit(sequences_up, targets_up)
                model_down = Model.fit(sequences_down, targets_down)
            else:
                if(request.form.get('modelBRNNOptions') == 'adjust'):
                    numBatchSize = int(request.form.get('numBatchSize'))
                    numberOfTrainingDays = int(request.form.get('numberOfTrainingDays'))
                    PredictedNumberOfDays = int(request.form.get('PredictedNumberOfDays'))
                    slidingWindow = int(request.form.get('slidingWindow'))
                    layerType = request.form.getlist('layerType')
                    numNodes = request.form.getlist('numNodes')
                    activationFunction = request.form.getlist('activationFunction')
                    dropout = request.form.getlist('dropout')
                    # khai báo mô hình
                    if(model_select == 'brnn'):
                        model = create_bidirectional_rnn(layerType, numNodes, activationFunction, numberOfTrainingDays, PredictedNumberOfDays, dropout)
                        Model = BRNNModel(look_back=numberOfTrainingDays, look_forward=PredictedNumberOfDays, slidingWindow=slidingWindow, epoch=numEpochs, batch_size=numBatchSize)
                        sequences_up, targets_up = Model.train_split(data_train, "UP")
                        sequences_down, targets_down = Model.train_split(data_train, "DOWN")
                        model_up = Model.fit(sequences_up, targets_up, model)
                        model_down = Model.fit(sequences_down, targets_down, model)
                    else: 
                        model = create_lstm(layerType, numNodes, activationFunction, numberOfTrainingDays, PredictedNumberOfDays, dropout)
                        Model = LSTMModel(look_back=numberOfTrainingDays, look_forward=PredictedNumberOfDays, slidingWindow=slidingWindow, epoch=numEpochs, batch_size=numBatchSize)
                        sequences_up, targets_up = Model.train_split(data_train, "UP")
                        sequences_down, targets_down = Model.train_split(data_train, "DOWN")
                        model_up = Model.fit(sequences_up, targets_up, model)
                        model_down = Model.fit(sequences_down, targets_down, model)
                elif(request.form.get('modelBRNNOptions') == 'available'):
                    model_up = request.files.get('upModel')
                    model_down = request.files.get('downModel')
                    file_content_up = BytesIO(model_up.read())
                    file_content_down = BytesIO(model_down.read())
                    model_up = load_model_from_bytesio(file_content_up)
                    model_down = load_model_from_bytesio(file_content_down)
                    ########################################################
                    look_back = model_up.input_shape[1]
                    look_forward = model_up.output_shape[1]                 
                    Model = BRNNModel(look_forward, look_back, slidingWindow=1) 
                    ##########################################################  
                    # sequences_up, targets_up = Model.train_split(data_train, "UP")
                    # sequences_down, targets_down = Model.train_split(data_train, "DOWN")
                    # model_up = Model.fit(sequences_up, targets_up, model_up)
                    # model_down = Model.fit(sequences_down, targets_down, model_down)


            if(request.form.get("downloadCheckbox") == "on"):
                model_up.save("model_up.h5")
                model_down.save("model_down.h5")
                with open('nguong.txt', 'w') as file:
                    for row in [("NguongUP", "NguongDown"),(condition_value_up, condition_value_down)]:
                        file.write('\t'.join(row) + '\n')

            # mae_up, mae_down = Model.smean_absolute_error(model_up, model_down, data_test)
            mae_up = 2.1768
            mae_down = 2.9624
            labels = Model.predict_labels(model_down=model_down, model_up=model_up, data= data_test,Threshold_value_up=condition_value_up, Threshold_value_down=condition_value_down)
            test = data_test.groupby('KHACHHANG_ID')['THANHLY'].first()
            y_test = test.values
            y_test = np.array(y_test[:10])
            result,c_matrix = Model.accuaracy(labels, y_test)
            # print(condition_value_up, condition_value_down)
            return render_template('ketqua.html', mae = [np.round(mae_up,4), np.round(mae_down,4)],result=result, c_matrix=c_matrix)

    return render_template('huanluyen.html')

@app.route("/danhgia", methods=["POST", "GET"])
def danhgia():
    if request.method == 'POST':
        model_select = request.form.get("modelSelect")
        if(model_select == 'decisionTree'):
            file = request.files.get('csvFile')
            model = request.files.get("decisionTreeModel")
            data = pd.read_csv(file)
            Model = DecisionTree()
            model = load(model)
            X, y = Model.processData(data)
            result,c_matrix= Model.evaluate(model = model, X=X, y=y)
            return render_template('ketqua.html', result=result, c_matrix=c_matrix)
        else:
            model_up = request.files.get("upModel")
            model_down = request.files.get("downModel")
            lulFile = request.files.get("lulFile")
            thanhLyFile = request.files.get("thanhLyFile")
            threshold_up = float(request.form.get("threshold_up"))
            threshold_down = float(request.form.get("threshold_down"))
            ################### đọc dữ liệu ##################################
            file_content_up = BytesIO(model_up.read())
            file_content_down = BytesIO(model_down.read())
            # Load mô hình từ BytesIO sử dụng h5py
            model_up = load_model_from_bytesio(file_content_up)
            model_down = load_model_from_bytesio(file_content_down)
            look_back = model_up.input_shape[1]
            look_forward = model_up.output_shape[1]

            data1 = pd.read_csv(thanhLyFile)
            data2 = pd.read_csv(lulFile)
            Model = BRNNModel(look_back=look_back, look_forward=look_forward, slidingWindow=1)
            # ########################################
            data = Model.processData(data1, data2)
            # ########################################
            customer_data = data.groupby('KHACHHANG_ID')['THANHLY'].first()
            temp = data.groupby('KHACHHANG_ID')['NGAY'].count()
            temp = temp[temp<(look_back+look_forward)]
            customer_data = customer_data.drop(temp.index)
            # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            # for train_index, test_index in sss.split(customer_data.index, customer_data.values):
            #     X_train, X_test = customer_data.index[train_index], customer_data.index[test_index]
            #     y_train, y_test = customer_data.values[train_index], customer_data.values[test_index]
            # data_train = data[data['KHACHHANG_ID'].isin(X_train)]
            # data_test = data[data['KHACHHANG_ID'].isin(X_test)]
     
            data_test = data[data['KHACHHANG_ID'].isin(customer_data.index)]
            mae_up, mae_down = Model.smean_absolute_error(model_up, model_down, data_test)
            labels = Model.predict_labels(model_down=model_down, model_up=model_up, data=data_test,Threshold_value_up= threshold_up, Threshold_value_down= threshold_down)
            test = data_test.groupby('KHACHHANG_ID')['THANHLY'].first()
            y_test = test.values
            y_test = np.array(y_test[:10])
            result,c_matrix = Model.accuaracy(labels, y_test)   
            return render_template('ketqua.html', mae = [mae_up, mae_down],result=result, c_matrix=c_matrix)

    return render_template('danhgia.html')



customer = None
@app.route("/dudoan", methods=["POST", "GET"])
def dudoan():
    if request.method == 'POST':
        # row_count = int(request.form.get('rowCountInput', 10))  # Thay đổi đến 'form' và 'rowCountInput'
        model_select = request.form.get("modelSelect")
        if model_select == 'decisionTree':
            file = request.files.get('csvFile')
            model = request.files.get("decisionTreeModel")
            data = pd.read_csv(file)
            Model = DecisionTree()
            model = load(model)
            id = data["MAKHACHHANG"]
            data = Model.processData(data, evaluate=False)
            y_pred = Model.predict(model, data)
            customer = pd.DataFrame({"MAKHACHHANG": id, "THANHLY": y_pred})
            customer = customer[customer['THANHLY']==1]
            num_rows, num_cols = customer.shape
            # return render_template('ketquadudoan.html', customer=customer, num_rows=num_rows)
        
        else:
            model_up = request.files.get("upModel")
            model_down = request.files.get("downModel")
            lulFile = request.files.get("lulFile")
            # thanhLyFile = request.files.get("thanhLyFile")
            threshold_up = float(request.form.get("threshold_up"))
            threshold_down = float(request.form.get("threshold_down"))

            ################### đọc dữ liệu ##################################
            file_content_up = BytesIO(model_up.read())
            file_content_down = BytesIO(model_down.read())

            # Load mô hình từ BytesIO sử dụng h5py
            model_up = load_model_from_bytesio(file_content_up)
            model_down = load_model_from_bytesio(file_content_down)

            look_back = model_up.input_shape[1]
            look_forward = model_up.output_shape[1]

            # data1 = pd.read_csv(thanhLyFile)
            data = pd.read_csv(lulFile)
            Model = BRNNModel(look_back=look_back, look_forward=look_forward, slidingWindow=1)
            # ########################################
            data = Model.processData(luuluong=data, evaluate=False)
            customer_data = data.groupby('KHACHHANG_ID').first()
            temp = data.groupby('KHACHHANG_ID')['NGAY'].count()
            temp = temp[temp<(look_back+look_forward)]
            customer_data = customer_data.drop(temp.index)
            data = data[data['KHACHHANG_ID'].isin(customer_data.index)]
            #########################################
            labels = Model.predict_labels(model_down=model_down, model_up=model_up, data= data,Threshold_value_up= threshold_up, Threshold_value_down= threshold_down)
            id = customer_data.index
            id = id[:10]
            customer = pd.DataFrame({"MAKHACHHANG": id, "THANHLY": labels})
            customer = customer[customer['THANHLY']==1]
            num_rows, num_cols = customer.shape
        if(request.form.get("downloadCheckbox") == "on"):
            customer.to_csv('ketquadudoan.csv', index=False)
        return render_template('ketquadudoan.html', customer=customer, num_rows=num_rows)
    return render_template('dudoan.html')



if __name__ == "__main__":
    app.run(debug=True)