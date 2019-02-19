# -*- coding:utf-8 -*-


import sys
import os
import cv2
import argparse
import numpy as np
from keras.layers import *
from  keras.models import *
from keras.optimizers import *
from keras.utils import np_utils
# from trainmonitor import TrainingMonitor
from sklearn.model_selection import train_test_split
# from jsonrpc import  Server
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# from sklearn.utils import shuffle
from PIL import Image
from  tqdm import tqdm
#from jsonrpcclient import request

# from keras.applications import VGG16
# from trainmonitor import TrainingMonitor
# from trainingmonitor import TrainingMonitor
# import  matplotlib.pyplot as plt
# from keras.callbacks import TensorBoard
# import  pyjsonrpc
#

def preprocessing_img(img_src, width, height, categorical_num, validate_percent, test_percent):
    X, y = [], []

    for i, folder in enumerate(os.listdir(img_src)):
        print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(img_src, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(img_src, folder, filename)
            print(index, imgFile)
            img = cv2.resize(cv2.imread(imgFile), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(i)
    X = np.array(X).astype('float32') / 255
    y = np_utils.to_categorical(y, categorical_num)
    # y = np_utils.to_categorical(y).reshape(-1, categorical_num)
    print(X.shape)
    print(y.shape)

    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validate_percent, shuffle=True)
    print('preprocessing images is OK..')
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_src', type=str, nargs='+', help='train image path')
    ap.add_argument('-width', type=int, nargs='+', help='img_resize_width')
    ap.add_argument('-height', type=int, nargs='+', help='img__resize_height')
    ap.add_argument('-test_percent', type=float, nargs='+', help='test_dataset_percent')
    ap.add_argument('-validata_percent', type=float, nargs='+', help='validata_dataset_percent')
    ap.add_argument('-channel_num', type=int, nargs='+', help='channels_num')
    ap.add_argument('-num_classes', type=int, nargs='+', help='num_classes')

    ap.add_argument('-batch_size', type=int, nargs='+', help='batch_size')
    ap.add_argument('-lr', type=float, nargs='+', help='learning_rate')
    ap.add_argument('-epochs', type=int, nargs='+', help='epochs')
    # ap.add_argument('data_augmentation', type=bool, nargs='+', help='data_augmentation')
    ap.add_argument('-save_dir', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model h5  file')
    ap.add_argument('-jsonrpcMlClientPoint', type=str,help='IP address of server endpoint...')
    ap.add_argument('-model_name', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model file')
    ap.add_argument('-model_id', type=str, help="define the unique model identifier id")
    ap.add_argument('-model_userid', type=str, help="define the model user who are")
    ap.add_argument('-model_version', type=str, help="define the model version is what")
    ap.add_argument('-user_Optimizer', type=str, nargs='+', help="user selective optimizers")
    ap.add_argument('-ams_id', type=str, nargs='+', help="callback the trained model id")

    return vars(ap.parse_args())

# keras.backend.clear_session()


def build_model(width, height, channel_num, num_classes, UserOptimizer='adam'):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channel_num)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    all_optimizers = {'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      'rmsprop': RMSprop(lr=0.001, rho=0.9, epsilon=1e-6),
                      'adagrad': Adagrad(lr=0.01, decay=1e-6),
                      'adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=1e-6),
                      'adam': Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                      # 'adamax': Adamax,
                      # 'nadam': Nadam,
                      # 'tfoptimizer': TFOptimizer,
                      }

    UserOptimizer = all_optimizers[user_optimizer]
    print("use define optimier is ", user_optimizer)

    if (int(num_classes >= 2)):
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
    return model

# 定义计算模型评估指标
def metrics_result(y_label, y_pred):
    _recall = recall_score(y_label, y_pred, average='weighted')
    _precision = precision_score(y_label, y_pred, average='weighted')
    _f1 = f1_score(y_label, y_pred, average='macro')
    _acc = accuracy_score(y_label, y_pred)
    _cm = confusion_matrix(y_label, y_pred)
    return _recall, _precision, _f1, _acc, _cm


if __name__ == '__main__':

    # 参数动态传递
    arguments = parse_arguments(sys.argv[1:])
    print(type(arguments))
    img_src = arguments['img_src'][0]
    width = arguments['width'][0]
    height = arguments['height'][0]
    test_percent = arguments['test_percent'][0]
    validata_percent = arguments['validata_percent'][0]
    channel_num = arguments['channel_num'][0]
    num_classes = arguments['num_classes'][0]
    batch_size = arguments['batch_size'][0]
    lr = arguments['lr'][0]
    epochs = arguments['epochs'][0]
    save_dir = arguments['save_dir']
    model_name = arguments['model_name']
    modelId = arguments["model_id"]
    user_optimizer = arguments['user_Optimizer'][0]
    #pretrain_model_name = arguments['pretrain_model'][0]
    model_version = arguments['model_version']
    model_userid = arguments['model_userid']
    ams_id = arguments['ams_id'][0]
    jsonrpcMlClientPoint= arguments['jsonrpcMlClientPoint']
    print('ip address is ',jsonrpcMlClientPoint)


    # 数据集预处理
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing_img(img_src, width, height, num_classes,
                                                                           validata_percent, test_percent)
    #
    # model = build_model(width,height,channel_num,num_classes)
    # 模型构建
    model = build_model(width, height, channel_num, num_classes, user_optimizer)

    print(model.summary())

    # 使用tensorfboard实时监控
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
    #                     shuffle=True, callbacks=[tensorboard])



    # http_client = pyjsonrpc.HttpClient(
    #     url="http://192.168.10.141:8080/rpc/myservice"
    # )
    #http_client = Server('http://192.168.10.141:8080/rpc/myservice')



    # http_client = Server(jsonrpcMlClientPoint)
    # print('client is : ',http_client)
    # # 训练可视化，返回val_acc, val_loss, train_acc, train_loss
    # callbacks = [TrainingMonitor(http_client=http_client, model_id=modelId, model_userid=model_userid,
    #                              model_version=model_version, ams_id=ams_id)]

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        # callbacks=callbacks,
                        validation_split=validata_percent, verbose=1)




    train_y_pred = np.argmax(np.asarray(model.predict(X_train)), axis=1)
    valid_y_pred = np.argmax(np.asarray(model.predict(X_valid)), axis=1)
    test_y_pred = np.argmax(np.asarray(model.predict(X_test)), axis=1)

    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 训练集模型评估结果
    metric_train = metrics_result(y_train, train_y_pred)
    print(type(metric_train), metric_train)

    metric_valid = metrics_result(y_valid, valid_y_pred)
    print(type(metric_valid), metric_valid)
    metric_test = metrics_result(y_test, test_y_pred)
    print(type(metric_test), metric_test)

    # 调用metric_result 方法返回的是tuple： 元素顺序为：  _recall, _precision, _f1, _acc, _cm
    call_res = {'model_id': modelId, 'model_userid': model_userid, 'model_version': model_version, 'ams_id': ams_id,
                'calculationType': 'result',
                'train_recall_score': metric_train[0], 'train_precision_score': metric_train[1],
                'train_f1_score': metric_train[2], 'train_acc': metric_train[3], 'train_cm': metric_train[4].tolist(),
                'valid_recall_score': metric_valid[0], 'valid_precision_score': metric_valid[1],
                'valid_f1_score': metric_valid[2], 'valid_acc': metric_valid[3], 'valid_cm': metric_valid[4].tolist(),
                'test_recall_score': metric_test[0], 'test_precision_score': metric_test[1],
                'test_f1_score': metric_test[2], 'test_acc': metric_test[3], 'test_cm': metric_test[4].tolist()}
    print(call_res)


    # 回调，向服务端发送评估指标

    # response = http_client.modelTrain(str(call_res))

    # response = request("http://192.168.10.141:8080/rpc/myservice", "modelTrain", str(call_res))

    # http_client.call("sayHelloWorld",call_res)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, model_name) + '.h5', overwrite=True)

    print('\r\nmodel has been saved in  ', os.path.join(save_dir, model_name) + '.h5')
