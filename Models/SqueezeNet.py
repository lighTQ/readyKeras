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
# from trainmonitor import TrainingMonitor\
from  sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
# from jsonrpc import  Server
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
# import  keras as K
# config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
# session = tf.Session(config=config)
# K.set_session(session)

# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# from sklearn.utils import shuffle
# import  matplotlib.pyplot as plt
# from keras.callbacks import TensorBoard
#

def preprocessing_img(img_src, width, height, validate_percent, test_percent):
    X, y = [], []
    categorical_num = len(set(os.listdir(img_src)))
    for i, folder in enumerate(os.listdir(img_src)):
        print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(img_src, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(img_src, folder, filename)
            print(index, imgFile)
            image = cv2.imread(imgFile)
            if isinstance(image, np.ndarray):
                pass
            else:
                continue
            img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(i)
    X = np.array(X).astype('float32') / 255
    y_label = np_utils.to_categorical(y, categorical_num)
    # y = np_utils.to_categorical(y).reshape(-1, categorical_num)
    print(X.shape)
    print(y_label.shape)
    print(len(y))

    x_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=test_percent, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validate_percent, shuffle=True)
    print('preprocessing images is OK..')
    return X_train, X_valid, X_test, y_train, y_valid, y_test,y,categorical_num

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_src', type=str, help='train image path')
    ap.add_argument('-width', type=int, help='img_resize_width')
    ap.add_argument('-height', type=int, help='img__resize_height')
    ap.add_argument('-test_percent', type=float, help='test_dataset_percent')
    ap.add_argument('-validata_percent', type=float , help='validata_dataset_percent')
    ap.add_argument('-channel_num', type=int , help='channels_num')
    # ap.add_argument('-num_classes', type=int , help='num_classes')

    ap.add_argument('-batch_size', type=int , help='batch_size')
    ap.add_argument('-lr', type=float, help='learning_rate')
    ap.add_argument('-epochs', type=int, help='epochs')
    # ap.add_argument('data_augmentation', type=bool, , help='data_augmentation')
    ap.add_argument('-save_dir', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model h5  file')
    ap.add_argument('-jsonrpcMlClientPoint', type=str,help='IP address of server endpoint...')
    ap.add_argument('-model_name', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model file')
    ap.add_argument('-model_id', type=str, help="define the unique model identifier id")
    ap.add_argument('-model_userid', type=str, help="define the model user who are")
    ap.add_argument('-model_version', type=str, help="define the model version is what")
    ap.add_argument('-user_Optimizer', type=str , help="user selective optimizers")
    ap.add_argument('-ams_id', type=str , help="callback the trained model id")

    return vars(ap.parse_args())

# keras.backend.clear_session()


def build_model(width, height, channel_num, num_classes, user_optimizer='adam'):

    inputs = Input((width, height, channel_num))
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x= MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    x= Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x= Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    xout = Concatenate(axis=-1)([fire1_expan1, fire1_expan2])
    xsquee = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xout)
    xexpan1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire2_expan1, fire2_expan2])
    xsquee = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xout)
    xexpan1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire3_expan1, fire3_expan2])
    xl2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire3_out)
    xsquee = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xl2)
    xexpan1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire4_expan1, fire4_expan2])
    xsquee = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xout)
    xexpan1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire5_expan1, fire5_expan2])
    xsquee = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xout)
    xexpan1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire6_expan1, fire6_expan2])
    xsquee = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xout)
    xexpan1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire7_expan1, fire7_expan2])
    xl3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire7_out)
    xsquee = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xl3)
    xexpan1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xexpan2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(
    xsquee)
    xout = Concatenate(axis=-1)([fire8_expan1, fire8_expan2])
    x= Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire8_out)
    xGlobalAvgPool2D(data_format='channels_last')(conv2)
    x= Model(inputs=x, outputs=Gap)



    # model.add(Dense(num_classes, activation='softmax'))
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

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
    print("use define optimier is ", UserOptimizer)

    if (int(num_classes >= 2)):

        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, x)
        model.compile(optimizer=UserOptimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # model.add(Dense(num_classes, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
    else:

        x = Dense(num_classes, activation='sigmoid')(x)
        model = Model(inputs, x)
        model.compile(optimizer=UserOptimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # model.add(Dense(num_classes, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
    return model

# 定义计算模型评估指标
def metrics_result(y_label, y_pred):
    _recall = recall_score(y_label, y_pred, average='weighted')
    _precision = precision_score(y_label, y_pred, average='weighted')
    _f1 = f1_score(y_label, y_pred, average='macro')
    _acc = accuracy_score(y_label, y_pred)
    _cm = confusion_matrix(y_label, y_pred)
    return _recall, _precision, _f1, _acc, _cm



# 具体计算评估指标函数
def call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test,model):


    train_y_pred = np.argmax(np.asarray(model.predict(X_train)), axis=1)
    valid_y_pred = np.argmax(np.asarray(model.predict(X_valid)), axis=1)
    test_y_pred = np.argmax(np.asarray(model.predict(X_test)), axis=1)

    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 训练集模型评估结果
    metric_train = metrics_result(y_train, train_y_pred)
    metric_valid = metrics_result(y_valid, valid_y_pred)
    metric_test = metrics_result(y_test, test_y_pred)


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
    return  call_res

if __name__ == '__main__':

    # 参数动态传递
    arguments = parse_arguments(sys.argv[1:])
    print(type(arguments))
    img_src = arguments['img_src']
    width = arguments['width']
    height = arguments['height']
    test_percent = arguments['test_percent']
    validata_percent = arguments['validata_percent']
    channel_num = arguments['channel_num']
    # num_classes = arguments['num_classes']
    batch_size = arguments['batch_size']
    lr = arguments['lr']
    epochs = arguments['epochs']
    save_dir = arguments['save_dir']
    model_name = arguments['model_name']
    modelId = arguments["model_id"]
    user_optimizer = arguments['user_Optimizer']
    #pretrain_model_name = arguments['pretrain_model'][0]
    model_version = arguments['model_version']
    model_userid = arguments['model_userid']
    ams_id = arguments['ams_id']
    jsonrpcMlClientPoint= arguments['jsonrpcMlClientPoint']
    print('ip address is ',jsonrpcMlClientPoint)


    # 数据集预处理
    X_train, X_valid, X_test, y_train, y_valid, y_test, y, categorical_num= preprocessing_img(img_src, width, height,
                                                                           validata_percent, test_percent)
    #
    # model = build_model(width,height,channel_num,num_classes)
    # 模型构建
    model = build_model(width, height, channel_num, categorical_num, user_optimizer)
    print(model.summary())

    # print('type class:', y,np.unique(y))
    #
    # class_weights = compute_class_weight('balanced', np.unique(y),y)  # computing weights of different classes
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # filepath = os.path.join(save_dir,model_name)+'.h5'
    #
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #
    # callbacks_list = [checkpoint]  # model check pointing based on validation loss
    #
    # model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), class_weight=class_weights,callbacks=callbacks_list)



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
                        validation_data=(X_valid,y_valid), verbose=1)
#
# # 指标返回
    call_res = call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test,model)



    # 回调，向服务端发送评估指标

    # response = http_client.modelTrain(str(call_res))

    # response = request("http://192.168.10.141:8080/rpc/myservice", "modelTrain", str(call_res))

    # http_client.call("sayHelloWorld",call_res)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, model_name) + '.h5', overwrite=True)

    print('\r\nmodel has been saved in  ', os.path.join(save_dir, model_name) + '.h5')

    # print(type(history.history['loss']))
    # print(history.history['loss'])
    # print(type((history.history['acc'])))
    # print(type(history.history['loss']))
