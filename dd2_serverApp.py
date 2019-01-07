# -*- coding:utf-8 -*-

from jsonrpcserver import  method, serve
import os
from time import sleep

@method
def modelTrain(dict_parameters):
    """Test method"""
    print(type(dict_parameters))
        #dict_parameters = dict(dict_parameters)

    img_src =dict_parameters.get('img_src')
    width = dict_parameters.get("width")
    height = dict_parameters.get("height")
    test_percent =  dict_parameters.get("test_percent")
    validata_percent = dict_parameters.get("validata_percent")
    channel_num =dict_parameters.get("channel_num")
    num_classes =dict_parameters.get("num_classes")
    batch_size =dict_parameters.get("batch_size")
    lr =dict_parameters.get("lr")
    epochs = dict_parameters.get("epochs")
    save_dir = dict_parameters.get("save_dir")
    model_name = dict_parameters.get("model_name")
    model_id = dict_parameters.get('model_id')
    model_userid = dict_parameters.get('model_userid')
    model_version = dict_parameters.get('model_version')
    user_optimizer = dict_parameters.get('user_Optimizer')
    ams_id = dict_parameters.get('ams_id')
    jsonrpcMlClientPoint = dict_parameters.get('jsonrpcMlClientPoint')

    print(img_src, width, height, test_percent,validata_percent, channel_num, num_classes, batch_size, lr, epochs,save_dir,model_name,model_id,model_userid,model_version,user_optimizer,ams_id)
    print(''' recevived paramters...........................
                    python /root/Codes/newServer_kerasTrain/madel_model.py not /online_train.py  -img_src %s -width %d
                    -height %d
                    -test_percent %f
            -validata_percent %f
                    -channel_num %d
                    -num_classes %d
                    -batch_size %d
                    -lr %f
                    -epochs %d
                    -save_dir %s
                    -model_name %s
            -model_id  %s
            -model_userid %s
            -model_version %s
                -user_optimizer %s
            -ams_id  %s
	    -jsonrpcMlClientPoint  %s
                    '''
                  % (img_src,width,height,test_percent, validata_percent,channel_num,num_classes,batch_size,lr,epochs,save_dir,model_name,model_id,model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint))
    result = os.system('''
                    python /root/Codes/newServer_kerasTrain/made_modelApp.py  -img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d    -num_classes %d  -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint  %s
                    '''
                  %(img_src,width,height,test_percent, validata_percent,channel_num,num_classes,batch_size,lr,epochs,save_dir,model_name, model_id, model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint))
    print('current server ip address',jsonrpcMlClientPoint)
    print("starting executing the python scripts.......")

    Codes_path = '/root/Codes/uptest/made_modelApp.py'
    return  Codes_path

if __name__ =='__main__':
    print("serve is running!!!")
    serve("192.168.10.101", 10073)
    print("serve is stop!")
