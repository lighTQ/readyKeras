# -*- coding:utf-8 -*-
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    def __init__(self, http_client, model_id, model_userid, model_version, ams_id, startAt=0):
        super(TrainingMonitor, self).__init__()

        # # 开始模型开始保存的开始epoch
        self.startAt = startAt
        self.http_client = http_client
        self.model_id = model_id
        self.model_userid = model_userid
        self.model_version = model_version
        self.ams_id = ams_id

    def on_train_begin(self, logs={}):
        # 初始化保存文件的目录dict
        self.H = {}
        # 判断是否存在文件和该目录

    def on_epoch_end(self, epoch, logs={}):
        # 不断更新logs和loss accuracy等等

        if self.model_id is not None:
            call_res = {'model_id': self.model_id, 'model_userid': self.model_userid,
                        'model_version': self.model_version, 'epoch': epoch, 'ams_id': self.ams_id,
                        'calculationType': 'process'}
            for (k, v) in logs.items():
                self.H[k] = v
                print(k, self.H[k])
                call_res.setdefault(k, self.H[k])
                status = self.http_client.modelTrain(str(call_res))
            print(call_res)
            print("-=-=-=--==---=-= received staussis " + status)
