from data.data import Data
from util.conf import OptionConf
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time


class Recommender(object):
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)
        self.model_name = self.config.model
        self.emb_size = int(self.config.embedding_size)
        self.maxEpoch = int(self.config.epoch)
        self.batch_size = int(self.config.batch_size)
        self.lRate = float(self.config.lr)
        self.reg = float(self.config.reg)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, self.model_name + ' ' + current_time)
        self.result = []
        self.recOutput = []

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        print('### model configuration ###')
        self.model_log.add(self.config)

    def print_model_info(self):
        print('Model:', self.config.model)
        print('Training Set:', abspath('./dataset/' + self.config.dataset_name + '/train.txt'))
        print('Test Set:', abspath('./dataset/' + self.config.dataset_name  + '/test.txt'))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:',  self.reg)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)
        print('Saving embedding...')
