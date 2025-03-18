from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        training_path = './dataset/' + config.dataset_name + '/train.txt'
        test_path = './dataset/' + config.dataset_name + '/test.txt'
        self.training_data = FileIO.load_data_set(training_path, config.model_type)
        self.test_data = FileIO.load_data_set(test_path, config.model_type)

        self.kwargs = {}


    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config.model_type +'.' + self.config.model + ' import ' + self.config.model
        exec(import_str)
        recommender = self.config.model + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()
