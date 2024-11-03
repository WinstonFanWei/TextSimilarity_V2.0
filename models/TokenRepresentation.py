'''
功能：进行每个层级token的表示，输出向量。
'''
import models.LDA_model as lda_m

class representor:
    def __init__(self, data, paras, count):
        self.paras = paras
        self.data = data
        self.count = count

    def get_token_representation(self):
        if self.paras["model_config"][self.count]["token_representation_method"] == 'LDA':
            lda_model = lda_m.LDA_model()
            lda_model.train_LDA_model(self.data, self.paras, self.count)
            return lda_model.token_represent()
        elif self.paras["model_config"][self.count]["token_representation_method"] == 'sentence_bert':
            return None
        else:
            return None