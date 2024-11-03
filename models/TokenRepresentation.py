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
        if self.paras["model_config"][self.count] == 'LDA':
            lda_model = lda_m.LDA_model()
            return lda_model.LDA_model(self.data, self.paras, self.count)
        elif self.paras["model_config"][self.count] == 'sentence_bert':
            pass