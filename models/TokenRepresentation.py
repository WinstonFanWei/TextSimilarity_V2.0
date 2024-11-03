'''
功能：进行每个层级token的表示，输出向量。
'''
import LDA_model as lda_m

class representor:
    def __init__(self, data, paras):
        self.paras = paras
        self.data = data

    def get_token_representation(self):
        if self.paras["token_representation_method"] == 'lda':
            lda_model = lda_m()
            return lda_model.LDA_model(self.data, self.paras)
        elif self.paras["token_representation_method"] == 'sentence_bert':
            pass