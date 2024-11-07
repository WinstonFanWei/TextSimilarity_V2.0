import os
import logging
from sklearn.metrics import f1_score

def compute_metrics(df_compare, paras, len):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)

    message = "\n---------------------- RMSE ---------------------- \n[0, +inf] RMSE smaller is better." 
    
    for i in range(len):
        message += f"\n[方案{i}] RMSE: " + str(compute_rmse(df_compare.iloc[:, i-len], df_compare["GroundTruth"]))

    message += "\n--------------------------------------------------\n"

    logger.debug(message)

    message = "\n---------------------- Correlation ---------------------- \n[-1, 1] Correlation bigger is better."

    for i in range(len):
        message += f"\n[方案{i}] CORR: " + str(compute_correlation(df_compare.iloc[:, i-len], df_compare["GroundTruth"]))

    message += "\n--------------------------------------------------\n"
    
    logger.debug(message)

    message =  "\n---------------------- F1-score ---------------------- \n[0, 1] F1-score bigger is better. "

    for i in range(len):
        message += f"\n[方案{i}] F1-score: " + str(compute_f1(df_compare.iloc[:, i-len], df_compare["GroundTruth"]))

    message += "\n--------------------------------------------------\n"
    
    logger.debug(message)


def rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean()) ** 0.5

def compute_rmse(a, b):
    return round(rmse(a, b), 4)
    
def compute_correlation(a, b):
    return round(a.corr(b), 4)

def compute_f1(a, b):
    return round(f1_score((a > 0.5).astype(int), (b > 0.5).astype(int)), 4)
    