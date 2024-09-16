import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

if __name__ == '__main__':
    pred = np.array([1,2,3,4,5]).reshape(-1,1)
    target = np.array([2,3,4,5,6]).reshape(-1,1)
    print(r2_score(pred,target))
