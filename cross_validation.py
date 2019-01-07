import numpy as np
from sklearn.metrics import accuracy_score

def cv_config(model, train_X, train_y, val_X, val_y, C=[0.01,0.03,0.1,0.3,1,3,10,30], sigma=[0.01,0.03,0.1,0.3,1,3,10,30]):
    """
        Try every possible C-sigma combination using cross-validation
        Returns scores and configs arrays
    """
    scores = []
    configs = []
    for c in C:
        for s in sigma:
            model.set_params(C=c, gamma=1 / (2 * s**2))
            model.fit(X=train_X, y=train_y)
            config_score = accuracy_score(val_y, model.predict(val_X))
            scores.append(config_score)
            configs.append((c,s))
    return (np.array(scores), np.array(configs))