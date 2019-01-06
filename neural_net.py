import numpy as np
import scipy.optimize as opt

def _sigmoid(z):
    return 1/(1 + np.e**(-z))

def _sigmoid_deriv(z):
    return _sigmoid(z)*(1-_sigmoid(z))

class NeuralNet:

    def __init__(self, num_features, num_hidden_nodes, num_labels, th=None,
                            reg_term=0.5, fun=_sigmoid, dfun=_sigmoid_deriv,
                            th_init_low=-0.12, th_init_high=0.12):                    
        self.reg = reg_term
        self.fun = fun
        self.dfun = dfun
        self.num_features = num_features
        self.num_hidden_layers = 1
        self.num_hidden_nodes = num_hidden_nodes
        self.num_labels = num_labels
        self.layers_sizes = list([num_features, num_hidden_nodes, num_labels])
        self.th_init_low = th_init_low
        self.th_init_high = th_init_high

        if th is None:
            self._initialize_thetas()
        else:
            self.th1, self.th2 = th[0], th[1]
    
    def _random_thetas_layer(self, L_in, L_out):
        """ Construye los pesos de una capa de la red
        de forma aleatoria
        """
        return np.random.uniform(low=self.th_init_low, high=self.th_init_high,
                                                 size=(L_out, L_in+1))

    def _initialize_thetas(self):
        """ Construye los pesos de toda la red
        de forma aleatoria
        """
        self.th1 = self._random_thetas_layer(self.layers_sizes[0], self.layers_sizes[1])
        self.th2 = self._random_thetas_layer(self.layers_sizes[1], self.layers_sizes[2])

    def _cost(self, y_true, y_pred):
        """ Calcula el coste del modelo utilizando una funcion
        sigmoide
        """
        assert len(y_true) == len(y_pred)

        m = len(y_true)
        J = np.sum( -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred) ) / m
        J += (self.reg / (2 * m)) * (np.sum(self.th1[:, 1:] ** 2)
                            + np.sum(self.th2[:, 1:] ** 2))
        return J

    def _backprop(self, params, X, y):
        """ Devuelve el coste y el gradiente de la red asumiendo
        one-hot encoding para 'y'
        """
        m = len(X)

        # forward propagation
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = np.matmul(a1, self.th1.T)
        a2 = np.insert(self.fun(z2), 0, values=np.ones(m), axis=1)
        z3 = np.matmul(a2, self.th2.T)
        pred = self.fun(z3)

        # cost calc
        J = self._cost(y,pred)
        
        # back propagation
        delta3 = pred - y
        z2s = np.insert(z2, 0, values=np.ones(m), axis=1)
        delta2 = np.multiply(np.matmul(delta3, self.th2), self.dfun(z2s))
        
        delta1_ = delta2.T.dot(a1) 
        delta2_ = delta3.T.dot(a2) 
        
        theta1_ = np.c_[np.zeros((self.th1.shape[0],1)),self.th1[:,1:]]
        theta2_ = np.c_[np.zeros((self.th2.shape[0],1)),self.th2[:,1:]]
        
        D1 = (delta1_[1:,:]/m + (theta1_*self.reg)/m ).ravel()
        D2 = (delta2_/m + (theta2_*self.reg)/m ).ravel()
        # rolling de gradient para salida
        return(J, np.r_[D1, D2])

    def fit(self, X, y, maxiter=70):
        """ Entrena los pesos del modelo utilizando 'minimize' de 
        la libreria 'scipy.optimize'
        """
        fmin = opt.minimize(fun=self._backprop, x0=np.append(self.th1, self.th2).reshape(-1), args=(X, y),
                    method='TNC', jac=True, options={'maxiter':maxiter})

        self.th1 = fmin.x[:(self.num_hidden_nodes*(self.num_features+1))].reshape(self.num_hidden_nodes,(self.num_features+1))
        self.th2 = fmin.x[(self.num_hidden_nodes*(self.num_features+1)):].reshape(self.num_labels,(self.num_hidden_nodes+1))

    def predict(self, X):
        """ Devuelve las predicciones del modelo para los datos
        de 'X' utilizando 'threshold' si el modelo solo
        tiene dos posibles etiquetas
        """
        m = len(X)
        X_ones = np.insert(X, 0,values=np.ones(m), axis=1)
        pred = []
        for x in X_ones:
            output_layer1 = np.array([self.fun(self.th1[i].dot(x.T))
                                  for i in range(0,len(self.th1))])
            output_layer1 = np.insert(output_layer1, 0, 1)
            output_layer2 = np.array([self.fun(self.th2[i].dot(output_layer1))
                                    for i in range(0,len(self.th2))])
            
            probability_per_label = output_layer2
            best_probability_prediction_index = np.argmax(probability_per_label)
            prediction = best_probability_prediction_index+1
            
            pred.append(prediction)

        return pred

    def accuracy_score(self, y_true, y_pred, normalize=True):
        """ Calcula el porcentaje de aciertos del modelo sobre 'y_true'
        utilizando las predicciones de 'y_pred'
        Si 'normalize'=False, devuelve el numero de aciertos
        """
        assert len(y_true) == len(y_pred)
    
        num_hits = 0
        m = len(y_true)
        for i in range(0,m):
            if(y_true[i] == y_pred[i]):
                num_hits += 1

        return (num_hits/m)*100 if normalize else num_hits
