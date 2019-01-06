import numpy as np
import scipy.optimize as opt

def _sigmoid(z):
    return 1/(1 + np.e**(-z))

def _sigmoid_deriv(z):
    return _sigmoid(z)*(1-_sigmoid(z))

class NeuralNet:

    def __init__(self, num_features, num_hidden_layers, num_hidden_nodes, num_labels, th=None,
                            reg_term=0.5, fun=_sigmoid, dfun=_sigmoid_deriv,
                            th_init_low=-0.12, th_init_high=0.12):                    
        self.reg = reg_term
        self.fun = fun
        self.dfun = dfun
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_labels = num_labels
        self.layers_sizes = [num_features, *num_hidden_nodes, num_labels]
        self.th_init_low = th_init_low
        self.th_init_high = th_init_high
        self.th = self._initial_thetas() if th is None else th
    
    def _random_thetas_layer(self, L_in, L_out):
        """ Construye los pesos de una capa de la red
        de forma aleatoria
        """
        return np.random.uniform(low=self.th_init_low, high=self.th_init_high,
                                                 size=(L_out, L_in+1))

    def _initial_thetas(self):
        """ Construye los pesos de toda la red
        de forma aleatoria
        """
        temp_th = []
        for i in range(0, len(self.layers_sizes)-1):
            l_in = self.layers_sizes[i]
            l_out = self.layers_sizes[i+1]
            temp_th.append(self._random_thetas_layer(l_in, l_out))
        self.th = np.asarray(temp_th)

    def _cost(self, y_true, y_pred):
        """ Calcula el coste del modelo utilizando una funcion
        sigmoide
        """
        assert len(y_true) == len(y_pred)
        m = len(y_true)
        J = np.sum( -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred) ) / m
        J += (self.reg / (2 * m)) * (np.sum(self.th[:-1, 1:] ** 2)) + np.sum(self.th[-1, 1:] ** 2)
        return J

    def _backprop(self, X, y):
        """ Devuelve el coste y el gradiente de la red asumiendo
        one-hot encoding para 'y'
        """
        m = len(X)

        # forward propagation
        a = [np.insert(X, 0, values=np.ones(m), axis=1)] # a0 = x
        z = [None] # no hay z0
        for i in range(1, len(self.layers_sizes)):
            z.append( np.matmul( a[i-1], self.th[i-1].T))
            a[i] = np.insert(self.fun(z[i]), 0, values=np.ones(m), axis=1)
        pred = self.fun(z[-1])
        
        J = self._cost(y,pred)
        
        # back propagation
        delta = [pred - y]
        for i in range(1, len(self.layers_sizes)-1):
            zs = np.insert(z[-i-1], 0, values=np.ones(m), axis=1)
            delta.append( np.multiply(np.matmul(delta[i-1], self.th[-i]), self.dfun(zs)))
        delta = np.asarray(delta)

        delta_ = [ delta[-1-i].T.dot(a[i]) for i in range(0, len(self.layers_sizes) - 1) ]
        delta_ = np.asarray(delta_)
        
        th_ = [np.c_[np.zeros((self.th[i].shape[0],1)),self.th[i][:,1:]]
         for i in range(0, len(self.th))]
        th_ = np.asarray(th_)

        D = [ (delta_[i][1:,:]/m + (th_[i]*self.reg)/m ).ravel() for i in range(0, len(th_))]

        # rolling del gradiente para la salida
        G = D[0]
        for i in range(1, len(D)):
            G = np.r_[G, D[i]]

        return(J, G)

    def fit(self, X, y, maxiter=70):
        """ Entrena los pesos del modelo utilizando 'fmin_tnc' de 
        la libreria 'scipy.optimize'
        """
        fmin = opt.minimize(fun=self._backprop, x0=self.th, args=(X, y),
                    method='TNC', jac=True, options={'maxiter':maxiter})
        temp_th = []
        range_ini, range_end = 0,0
        for i in range(0, len(self.th)):
            range_end =  range_end + self.layers_sizes[i+1]*(self.layers_sizes[i]+1) if i+1 < len(self.th) else len(fmin.x)
            temp_th.append(fmin.x[range_ini:range_end].reshape(self.layers_sizes[i+1],(self.layers_sizes[i]+1)))
            range_ini = range_end
        self.th = np.asarray(temp_th)

    def predict(self, X, threshold=0.5):
        """ Devuelve las predicciones del modelo para los datos
        de 'X' utilizando 'threshold' si el modelo solo
        tiene dos posibles etiquetas
        """
        pred = []
        for x in X:
            a = [np.insert(x, 0, values=np.ones(1), axis=1)] # a0 = x
            z = [None] # no hay z0
            for i in range(1, len(self.layers_sizes)):
                z.append( np.matmul( a[i-1], self.th[i-1].T))
                a[i] = np.insert(self.fun(z[i]), 0, values=np.ones(1), axis=1)

            probability_per_label = self.fun(z[-1])
            best_probability_prediction_index = np.argmax(probability_per_label)
            prediction = best_probability_prediction_index+1
        
            pred.append(prediction)
        return np.asarray(pred)

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