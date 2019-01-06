import numpy as np

def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W


def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


if __name__ == "__main__":
    from neural_net import NeuralNet

    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    net = NeuralNet(input_layer_size, 1, [hidden_layer_size], num_labels, th=[Theta2,Theta1], reg_term=0.8)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    # Unroll parameters
    nn_params = np.append(Theta2, Theta1).reshape(-1)

    y = (np.ravel(y) - 1)
    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i][y[i]] = 1

    
    # Compute Cost
    cost, grad = net._backprop(X,np.array(ys))

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        net.th[0] = np.reshape(p[:hidden_layer_size*(input_layer_size+1)],
                        (hidden_layer_size, (input_layer_size+1)))
        net.th[1] = np.reshape(p[hidden_layer_size*(input_layer_size+1):],
                        (num_labels, (hidden_layer_size+1)))
        return net._backprop(X,np.array(ys))[0]

    numgrad = computeNumericalGradient(reduced_cost_func, nn_params)
    
    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)
    print(grad - numgrad)