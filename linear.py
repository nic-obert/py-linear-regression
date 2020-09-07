
class Model:
    def __init__(self, dims, learning_rate=0.1):
        # these are vectors of n-dimentions where n is the number of features a sample has
        self.dims = dims
        self.biases = [0 for _ in range(self.dims)]
        self.weights = [1 for _ in range(self.dims)]
        self.learning_rate = learning_rate
    
    def fit(self, data):
        batch_error = 0
        batch_size = len(data)
        if len(data[0]) -1 != self.dims:
            raise ValueError('too many features in dataset')

        for sample in data:
            pred = 0
            for dim in range(self.dims):
                # pred += x * w + b
                pred += sample[dim] * self.weights[dim] + self.biases[dim]

            
            error = (sample[-1] - pred)**2  # squared error

            for dim in range(self.dims):
                """
                    error = (y - pred)^2
                    pred = xw + b
                    error = (y - (xw + b))^2 = (y - xw - b)^2
                    dw_error = 2(y - xw - b) * (0 - x - 0)
                            = 2(y - xw - b) * (-w)
                            = -2w(y - xw - b)
                    db_error = 2(y - xw - b) * (0 - 0 - 1)
                            = -2(y - xw - b)
                    
                """
                dw = -2 * self.weights[dim] * (sample[-1] - sample[dim] * self.weights[dim] + self.biases[dim])
                db = -2 * (sample[-1] - sample[dim] * self.weights[dim] + self.biases[dim])

                #print(dw, db, '\t\t', self.weights[dim], self.biases[dim])
                #if dw > self.learning_rate or db > self.learning_rate: return 

                self.weights[dim] -= self.learning_rate * dw
                self.biases[dim] -= self.learning_rate * db
                    

            batch_error += error
                
        
        batch_error /= batch_size
        print('error:', batch_error)
        

if __name__ == "__main__":
    
    data = [
        [4, 2, 8],
        [3, 3, 9],
        [2, 6, 12],
        [3, 2, 6]
    ]
    """
    data = [
        [2, 4],
        [1, 2],
        [3, 6],
        [5, 10]
    ]
    """
    #data = [[2, 4]]
    

    model = Model(2, 0.1)
    epochs = 10
    print(model.weights, model.biases)
    for i in range(epochs):
        model.fit(data)
    print(model.weights, model.biases)
