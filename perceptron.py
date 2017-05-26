from functools import reduce


class Perceptron:
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weigths = [0.0 for _ in range(0, input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weigths:{}\nbias:{}\n'.format(list(self.weigths), self.bias)

    ## f(x)=wx+b
    def predict(self, input_vec):
        return self.activator(
            reduce(lambda x, y: x + y,
                   map(lambda x, w: x * w, input_vec, self.weigths),
                   self.bias))

    def train(self, input_vecs, labels, rate, iteration):
        for i in range(0, iteration):
            self.one_iteration(input_vecs, labels, rate)

    def one_iteration(self, input_vecs, labels, rate):
        sample = zip(input_vecs, labels)
        for input_vec, label in sample:
            output = self.predict(input_vec)
            self.update_weigths(input_vec, output, label, rate)

    def update_weigths(self, input_vec, output, label, rate):
        delta = label - output
        self.weigths = list(map(lambda w, x: w + rate * x * delta, self.weigths, input_vec))
        self.bias += rate * delta


def activator(s):
    return 1 if s > 0 else 0


def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    input_num = 2
    return input_vecs, labels, input_num


def train_and_perceptron():
    input_vecs, labels, input_num = get_training_dataset()
    p = Perceptron(input_num, activator)
    p.train(input_vecs, labels, 0.1, 10)

    return p


if __name__ == '__main__':
    perceptron = train_and_perceptron()

    print(perceptron)
    print('1 and 1 = {}'.format(perceptron.predict([1, 1])))
    print('0 and 0 = {}'.format(perceptron.predict([0, 0])))
    print('0 and 1 = {}'.format(perceptron.predict([0, 1])))
    print('1 and 0 = {}'.format(perceptron.predict([1, 0])))
