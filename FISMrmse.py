import math
import logging
from time import time

import numpy as np
import scipy.sparse as sp

from dataset import DataSet
from evaluate import evaluate_model

"""
Kabbur et al., <strong>FISM: Factored Item Similarity Models for Top-N Recommender Systems</strong>, KDD 2013.
"""


class FISMrmse(object):
    def __init__(self,
                 num_users,
                 num_items,
                 lr=0.001,
                 rho=4,
                 alpha=0.5,
                 beta=0.6,
                 item_bias_reg=0.1,
                 user_bias_reg=0.1,
                 num_factors=16):
        self.lr = lr
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.P = self.normal(size=(num_items, num_factors))
        self.Q = self.normal(size=(num_items, num_factors))
        self.user_biases = self.normal(size=num_users)
        self.item_biases = self.normal(size=num_items)
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.train_matrix = None

    @staticmethod
    def normal(mean=0.0, stddev=0.01, size=None):
        return np.random.normal(loc=mean, scale=stddev, size=size)

    def train(self, train_matrix, epochs=10, verbose=1):
        print('start training...')
        logging.info('start...')
        self.train_matrix = train_matrix
        losses = []
        for epoch in range(epochs):
            start = time()

            # sample
            R = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
            count = 0
            for (u, i) in self.train_matrix.keys():
                R[u, i] = 1
                count += 1
                print('Sample%d: %d' % (epoch, count))

                # negative sampling
                for t in range(self.rho):
                    j = np.random.randint(self.num_items)
                    while (u, j) in self.train_matrix.keys() or (u, j) in R.keys():
                        j = np.random.randint(self.num_items)
                    R[u, j] = 0

            print('Sample finished.')

            loss = 0

            count = 0
            for (u, i) in R.keys():
                count += 1
                print('Train%d: %d' % (epoch, count))

                n_u = len(self.train_matrix[u])

                x = np.zeros(shape=(self.num_factors,))
                for j in self.train_matrix[u].keys():
                    j = j[1]
                    if j == i:
                        continue
                    x += self.P[j]
                x = x / math.pow(n_u - 1, self.alpha)

                b_u = self.user_biases[u]
                b_i = self.item_biases[i]
                predict_r_ui = b_u + b_i + np.dot(self.Q[i], x)

                e_ui = R[u, i] - predict_r_ui

                self.user_biases[u] = b_u + self.lr * (e_ui - self.user_bias_reg * b_u)
                self.item_biases[i] = b_i + self.lr * (e_ui - self.item_bias_reg * b_i)

                loss += e_ui * e_ui + self.user_bias_reg * b_u * b_u + self.item_bias_reg * b_i * b_i

                self.Q[i] = self.Q[i] + self.lr * (e_ui * x - self.beta * self.Q[i])

                loss += self.beta * np.dot(self.Q[i], self.Q[i])

                for j in self.train_matrix[u].keys():
                    j = j[1]
                    if j == i:
                        continue
                    self.P[j] = self.P[j] + self.lr * \
                                (e_ui / math.pow(n_u - 1, self.alpha) * self.Q[i] - self.beta * self.P[j])
                    loss += self.beta * np.dot(self.P[j], self.P[j])

            loss /= 2
            if losses:
                delta_loss = losses[-1] - loss
            else:
                delta_loss = loss

            losses.append(loss)

            if verbose:
                print('Epoch %d: loss = %.4f, delta_loss = %.4f [%.1fs]' %
                      (epoch, loss, delta_loss, time() - start))

                logging.info('Epoch %d: loss = %.4f, delta_loss = %.4f [%.1fs]' %
                             (epoch, loss, delta_loss, time() - start))

            # is converge
            if math.fabs(delta_loss) < 1e-5:
                return losses
        return losses

    def save_weights(self, file):
        np.save(file, np.array([self.P,
                                self.Q,
                                self.user_biases,
                                self.item_biases,
                                self.train_matrix]))

    def load_weights(self, file):
        weights = np.load(file, allow_pickle=True)
        self.P = weights[0]
        self.Q = weights[1]
        self.user_biases = weights[2]
        self.item_biases = weights[3]
        self.train_matrix = weights[4]

    def predict(self, users, items):
        predictions = []
        for idx in range(len(users)):
            u = users[idx]
            i = items[idx]

            bias = self.user_biases[u] + self.item_biases[i]

            dot_sum = 0
            for j in self.train_matrix[u].keys():
                j = j[1]
                dot_sum += np.dot(self.P[i], self.Q[j])

            n_u = len(self.train_matrix[u])
            if n_u <= 0:
                w_u = 0
            else:
                w_u = math.pow(n_u, -self.alpha)
            predictions.append(bias + w_u * dot_sum)
        return predictions


def plot_losses(epochs, losses):
    import matplotlib.pyplot as plt

    plt.plot(epochs, losses, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(xmin=1, xmax=epochs[-1] + 1)
    plt.ylim(ymin=0)
    plt.show()


if __name__ == '__main__':
    epochs = 10
    path = 'data'
    data_set = 'ml-100k'
    dataset = DataSet(path=path, data_set=data_set)

    root = logging.getLogger()
    if root.handlers:
        root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        filename='FISMrmse_%s.log' % data_set,
                        level=logging.INFO)

    fism = FISMrmse(num_users=dataset.num_users,
                    num_items=dataset.num_items)

    losses = fism.train(train_matrix=dataset.data_matrix,
                        epochs=epochs)
    fism.save_weights('fism_rmse_weights.npy')

    # plot_losses(np.arange(start=0, stop=epochs, step=1), losses)

    fism.load_weights('fism_rmse_weights.npy')

    hits, ndcgs, arhrs = evaluate_model(fism, dataset.test_ratings, dataset.test_negatives, 10)
    hit, ndcg, arhr = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(arhrs).mean()
    print('HR = %.4f, NDCG = %.4f, ARHR = %.4f' % (hit, ndcg, arhr))
