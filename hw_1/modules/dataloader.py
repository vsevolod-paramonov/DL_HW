import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """

        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """

        if self.shuffle:

            idx = np.random.permutation(self.X.shape[0])
            self.X, self.y = self.X[idx, :], self.y[idx]
            
        self.batch_id = 0
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        
        if self.batch_id * self.batch_size < self.X.shape[0]:

            X, y = self.X[self.batch_id * self.batch_size : min((self.batch_id + 1) * self.batch_size, self.X.shape[0])], \
                   self.y[self.batch_id * self.batch_size : min((self.batch_id + 1) * self.batch_size, self.X.shape[0])]
            
            self.batch_id += 1

            return X, y

        raise StopIteration
