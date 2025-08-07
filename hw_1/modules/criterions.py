import numpy as np
from .base import Criterion
from .activations import LogSoftmax
from .activations import Softmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        return ((input - target) ** 2).mean()
    
    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'

        return 2 * (input - target) / (input.shape[0] * input.shape[1])


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.softmax = Softmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        
        logsoftmax_val = self.log_softmax(input)
        c_eq_yi = logsoftmax_val[np.arange(input.shape[0]), target.ravel()]

        return -c_eq_yi.mean()

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """

        sftmax_grad = np.zeros_like(input)
        sftmax_grad[np.arange(input.shape[0]), target.ravel()] = -1 / input.shape[0]

        return self.log_softmax.compute_grad_input(input, sftmax_grad)
