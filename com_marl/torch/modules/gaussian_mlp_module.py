"""GaussianMLPModule."""
import torch
from torch import nn
from garage.torch.modules.mlp_module import MLPModule


class GaussianMLPModule(nn.Module):
    """GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 single_agent_action_dim=None, # used for centralized
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 # duplicate_std_copies=None,
                 share_std=False, 
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim # can be multiagent action dim for centralized
        self._single_agent_action_dim = single_agent_action_dim
        self._learn_std = learn_std
        # self._duplicate_std_copies = duplicate_std_copies # n agents, for centralized
        self._share_std = share_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        if share_std:
            init_std_param = torch.Tensor([init_std]).log()
        else:
            if single_agent_action_dim is not None:
                init_std_param = torch.Tensor([init_std] * single_agent_action_dim).log()
            else:
                init_std_param = torch.Tensor([init_std] * self._action_dim).log()

        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()

        self._mean_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    def forward(self, inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: Module output.

        """
        mean = self._mean_module(inputs)

        if self._share_std:
            broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
            log_std_uncentered = torch.zeros(*broadcast_shape, device=inputs.device) + self._init_std
        else:
            log_std_uncentered = self._init_std
            # if self._duplicate_std_copies is not None:
            #     log_std_uncentered = self._init_std.repeat(self._duplicate_std_copies)


        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=self._to_scalar_if_not_none(self._min_std_param),
                max=self._to_scalar_if_not_none(self._max_std_param))

        if self._share_std:
            if self._std_parameterization == 'exp':
                std = log_std_uncentered.exp()
            else:
                std = log_std_uncentered.exp().exp().add(1.).log()
            # dist = Independent(Normal(mean, std), 1)
        else:
            if self._std_parameterization == 'exp':
                std = torch.diag(log_std_uncentered.exp())
            else:
                std = torch.diag(log_std_uncentered.exp().exp().add(1.).log())
            # dist = MultivariateNormal(mean, std)

        return mean, std

    # pylint: disable=no-self-use
    def _to_scalar_if_not_none(self, tensor):
        """Convert torch.Tensor of a single value to a Python number.

        Args:
            tensor (torch.Tensor): A torch.Tensor of a single value.

        Returns:
            float: The value of tensor.

        """
        return None if tensor is None else tensor.item()