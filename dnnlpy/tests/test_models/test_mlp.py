import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from dnnlpy.models import mlp


def test_parameter_tracks_grad_and_returns_plain_data():
    param = mlp.Parameter([[1.0, 2.0]], dtype=np.float64)
    param.grad = np.array([[0.5, -0.25]])

    result = param + 1.0

    assert isinstance(param, np.ndarray)
    assert not isinstance(param.data, mlp.Parameter)
    assert not isinstance(result, mlp.Parameter)
    assert_allclose(param.data, np.array([[1.0, 2.0]]))
    assert_allclose(param.grad, np.array([[0.5, -0.25]]))


def test_flatten_forward_and_backward():
    module = mlp.Flatten()
    x = np.arange(24).reshape(2, 3, 4)

    output = module(x)
    dx = module.backward(np.ones_like(output))

    assert output.shape == (2, 12)
    assert_equal(output[0], np.arange(12))
    assert dx.shape == x.shape


def test_linear_forward_backward_and_parameters():
    x = np.array([[2.0, -1.0], [0.0, 4.0]])
    grad = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])

    module = mlp.Linear(in_features=2, out_features=3)
    module.weight[:] = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    module.bias[:] = np.array([0.5, -0.5, 1.0])

    actual = module(x)
    actual_grad = module.backward(grad)
    params = list(module.parameters())

    expected = np.array([[-0.5, -4.5, 3.0], [12.5, -0.5, -3.0]])
    expected_W_grad = np.array([[2.0, 4.0, -2.0], [1.0, -4.0, 13.0]])
    expected_b_grad = np.array([1.5, 1.5, 2.0])
    expected_grad = np.array([[-3.5, 4.0], [3.0, -1.5]])

    assert_allclose(actual, expected)
    assert module.weight.grad is not None
    assert_allclose(module.weight.grad, expected_W_grad)
    assert module.bias.grad is not None
    assert_allclose(module.bias.grad, expected_b_grad)
    assert_allclose(actual_grad, expected_grad)
    assert params[0] is module.weight
    assert params[1] is module.bias
    assert 'in_features=2' in repr(module)


def test_linear_backward_requires_forward():
    module = mlp.Linear(in_features=2, out_features=3)

    with pytest.raises(AssertionError, match='forward'):
        module.backward(np.ones((1, 3)))


@pytest.mark.parametrize(
    ('module', 'expected', 'expected_grad'),
    [
        (
            mlp.Sigmoid(),
            np.array([[0.26894142, 0.5, 0.73105858]]),
            np.array([[0.19661193, 0.25, 0.19661193]]),
        ),
        (
            mlp.Tanh(),
            np.array([[-0.76159416, 0.0, 0.76159416]]),
            np.array([[0.41997434, 1.0, 0.41997434]]),
        ),
        (
            mlp.ReLU(),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_elementwise_activations_forward_and_backward(
    module: mlp.Module,
    expected: np.ndarray,
    expected_grad: np.ndarray,
):
    x = np.array([[-1.0, 0.0, 1.0]])
    grad = np.ones_like(x)

    actual = module(x)
    actual_grad = module.backward(grad)

    assert_allclose(actual, expected)
    assert_allclose(actual_grad, expected_grad)


def test_softmax_forward_and_backward():
    module = mlp.Softmax()
    x = np.array([[1.0, 2.0, 3.0]])
    grad = np.array([[0.2, -0.1, 0.4]])

    actual = module(x)
    actual_grad = module.backward(grad)

    expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
    expected_dot = np.sum(grad * expected, axis=1, keepdims=True)
    expected_grad = expected * (grad - expected_dot)

    assert_allclose(actual, expected)
    assert_allclose(np.sum(actual, axis=1), np.array([1.0]))
    assert_allclose(actual_grad, expected_grad)


def test_cross_entropy_loss_forward_backward():
    module = mlp.CrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = np.array([0, 2])

    actual = module(logits, targets)
    actual_grad = module.backward()

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    expected = -np.mean(np.log(probs[np.arange(2), targets] + module.eps))
    expected_grad = probs.copy()
    expected_grad[np.arange(2), targets] -= 1
    expected_grad = expected_grad / 2

    assert_allclose(actual, expected)
    assert_allclose(actual_grad, expected_grad)


def test_cross_entropy_backward_requires_forward():
    module = mlp.CrossEntropyLoss()

    with pytest.raises(AssertionError, match='forward'):
        module.backward()


def test_mlp_forward_backward_yields_recursive_parameter_gradients():
    x = np.array([[2.0, -1.0], [0.0, 1.0]])
    grad = np.array([[0.2, -0.3], [0.5, 0.1]])

    model = mlp.MLP(input_dim=2, hidden_dim=3, num_classes=2)
    model.fc1.weight[:] = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
    model.fc1.bias[:] = np.array([0.0, 0.5, -0.25])
    model.fc2.weight[:] = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    model.fc2.bias[:] = np.array([0.1, -0.2])

    logits = model(x)
    dx = model.backward(grad)
    params = list(model.parameters())

    assert logits.shape == (2, 2)
    assert dx.shape == x.shape
    assert len(params) == 4
    assert all(param.grad is not None for param in params)


def test_sgd_updates_parameters_and_zeroes_gradients():
    param = mlp.Parameter([1.0, -2.0])
    param.grad = np.array([0.5, -0.25])
    # Skip it because it does not have a gradient.
    skipped = mlp.Parameter([3.0])

    optimizer = mlp.SGD([param, skipped], lr=0.1)
    optimizer.step()

    assert_allclose(param, np.array([0.95, -1.975]))
    assert_allclose(skipped, np.array([3.0]))

    optimizer.zero_grad(set_to_none=False)
    assert_allclose(param.grad, np.zeros_like(param))

    optimizer.zero_grad()
    assert param.grad is None


def test_mlp_training_step_reduces_cross_entropy_loss():
    x = np.array([[2.0, -1.0], [0.0, 1.0]])
    targets = np.array([0, 1])

    model = mlp.MLP(input_dim=2, hidden_dim=3, num_classes=2)
    model.fc1.weight[:] = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
    model.fc1.bias[:] = np.array([0.0, 0.5, -0.25])
    model.fc2.weight[:] = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    model.fc2.bias[:] = np.array([0.1, -0.2])

    loss_fn = mlp.CrossEntropyLoss()
    optimizer = mlp.SGD(model.parameters(), lr=0.1)

    loss_before = loss_fn(model(x), targets)
    model.backward(loss_fn.backward())
    optimizer.step()
    loss_after = loss_fn(model(x), targets)

    assert loss_after.item() < loss_before.item()
