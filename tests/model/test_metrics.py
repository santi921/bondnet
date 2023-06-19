import torch
from torch.nn import functional as F
import numpy as np
from math import log2
from bondnet.model.metric import (
    Metrics_WeightedMAE,
    Metrics_WeightedMSE,
    Metrics_Accuracy_Weighted,
    Metrics_Cross_Entropy,
)

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def test_mse():
    mse_obj = Metrics_WeightedMSE(reduction="mean")
    test_a = torch.tensor([1.0, 2.0])
    test_b = torch.tensor([2.0, 4.0])
    weight = torch.tensor([1.0, 1.0])
    test_a_full = [[1.0, 2.0] for i in range(3)]
    test_b_full = [[2.0, 4.0] for i in range(3)]
    weight_full = [[1.0, 2.0] for i in range(3)]

    [mse_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]

    # unfurl the list
    test_a_full = np.array(test_a_full).flatten()
    test_b_full = np.array(test_b_full).flatten()
    weight_full = np.array(weight_full).flatten()

    sklearn_mse = mean_squared_error(test_a_full, test_b_full, squared=True)
    sklearn_mse_w = mean_squared_error(
        test_a_full, test_b_full, sample_weight=weight_full, squared=True
    )

    metric_mse = mse_obj.compute()
    assert np.allclose(metric_mse, sklearn_mse), "MSE unweighted not equal"

    mse_obj = Metrics_WeightedMSE(reduction="mean")
    weight = torch.tensor([1.0, 2.0])
    [mse_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]

    metric_mse = mse_obj.compute()
    assert np.allclose(metric_mse, sklearn_mse_w), "MSE weighted not equal"


def test_mae():
    mse_obj = Metrics_WeightedMAE(reduction="mean")
    test_a = torch.tensor([1.0, 2.0])
    test_b = torch.tensor([2.0, 4.0])
    weight = torch.tensor([1.0, 1.0])
    test_a_full = [[1.0, 2.0] for i in range(3)]
    test_b_full = [[2.0, 4.0] for i in range(3)]
    weight_full = [[1.0, 2.0] for i in range(3)]

    [mse_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]

    # unfurl the list
    test_a_full = np.array(test_a_full).flatten()
    test_b_full = np.array(test_b_full).flatten()
    weight_full = np.array(weight_full).flatten()
    print(np.sum((test_a_full - test_b_full) ** 2) / len(test_a_full))
    print(np.sum(weight_full * (test_a_full - test_b_full) ** 2) / np.sum(weight_full))

    sklearn_mse = mean_absolute_error(test_a_full, test_b_full)
    sklearn_mse_w = mean_absolute_error(
        test_a_full, test_b_full, sample_weight=weight_full
    )

    metric_mse = mse_obj.compute()
    assert np.allclose(metric_mse, sklearn_mse), "MAE unweighted not equal"

    mse_obj = Metrics_WeightedMAE(reduction="mean")
    weight = torch.tensor([1.0, 2.0])
    [mse_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]

    metric_mse = mse_obj.compute()
    assert np.allclose(metric_mse, sklearn_mse_w), "MAE weighted not equal"


def test_accuracy():
    test_a = torch.tensor([[1, 0], [0, 1], [0, 1]])
    test_b = torch.tensor([[0, 1], [0, 1], [0, 1]])
    weight = torch.tensor([1.0, 1.0, 1.0])

    test_a_full = [[0, 1, 1] for i in range(3)]
    test_b_full = [[1, 1, 1] for i in range(3)]
    weight_full = [[1.0, 2.0, 3.0] for i in range(3)]

    test_a_full = np.array(test_a_full).flatten()
    test_b_full = np.array(test_b_full).flatten()
    weight_full = np.array(weight_full).flatten()

    acc_obj = Metrics_Accuracy_Weighted(reduction="mean")
    [acc_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]
    acc_sklearn = accuracy_score(test_a_full, test_b_full)
    acc_sklearn_w = accuracy_score(test_a_full, test_b_full, sample_weight=weight_full)
    acc_metric = acc_obj.compute()

    assert np.allclose(acc_metric, acc_sklearn), "Accuracy unweighted not equal"

    acc_obj = Metrics_Accuracy_Weighted(reduction="mean")
    weight = torch.tensor([1.0, 2.0, 3.0])
    [acc_obj.update(preds=test_a, target=test_b, weight=weight) for i in range(3)]
    acc_metric = acc_obj.compute()
    assert np.allclose(acc_metric, acc_sklearn_w), "Accuracy weighted not equal"


def test_cross_entropy():
    test_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    test_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    test_a_full = torch.tensor(np.array([[1.0, 0.0] for i in range(18)]))
    # create a list of lists alternating between [0,1] and [1,0]
    test_b_full = [[0.0, 1.0] if i % 2 == 0 else [1.0, 0.0] for i in range(18)]
    test_b_full = torch.tensor(np.array(test_b_full))

    weight = torch.tensor([1.0, 2.0])

    ce_obj = Metrics_Cross_Entropy(reduction="sum", n_categories=2)
    ce_obj_w = Metrics_Cross_Entropy(reduction="sum", n_categories=2)

    [ce_obj.update(preds=test_a, target=test_b) for i in range(9)]
    [ce_obj_w.update(preds=test_a, target=test_b, weight=weight) for i in range(9)]

    ce_metric = ce_obj.compute()
    ce_metric_w = ce_obj_w.compute()

    ce_sklearn = F.cross_entropy(
        input=test_a_full.double(),
        target=torch.argmax(test_b_full, axis=1).long(),
        reduction="sum",
    )
    ce_sklearn_w = F.cross_entropy(
        input=test_a_full.double(),
        target=torch.argmax(test_b_full, axis=1).long(),
        reduction="sum",
        weight=weight.double(),
    )

    assert np.allclose(ce_metric, ce_sklearn), "Cross Entropy unweighted not equal"
    assert np.allclose(ce_metric_w, ce_sklearn_w), "Cross Entropy weighted not equal"
