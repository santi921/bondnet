import torch
from torch.nn import functional as F
import numpy as np
from math import log2
from bondnet.model.metric import (
    Metrics_Accuracy_Weighted,
    Metrics_Cross_Entropy,
)
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def test_mse():
    mse_obj = MeanSquaredError(squared=True)
    test_data = torch.tensor([1.0, 2.0])
    test_pred = torch.tensor([2.0, 4.0])
    # repeat the test data 3 times
    test_data_3 = torch.tensor([[1.0, 2.0] for i in range(3)]).flatten()

    # get mean of data
    test_data_mean = torch.mean(test_data_3, axis=0)
    # get std of data
    test_data_std = torch.std(test_data_3, axis=0)

    # scale the data
    test_data_batch_scale = (test_data - test_data_mean) / test_data_std
    test_pred_batch_scale = (test_pred - test_data_mean) / test_data_std

    [
        mse_obj.update(preds=test_pred_batch_scale, target=test_data_batch_scale)
        for i in range(3)
    ]
    # unfurl the list

    metric_mse = mse_obj.compute()
    # compute mse for data that was normalized
    weighted_mse = metric_mse * test_data_std * test_data_std
    print(weighted_mse)
    assert np.allclose(weighted_mse, 2.5), "MSE weighted not equal"


def test_mae():
    mae_obj = MeanAbsoluteError()
    test_data = torch.tensor([1.0, 2.0])
    test_pred = torch.tensor([2.0, 4.0])
    # repeat the test data 3 times
    test_data_3 = torch.tensor([[1.0, 2.0] for i in range(3)]).flatten()

    # get mean of data
    test_data_mean = torch.mean(test_data_3, axis=0)
    # get std of data
    test_data_std = torch.std(test_data_3, axis=0)

    # scale the data
    test_data_batch_scale = (test_data - test_data_mean) / test_data_std
    test_pred_batch_scale = (test_pred - test_data_mean) / test_data_std

    [
        mae_obj.update(preds=test_pred_batch_scale, target=test_data_batch_scale)
        for i in range(3)
    ]

    metric_mae = mae_obj.compute()
    # assert that all the values of the weight are the same
    weighted_metric_mae = metric_mae * test_data_std
    print(weighted_metric_mae)
    assert np.allclose(weighted_metric_mae, 1.5), "MAE weighted not equal"


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


test_mae()
test_mse()
