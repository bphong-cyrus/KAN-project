import torch


def rmse(model, x, target):
    """Đo RMSE mà không cập nhật mô hình."""
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            prediction = model(x)

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Shape không khớp: {prediction.shape} và {target.shape}"
                )

            return (prediction - target).square().mean().sqrt().item()
    finally:
        model.train(was_training)