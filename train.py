import mlflow, mlflow.pytorch
import dotenv

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/<YOU>/experiments/pytorch-demo")  # 워크스페이스 경로

mlflow.pytorch.autolog(log_models=True)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

with mlflow.start_run(run_name="uv-local->dbx"):
    params = {"lr": 3e-4, "epochs": 5, "batch_size": 128}
    mlflow.log_params(params)

    X = torch.randn(4096, 32)
    y = (X.sum(dim=1) > 0).long()
    dl = DataLoader(TensorDataset(X, y), batch_size=params["batch_size"], shuffle=True)

    model = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,2))
    opt = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, params["epochs"]+1):
        tot, ok, loss_sum = 0, 0, 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item()*xb.size(0)
            ok += (logits.argmax(1)==yb).sum().item()
            tot += xb.size(0)
        mlflow.log_metrics({"loss": loss_sum/tot, "acc": ok/tot}, step=epoch)