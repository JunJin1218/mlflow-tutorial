# log_run.py

import os, time, numpy as np, torch
import mlflow, mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities import Metric, Param
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Databricks MLflow Connection + Extract Experiment ID
mlflow.set_tracking_uri("databricks")
exp_name = os.environ["DATABRICKS_EXPERIMENT"]  # "/Users/<email>/your-exp"
client = MlflowClient()
exp = client.get_experiment_by_name(exp_name) or client.get_experiment(
    client.create_experiment(exp_name)
)
exp_id = exp.experiment_id

# Toy Data / Model
N, D, C = 4096, 32, 2
X = torch.randn(N, D).float()
y = (X.sum(1) > 0).long()
ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=128, shuffle=True)

model = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, C)) # This is how to spawn a model with Pytorch...
opt = optim.Adam(model.parameters(), lr=3e-4) # Optimizer setting
loss_fn = nn.CrossEntropyLoss() # Loss function

# Train loop
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    tot, ok, loss_sum = 0, 0, 0.0
    for xb, yb in dl:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        ok += (logits.argmax(1) == yb).sum().item()
        tot += xb.size(0)
    train_loss = loss_sum / tot
    train_acc = ok / tot
    print(f"[{epoch}] loss={train_loss:.4f} acc={train_acc:.4f}")

# Evaluation (reused training set)
model.eval()
with torch.no_grad():
    logits = model(X)
    pred = logits.argmax(1)
    test_acc = (pred == y).float().mean().item()
    test_loss = loss_fn(logits, y).item()

# MLflow run + Batch logging
with mlflow.start_run(run_name="demo-batch-log", experiment_id=exp_id) as run:
    run_id = run.info.run_id

    # Parameters
    model_params = {"hidden": 64, "act": "ReLU"}  # toy values
    other_params = [
        Param("epochs", str(epochs)),
        Param("batch_size", str(128)),
        Param("lr", str(3e-4)),
        Param("device", str("cuda" if torch.cuda.is_available() else "cpu")),
    ]
    mlflow.log_params(model_params)  # 많은 파라미터는 이쪽으로 한 번에

    # Metrics (ms recommended)
    ts = int(time.time() * 1000)
    metrics = [
        Metric("train_loss_final", float(train_loss), ts, epochs),
        Metric("train_acc_final", float(train_acc), ts, epochs),
        Metric("test_loss", float(test_loss), ts, 0),
        Metric("test_acc", float(test_acc), ts, 0),
    ]

    # Signiture + input_example
    input_example = X[:1].numpy().astype(np.float32)
    with torch.no_grad():
        sig_pred = model(torch.from_numpy(input_example))
    signature = infer_signature(input_example, sig_pred.numpy())

    # Batch Logging
    client = MlflowClient()
    client.log_batch(run_id=run_id, metrics=metrics, params=other_params)

    # (e) 모델 저장/등록 (Unity Catalog 쓰면 registry URI를 "databricks-uc"로 설정)
    # mlflow.set_registry_uri("databricks-uc")  # UC 쓸 때만 해제
    info = mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        # registered_model_name="catalog.schema.model_name"  # UC 사용 시
        # registered_model_name="MyTorchModel"              # 워크스페이스 레지스트리
    )
    print("model_uri:", info.model_uri)

print("done.")
