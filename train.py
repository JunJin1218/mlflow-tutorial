# log_run.py

import os, time, numpy as np, torch
import mlflow, mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities import Metric, Param
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Databricks MLflow Connection + Extract Experiment ID
mlflow.set_tracking_uri("databricks")
exp_name = os.environ["DATABRICKS_EXPERIMENT"]  # "/Users/<email>/your-exp"
client = MlflowClient()
exp = client.get_experiment_by_name(exp_name) or client.get_experiment(
    client.create_experiment(exp_name)
)
exp_id = exp.experiment_id

# MNIST / Model
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST(root="./data", train=True, download=True, transform=tfms)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=tfms)

val_ratio = 0.1
val_size  = int(len(train_full) * val_ratio)
train_size= len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

# This is how to spawn a model with Pytorch...
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                # 14x14
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                # 7x7
    nn.Flatten(),
    nn.Linear(64*7*7, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
model.to(device)
opt = optim.Adam(model.parameters(), lr=3e-4) # Optimizer setting
loss_fn = nn.CrossEntropyLoss() # Loss function

# Train loop
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    tot, ok, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * xb.size(0)
        ok += (logits.argmax(1) == yb).sum().item()
        tot += xb.size(0)

    train_loss = loss_sum / tot
    train_acc  = ok / tot

# ---- val ----
model.eval()
v_tot, v_ok, v_loss_sum = 0, 0, 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        v_loss_sum += loss.item() * xb.size(0)
        v_ok += (logits.argmax(1) == yb).sum().item()
        v_tot += xb.size(0)
val_loss = v_loss_sum / v_tot
val_acc  = v_ok / v_tot

print(f"[{epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

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
        Metric("val_loss", float(val_loss), ts, 0),
        Metric("val_acc", float(val_acc), ts, 0),
    ]

    # Signiture + input_example
    xb, yb = next(iter(train_loader))  # 배치 하나 꺼냄
    input_example = xb[:1].cpu().numpy().astype(np.float32)
    with torch.no_grad():
        sig_pred = model(xb[:1].to(device))
    signature = infer_signature(input_example, sig_pred.cpu().numpy())

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
