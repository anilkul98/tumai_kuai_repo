import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
import json
import wandb
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataset import SatImageDataset
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from utils import ConfusionMatrixTracker
from model import get_model

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is using")
checkpoints_folder = "checkpoints/efficientnet_b3"
log_file_path = os.path.join(checkpoints_folder, "log_weighted.txt")

wandb.init(project="tumai", entity="ku20-21")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 256
}

def checkpoint(checkpoints_folder, model, epoch):

    os.makedirs(checkpoints_folder, exist_ok=True)

    model_name = "{}.pth".format(epoch)
    checkpoint_final_path = os.path.join(checkpoints_folder, model_name)

    torch.save(model, checkpoint_final_path)

    print("Checkpoint saved to {}".format(checkpoint_final_path))

def train_step(model, optimizer, train_loader, criterization, le, num_classes):
    losses = []
    model.train()
    cmt = ConfusionMatrixTracker(num_classes=num_classes, 
        class_names=le.inverse_transform(list(range(num_classes))))
    for batch in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model.forward(images)
        loss = criterization(preds, labels.view(-1))

        loss.backward()
        optimizer.step()
        loss = loss.item()
        losses.append(loss)
        with torch.no_grad():
            cmt.update_state(labels, torch.argmax(preds, dim=1))
    epoch_loss = np.mean(losses)
    return cmt, epoch_loss

def test_step(model, test_loader, criterization, le, num_classes):
    test_losses = []
    cmt = ConfusionMatrixTracker(num_classes=num_classes, 
        class_names=le.inverse_transform(list(range(num_classes))))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model.forward(images)
            loss = criterization(preds, labels.view(-1))
            loss = loss.item()
            test_losses.append(loss)
            cmt.update_state(labels, torch.argmax(preds, dim=1))
    epoch_loss = np.mean(test_losses)
    return cmt, epoch_loss

with open("rgb_paths.txt") as fp:
    lines = fp.readlines()

lines = [l.replace("\n", "") for l in lines]
img_paths = [line.split(",")[0] for line in lines]
labels = [line.split(",")[1] for line in lines]

le = LabelEncoder()
le.fit(labels)
encoded_labels = le.transform(labels)
pickle.dump(le, open('le.pkl', 'wb'))
# To get back the labels:
# le = pickle.load(open('le.pkl', 'rb'))
# str_labels = le.inverse_transform(arr)

num_classes = len(list(le.classes_))

train_paths, test_paths, train_labels, test_labels = \
    train_test_split(img_paths, encoded_labels, test_size=0.2, random_state=61)

train_transform = T.Compose([T.Resize(128),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.RandomRotation(45),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

test_transform = T.Compose([T.Resize(128), 
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

train_dataset = SatImageDataset(train_paths, train_labels, train_transform)
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=64,
                        num_workers=2,
                        shuffle=True)

test_dataset = SatImageDataset(test_paths, test_labels, test_transform)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=256,
                        num_workers=2)

model = get_model("efficientnet", num_classes)
model.to(device)
model.train()

criterization = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)




general_metrics = {"accuracy": [],
                    "losses": []}
for epoch in range(50):
    print("Epoch: {}".format(epoch))
    cmt_train, train_loss = train_step(model, optimizer, train_loader, criterization, le, num_classes)
    checkpoint(checkpoints_folder, model, epoch) 
    cmt_test, test_loss = test_step(model, test_loader, criterization, le, num_classes)
    scheduler.step()

    log_metrics_train = cmt_train.log_metrics()
    log_metrics_test = cmt_test.log_metrics()
    wandb.log({"train_loss": train_loss,
                "val_loss": test_loss,
                "train_acc": log_metrics_train["Accuracy"],
                "val_acc": log_metrics_test["Accuracy"],
                "train_average_f1": log_metrics_train["Average F1"],
                "val_average_f1": log_metrics_test["Average F1"],
            })

    metrics={}
    metrics.update({"Train " + k: v for k, v in cmt_train.log_metrics().items()})
    metrics.update({"Val " + k: v for k, v in cmt_test.log_metrics().items()})
    
    losses ={}
    losses["Train Loss"] = train_loss
    losses["Val Loss"] = test_loss

    with open(log_file_path, "a") as fp:
        fp.write("""[Epoch {}]\n""")
        fp.write("""Test Conf. Matrix\n{}\nTest Acc: {}\n\n""". format(cmt_test.cm, metrics["Val Accuracy"]))
        fp.write("""\nTrain loss: {}\t Val Loss:{}\n""".format(train_loss,test_loss))
        
    print("""[Epoch {}]\n""")
    print("""Test Conf. Matrix\n{}\nTest Acc: {}\n\n""". format(cmt_test.cm, metrics["Val Accuracy"]))
    print("""\nTrain loss: {}\t Val Loss:{}\n""".format(train_loss,test_loss))
    
    general_metrics["accuracy"].append((str(metrics["Train Accuracy"]), str(metrics["Val Accuracy"])))
    general_metrics["losses"].append((str(losses["Train Loss"]), str(losses["Val Loss"])))
    print(general_metrics)
    with open(os.path.join(checkpoints_folder, "metrics.json"), "w") as fp:
        json.dump(general_metrics, fp)