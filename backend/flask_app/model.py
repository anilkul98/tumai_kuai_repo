import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from dataset import SatImageDataset
from land_selector import LandSelector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("checkpoints/26.pth", map_location=torch.device(device))
transform = T.Compose([T.Resize(128), 
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])


def transform_img(img):
    transform = T.Compose([T.Resize(128), 
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    img = Image.fromarray(img)
    img_tensor = transform(img)
    return img_tensor

def preprocess(divided_imgs, divided_imgs_names, batch_size=8):
    divided_imgs = np.array(divided_imgs)
    divided_imgs_names = np.array(divided_imgs_names)
    remaining_imgs = divided_imgs[-(len(divided_imgs)%batch_size):]
    splitted_imgs = np.split(divided_imgs[:(len(divided_imgs) - len(divided_imgs)%batch_size)], batch_size)
    splitted_imgs.append(remaining_imgs)
    batch_list_imgs = [torch.stack(tuple(split)) for split in splitted_imgs]
    
    remaining_names = divided_imgs_names[-(len(divided_imgs_names)%batch_size):]
    splitted_names = np.split(divided_imgs_names[:len(divided_imgs_names) - len(divided_imgs_names)%batch_size], batch_size)
    splitted_names.append(remaining_names)
    return batch_list_imgs, splitted_names

def get_cost_matrix(preds):
    land_selector = LandSelector(length_labels=10, length_grids=39)
    grouped_data, valid_grid_coordinates = land_selector.organize_prediction_data(preds)
    land_selector.cost_function(grouped_data)
    cost_matrix_filtered = land_selector.get_land_scores()
    # np.savetxt('cost_matrix.txt', cost_matrix, fmt='%1.4e')
    return cost_matrix_filtered

def get_prediction(divided_imgs):
    dataset = SatImageDataset(divided_imgs, transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=39,
                            num_workers=2,
                            shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch
            images = images.to(device)
            pred = model.forward(images)
            pred = pred.detach().cpu().numpy().astype(np.number)
            pred = np.argmax(pred, axis=1)
            preds.extend(list(pred))
    print(set(preds))
    preds = np.array(preds)
    preds = np.split(preds, 39)
    preds = [elt.tolist() for elt in preds]

    print("prediction done")
    # print(preds)
    cost_matrix_filtered = get_cost_matrix(np.array(preds))
    print("Cost matrix done")
    
    return preds, cost_matrix_filtered