from torchvision.models import vgg11, vgg16, vgg19, resnet18, resnet50, resnet101, resnet152
import os, pickle, torch, cv2
import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename="feature_extraction.log",
                    filemode='a',
                    level=logging.INFO)
device = "cuda"

model_dict = {
        "resnet18" : resnet18(pretrained=True),
        "resnet50" : resnet50(pretrained=True),
        "resnet101" : resnet101(pretrained=True),
        "resnet152" : resnet152(pretrained=True),
        "vgg11" : vgg11(pretrained=True),
        "vgg16" : vgg16(pretrained=True),
        "vgg19" : vgg19(pretrained=True),
             }
logging.info("Model Dictionary Defined")
def remove_linear_layer(model_dict):
    dict1 = {}
    for key, value in model_dict.items():
        dict1[key] = torch.nn.Sequential(*(list(value.children())[:-1]))
    return dict1

def eval_models(model_dict):
    for key, value in model_dict.items():
        value = value.to(device)
        value.eval()
        
model_dict = remove_linear_layer(model_dict)
logging.info("Removed Linear Layer")

eval_models(model_dict)
logging.info(f"Model Eval using {device}")

def preprocess(img_path, model):
    # Reading the image
    img = cv2.imread(img_path)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    torch_output = model(img.to(device))
    return torch_output[0, :, 0, 0].cpu().detach().numpy()

main_folder = "animal10/raw-img/"
animals = os.listdir(main_folder)
dict1 = {}
for folder in animals:
    path = os.path.join(main_folder, folder)
    images = os.listdir(path)
    images_path = []
    for image in images:
        img_path = os.path.join(path, image)
        images_path.append(img_path)
    dict1[folder] = images_path
    
dict2 ={}
for key, value in dict1.items():
    for path in value:
        dict2[path] = key
        
df = pd.DataFrame(zip(list(dict2.keys()),list(dict2.values())) , columns=['path', 'label'])
logging.info("Dataframe Defined")
if not os.path.isdir("csv"):
    os.mkdir("csv")
for key, value in model_dict.items():
    dummy = df.copy()
    dummy['embedding'] = 0
    for i in range(0, dummy.shape[0]):
        dummy['embedding'].iloc[i] = [preprocess(dummy['path'].iloc[i], value)]
        
    pickle.dump(dummy, open(f"csv/{key}_csv.pkl", "wb"))
    logging.info(f"{key} csv saved")
