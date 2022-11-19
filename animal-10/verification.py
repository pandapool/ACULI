import pickle, random, os
import numpy as np
import logging
logging.basicConfig(filename="verfication.log",
                    filemode='a',
                    level=logging.INFO)

def get_distance(img1_embedding, img2_embedding):
    
    num=np.dot(img1_embedding,img2_embedding)
    distance=[1]-num/(np.linalg.norm(img1_embedding)*np.linalg.norm(img2_embedding))
    return distance

THRESHOLD = 0.33
folder = "csv"
files = os.listdir(folder)
logging.info(f"Threshold set as {THRESHOLD}")
if not os.path.isdir("output"):
    os.mkdir("output")
    
for file in files:
    logging.info(f"Processing file : {file}")
    path = os.path.join(folder, file)
    df = pickle.load(open(path, "rb"))
    df['match_embed'] = 0
    df['match_path'] = 0
    df['mismatch_embed'] = 0
    df['mismatch_path'] = 0
    df['same_distance'] = 0
    df['different_distance'] = 0
    

    for i in range(df.shape[0]):
        cond1 = random.sample(list(df[(df['label'] == df['label'].iloc[i]) & (df['path'] != df['path'].iloc[i])].index),1)[0]
        df['match_embed'].iloc[i] = df['embedding'].iloc[cond1]
        df['match_path'].iloc[i] = df['path'].iloc[cond1]
        
        cond2 = random.sample(list(df[(df['label'] != df['label'].iloc[i])].index),1)[0]
        df['mismatch_embed'].iloc[i] = df['embedding'].iloc[cond2]
        df['mismatch_path'].iloc[i] = df['path'].iloc[cond2]
   
        df['same_distance'].iloc[i] = get_distance(df['embedding'].iloc[i][0],df['match_embed'].iloc[i][0])
        df['different_distance'].iloc[i] = get_distance(df['embedding'].iloc[i][0],df['mismatch_embed'].iloc[i][0])
    
    df["same"] = df["same_distance"].apply(lambda x: 0 if x > THRESHOLD else 1)
    df["different"] = df["different_distance"].apply(lambda x: 1 if x >= THRESHOLD else 0)
    pickle.dump(df, open(f"output/{file}", "wb"))
    logging.info(f"{file} :: {df.describe()}")
