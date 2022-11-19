1. Download the animal-10 dataset from https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. To use please make sure to follow the same folder structure -> Dataset Folder -> All Classes Folder -> Images belonging to same class
3. Install Python Dependencies
4. Run feature extraction script
5. All the model is pretrained using imagenet
6. Last linear layer was removed to get features :: most model extract 2048 features :: please verify once for different model
7. Code will generate pickle dataframe files for different models.
8. After successfull completion run the verification script. (By Default Threshold is set as 0.33)
9. Dataframe with distance will be saved in output folder as pickle.
