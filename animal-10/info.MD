1. Download the animal-10 dataset from https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. To use please make sure to follow the same folder structure -> Dataset Folder -> All Classes Folder -> Images belonging to same class
3. All the model is pretrained using imagenet
4. Last linear layer was removed to get features :: most model extract 2048 features :: please verify once for different model
5. Code will generate pickle dataframe files for different models.

Next Step using the pickled dataframe calculate distance between two images with same and different classes and tune threshold.
