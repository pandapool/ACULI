# ACULI

The objective here is to create a model which will compare two images and see if the two images are similar or not, Ultimately acting like a clustering algorithm for a bunch of images without the need for training.

The workflow of the model is as follows:

   Input_image            Input_image
        |                      |
        |                      |
        |                      |
Generate_Features      Generate_Features
         \                    /
          \                  /
           \                /
            \              /
          Binary Classification
            model(Same/Not)
            

### Plans

1. Use VGG16, VGG19, Resnet50 models to generate features
2. Binary Classification Model to classify as same or not same