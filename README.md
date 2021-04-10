# deep_learning_templates

The repo contains four root folders labeled:

- 01_prepare.tools
- 02_deep_learning
- data

## 01_prepare.tools

Follow the guidelines set in prepare_tools_guidelines.html to setup your machine to perform deep learning in both R and Python. 
The install.packages.r file contains the packages you will need to perform deep learning in R. 
_note_ that you could unsleep the commands in the script at lines 296-298 to install the required python modules. 
However the process is exhaustive due to R's reliance on memory. 
Some packages will install from the packages root in-order to load specific versions. 
To properly setup Python follow the guidelines set in prepare_tools_guidelines.html. 

## 02_deep_learning

There are several templates contained in the root for supervised deep learning using structured and unstructured data.

- dl_super_classification.r
- dl_super_image_classification.py
- dl_super_image_classification.r
- dl_super_regression.py
- dl_super_regression.r
- dl_super_text_classification.r

### dl_super_classification.r

For supervised classification problems using structured data sources should use this template.
The current script is setup for a binary classification problem but can quickly be adjusted to input a multi-classification problem. 
All that is needed to switch the template to multi is the adjustment line 85 and 88. 
On line 85 change the units = to the number of classes in the label column.
You should also make sure that the class column is the last variable in the data frame. 
You also need to change the activation to 'softmax'.
On line 88 you need to change the loss function to intake multiple classes, use a categorical_crossentropy fucntion. 

### dl_super_image_classification.py

The template provides a framework to perform binary image classification in python. 

### dl_super_image_classification.r

The template provides a frame work to train a binary supervise image classifier. 
With a few changes the template can quickly be adjusted for multi-classifcation. 
Re-design the image root to a multi-class scheme and make sure to edit line 10 in the script to a list that encompasses all the possible labels. 

### dl_super_regression.r & dl_super_regression.py

These scripts are templates to perform deep learning on supervised regression problems. 

### dl_super_text_classification.r

A simple template for natural language processing and deep binary text classification using word vector spaces. 
