# Indian-Scene-Text-Classification

The Indian scene text classification model is developed as part of the work towards [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board) by [AI4Bharat](https://ai4bharat.org/). I worked on this project under the mentorship of [Mitesh Khapra](http://www.cse.iitm.ac.in/~miteshk/) and [Pratyush Kumar](http://www.cse.iitm.ac.in/~pratyush/) from IIT Madras.

Indian Signboard Translation  involves 4 modular tasks:
1. *`T1`: Detection:* Detecting bounding boxes containing text in the images
2. **`T2`: Classification:** Classifying the language of the text in the bounding box identifed by `T1`
3. *`T3`: Recognition:* Getting the text from the detected crop by `T1` using the `T2` classified recognition model
4. *`T4`: Translation:* Translating text from `T3` from one Indian language to other Indian language

![Pipeline for sign board translation](../master/Images/Pipeline.jpg)
> Note: `T2`: Classification is not updated in the above picture


# Dataset

[Indian Scene Text Classification Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset/blob/master/README.md#d2-classification-dataset) is used to train this model (`D2` + `D2-English`)


# Model
A modifed version of Convolutional Recurrent Neural Network Model ([CRNN](https://arxiv.org/pdf/1507.05717v1.pdf)) is used to architect the classification model.
The model uses resnet-18 as the feature extractor of images (initialised with pretrained weights on ImageNet). Then the bidirectional gated recurrent units are used to learn from the spatially sequential output of the former CNN part. Finally, a linear output layer is used to classify the language taking flattened input from the sequential features output of the RNN part.

* Input Image Shape: [200, 50]
* CNN Output Shape: [13, 256]
* RNN Output Shape: [13, 16]
* Linear Output Shape: [5]

# Training
The classification model is trained for 30 epochs with the following hyperpararmeters. The model weights are saved every 3 epochs and you can find them in the [`Models`](../master/Models/) directory

* train_batch_size = 64
* lr = 0.0001
* weight_decay = 0.01
* lr_step_size = 5
* lr_gamma = 0.9

For detailed model architecture and its parameters, check the `Define model` section of the notebook [Language-Classification.ipynb](../master/Language-Classification.ipynb)

![Training Loss](../master/Images/Training.png) 


# Performance
The lowest validation loss is observed in epoch 24. Hence, the model [`Models/Language-Classifier-e24.pth`](../master/Models/Language-Classifier-e24.pth) is used to evaluate the classification performance. 

|Train Accuracy |Val Accuracy |Test Accuracy |
|:-------------:|:-----------:|:------------:|
|0.99           |0.94         |0.95          |

Check for the language confusion matrix of the testset below:

![Confusion Matrix](../master/Images/Confusion-Matrix.png) 

As there are high similarities among the characters of `Tamil & Malayalam` and `Hindi & Punjabi` over other language pairs, there are many misclassfications among these pairs.

**Misclassification Samples in Testset:**

![Misclassification](../master/Images/Misclassification.png) 


# Code
* [Language-Classification.ipynb](../master/Language-Classification.ipynb)


### Related Links:
1. [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board)
2. [Indian Scene Text Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset)
3. [Indian Scene Text Detection](https://github.com/GokulKarthik/Indian-Scene-Text-Detection)
4. [Indian Scene Text Classification](https://github.com/GokulKarthik/Indian-Scene-Text-Classification)
5. [Indian Scene Text Recognition](https://github.com/GokulKarthik/Indian-Scene-Text-Recognition)

### References:
1. https://arxiv.org/pdf/1507.05717v1.pdf
2. https://arxiv.org/pdf/1512.03385.pdf
3. https://github.com/GokulKarthik/Deep-Learning-Projects.pytorch
4. https://github.com/carnotaur/crnn-tutorial/
