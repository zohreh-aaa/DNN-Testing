# Black-Box Testing of Deep Neural Networks through Test Case Diversity

This repository is a companion page for the following paper 
> "Black-Box Testing of Deep Neural Networks through Test Case Diversity".

> Zohreh Aghababaeyan (uOttawa, Canada), Manel Abdellatif (uOttawa, Canada), Lionel Briand (uOttawa, Canada), Ramesh S (General Motors, USA), and Mojtaba Bagherzadeh (uOttawa, Canada)

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).

The focus of the paper is on black-box diversity metrics and comparing them with all white-box existing coverage metrics in terms of fault detection abilities.
We have five research questions in our papar, based on that you can find the code of RQ1 and RQ4 in RQ1.ipynb and RQ4.ipynb the main part of the paper is related to RQ2,RQ3 and RQ5, you can access to them via `Final Testing Experimnet.ipynb` and the last file is `Conf and validation.ipynb` is one of the required step for answering smoe concerns regarding the validity of our work and validity of fault definition.

Our main contributions are:
1- Proposing diversity metrics (GD, STD, NCD) in the context of testing DNNs
2- Proposing and validating the method for approximating faults in DNNs
3- Comparing existing coverage metrics (LSC,DSC, NC, KMNC, TKNC, NBS, SNAC) with diversity metrics in terms of computation time and fault detection abilities 


DNNs faults are determined and saved for six different combinations of models & datasets (LeNet1 & MNIST, LeNet5 & MNIST, LeNet4 & Fashion_mnist, 12_Conv_layer & cifar10, LeNet5 & SVHN, ResNet20 & cifar10).

In our paper, We rely on counting faults instead of calculating misprediction rate since this is misleading in the context of testing the models (See paper figure 2).
Below is the workflow of our method for fault definition in DNNs.
![image](https://user-images.githubusercontent.com/58783738/146591442-346cd4ec-44e7-4933-ac08-6e991f78eef8.png)



* [sadl11](sadl11/) folder contains some parts of [1] for computing the LSC and DSC coverage metrics and pre-trained models.

Requirements
---------------
You need to first install these Libraries:
  - `!pip install umap-learn`
  - `!pip install tslearn`
  - `!pip install hdbscan`

The code was developed and tested based on the following environment:

- python 3.8
- keras 2.7.0
- Tensorflow 2.7.1
- pytorch 1.10.0
- torchvision 0.11.1
- matplotlib
- sklearn

---------------
Here is a documentation on how to use this replication package.

### Getting started

1. First, you need to upload the repo on your Google drive and run the codes with https://colab.research.google.com/.
2. The main code that you need to run is `Final Testing Experimnet.ipynb`. This code covers all the datasets and models that we used in the paper, however if you want to test the code on ther models and datasets which are not used in our paper, you need to change two lines of the code in `sadl11/run.py` that are related to the loading model and the selected layer. 
To do so please:

Change these lines :

`model= load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet1.h5")`

`layer_names = ["conv2d_1"]`

With your desired model and datset.  


Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    Replication-package
     .
     |
     |--- sadl11/model/                    Pre-trained models used in the paper (LeNet-1, LeNet-4, LeNet-5, 12-Layer ConvNet, ResNet20)
     |
     |--- RQ2-3/Correlation/               Random samples (60 subsets with sizes of 100,...,1000) to replicate the paper's results
     |
     |--- RQ4/                             The preprocessing time related to VGG feature extaction on MNIST dataset             
  

Research Questions
---------------
Our experimental evaluation answers the research questions below.

_**1- RQ1: To what extent are the selected diversity metrics measuring actual diversity in input sets?**_


*To directly evaluate the capability of the selected metrics to actually measure diversity in input sets, we study how diversity scores change while varying, in a controlled manner, the number of image classes covered by the input sets.*

<img width="929" alt="Diversity" src="https://user-images.githubusercontent.com/58783738/146585778-6dd7c17c-c8f8-4c6c-bda3-316e20e871b9.png">

-->Outcome:  GD and STD showed good performance in measuring actual data diversity in all the studied datasets. This is not the case of NCD, which we exclude from the following experiments.


_**2- RQ2: How does diversity relate to fault detection?**_

*We aim to study whether higher diversity results in better fault detection. For this purpose, we randomly select, with replacement, 60 samples of sizes 100, 200, 300, 400, 1000. For each sample, we calculate the diversity scores and the number of faults. Finally, we calculate the correlation between diversity scores and the number of faults.*

-->Outcome: There is a moderate positive correlation between GD and faults in DNNs. GD is more significantly correlated to faults than STD. Consequently, GD should be used as a black-box approach to guide the testing of DNN models.


_**3- RQ3: How does coverage relate to fault detection?**_

*We aim to study the correlation between state-of-the-art coverage criteria and faults in DNNs.*

-->Outcome: In general, there is no significant correlation between DNN coverage and faults for the natural dataset. LSC coverage showed a moderate positive correlation in only one configuration.

(RQ2 and RQ3 results)

![image](https://user-images.githubusercontent.com/58783738/146563862-579f5227-450d-432d-a1ae-9c27c10f1781.png)


_**4- RQ4: How do diversity and coverage metrics perform in terms of computation time?**_

*In this research question, we aim to compare the computation time of diversity and coverage metrics.*

These are the results of computation time related to:
Dataset: Cifar10 
Model:  Resnet20
![cifar10res](https://user-images.githubusercontent.com/58783738/172948607-c090de7b-ffa1-485d-9e68-118e87f3a8e1.png)


Dataset: MNIST
Model:  LeNet5
![Screenshot (352)](https://user-images.githubusercontent.com/58783738/172949611-83082b46-ec9e-40ef-99b5-da982a85c174.png)

Note that Computation time of NBC and SNAC are the same as KMNC.

--> Outcome: Both diversity and coverage metrics are not computationally expensive. However, in general GD  significantly outperforms coverage criteria. In application contexts, such as test case selection and minimization, based for example on search where we can expect to perform many test set evaluations, this difference can become practically significant. 


_**5- RQ5. How does diversity relate to coverage?**_

*We want to study in this research question the relationship between diversity and coverage to assess if diverse input sets tend to increase the coverage of DNN models.*

![image](https://user-images.githubusercontent.com/58783738/146586359-b531c770-354e-4941-a6d5-8006b5f1dcb5.png)

--> Outcome: In general, there is no significant correlation between diversity and coverage in DNN models.

Notes
-----

1- We used the same recommended settings of LSC and DSC hyperparameters (upper bound, lower bound, number of buckets, etc.) as in the original paper for the different models and datasets in our experiments.

2- For the LeNet1 model, we use the same setting as the LeNet5 model.

3- For speed-up, you can use GPU-based TensorFlow by changing the Colab Runtime.

References
-----
1- [Surprise Adequacy](https://github.com/coinse/sadl)

2- [DBCV](https://github.com/christopherjenness/DBCV)

