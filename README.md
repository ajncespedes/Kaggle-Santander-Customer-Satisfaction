# Kaggle-Santander-Customer-Satisfaction

This repository contains the best code that I developed for the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction). It is written in R. The code generates a model which get 0.840336 in the public leader-board and 0.828114 in the private leader-board, which means Rank 12. 
At the end of the contest, I got rank 548 but I realised that I had a model that would have ranked about 12th. So in this repository I will try to improve that model as much as I can.


# Description

From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

# Model created

First of all, the dataset have repeated instances with different class labels, which means noise. So, I extracted the noisy instances and splitted the others instances into 5 folds. Finally, I made an ensemble to predict the noisy instances and I inserted them with the real class label into the train data. 

The dataset is highly unbalance, so I decided to separate the instances which have 0 class and 1 class in order to make number_of_class0/number_of_class1 folds (undersampling) of 0 class joined with the same 1 class instances.  After that, I run the 24 resulting models with xgboost and finally I made a simple average ensemble.
