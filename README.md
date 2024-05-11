# Udacity-Image-Classification-using-AWS-SageMaker

In this project, I demonstrated the finetuning of the ResNet50 pre-trained model for classification of images from the CIFAR10 image dataset. 
The project is made up of the following steps:

1. Data uploading and preprocessing
2. Model def
3. Training def
4. Testing def
5. Hyperparamter Search
6. Debugger and Profiler setup
7. Model training
8. Endpoint deployment
9. Prediction

I used PyTorch for my training script and the SageMaker SDK to setup the training environment and deploy the trained model. 

## Data uploading and preprocessing

The CIFAR10 dataset can be found here: https://www.cs.toronto.edu/~kriz/cifar.html
"The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images."

The 10 classes are as follows:

airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck

Downloaded the dataset tar file to my S3 bucket. I then moved it to my notebook, where I unzipped it. I uploaded the subsets back to S3. 
The 5 training substes and 1 test subset are then downloaded again from wihtin in the training script, making them available in the containerized training environment. 

As each image is stored in row-major format, within a pickle file, I needed to unpickle each dataset, before then reformatting each row into a 3-channel, 32x32 image. 

Arrays of the reformatted images are then passed to train and test dataloaders, ready to be passed to the model.

## Model def

I have used a pre-trained ResNet50 model. 
In the model function, I have added a fully connected layer with dropout to the end of the model.

In addition to the FC layer, I provide a `trainable_layers` parameter to specifiy how many of the model's pre-trained layers can be finetuned. 

I also provide a `weight_path` parameter to be able to load the weights from previous model outputs. This allows previous work to be built upon, without having to start from scratch for each round of training. 

## Training def

The training function takes the following parameters:

`model` - the ResNet50 model as previously defined.  
`train_loader` - a dataloader containing 50000 training examples.  
`criterion` - the loss criterion, in this case CrossEntropyLoss.  
`optimizer` - in this case, Adam, with a learning rate passed via hyperparameters.  
`epoch` - number of epochs.  
`hook` - the SageMaker Debugger hook to be able to monitor training metrics and resource utilization.  

I was initially confused by the implied links from the model parameters being passed to the optimizer, the output from the model being passed to loss and then the loss and optimizer being called as separate steps:

`optimizer = optim.Adam(model.parameters(), lr=args.lr)`

```
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

In coming to understand how PyTorch manages back propogation, I came across this great discussion:

https://discuss.pytorch.org/t/how-does-loss-function-affect-model-during-training/187244

## Testing def

The test function takes the following parameters:

`model` - the ResNet50 model as previously defined.  
`test_loader` - a dataloader containing 10000 test examples.  
`criterion` - the loss criterion, in this case CrossEntropyLoss.  
`hook` - the SageMaker Debugger hook to be able to monitor training metrics and resource utilization.   

## Hyperparamter Search

The following search space was provided to the SageMaker `HyperparameterTuner`:

```
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128]),
    "epochs": IntegerParameter(1,20),
    "trainable_layers": IntegerParameter(0,3)
}
```

Not that `trainable_layers` refers to how many layers in the pretrained model we want to finetune. 

Two runs provided the following recomendations:

1.
```
{
 'batch_size': '"128"',
 'epochs': '2',
 'lr': '0.0028907087694516006',
 'trainable_layers': '2'
}
```

2.
```
{
 'batch_size': '"128"',
 'epochs': '2',
 'lr': '0.002859651084098427',
 'trainable_layers': '2'
}
```

As it would turn out, these were not good recomendations. The learning rate is too small, the batches too large and the number of epochs too small. 
I went on the test other values to settle on the following: 




## Debugger and Profiler setup

To setup SageMaker Debugger, the following configs need to be profided:

Rules:

```
rules = [
        #Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
        Rule.sagemaker(rule_configs.vanishing_gradient(),),
        Rule.sagemaker(rule_configs.overfit()),
        Rule.sagemaker(rule_configs.overtraining()),
        Rule.sagemaker(rule_configs.poor_weight_initialization()),
        #Rule.sagemaker(base_config=rule_configs.loss_not_decreasing())
]
```

DebuggerHookConfig:

```
hook_config = DebuggerHookConfig(
    #hook_parameters={"train.save_interval": "50", "eval.save_interval": "5"},
    collection_configs=[
    CollectionConfig(
        name="losses",
        parameters={
            "train.save_interval": "50",
            "eval.save_interval": "50"
        }
    )
]
)
```
Note: you need to use save_interval, not save_step. A lot of time was wasted figuring this out. For save_step, you need to explicitly define each step. Hence, if you pass a single value to save_step, it will only save data for that one step, and you will not see anything in your plot. 

ProfilerConfig:

```
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)
```

In your script, hooks for the training and testing function need to be defined and passed to those respective functions. 

## Model training

Training was done using a Sagemaker Pytorch Estimator, and done in two rounds:

```
estimator = PyTorch(
    entry_point="train_model.py",
    role=role,
    framework_version="1.8.1",
    py_version="py3",
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    hyperparameters={"epochs": 5, "num_classes": 10, "batch_size":64, "lr":0.01, "trainable_layers":1, "weight_path":weight_path},
    rules=rules,
    debugger_hook_config=hook_config,
    profiler_config=profiler_config,
    output_path=output_path 
)
```


### Round 1:

Hyperparameters:

`hyperparameters={"epochs": 10, "num_classes": 10, "batch_size":128, "lr":0.0028597, "trainable_layers":2, "weight_path":None}`

Result:

```
Train Epoch: 1 [12800/50000 (26%)]#011Loss: 20.077261
Train Epoch: 1 [25600/50000 (51%)]#011Loss: 3.753358
Train Epoch: 1 [38400/50000 (77%)]#011Loss: 2.934975
Test set: Average loss: 3.0589, Accuracy: 3200/10000 (32%)
Train Epoch: 2 [0/50000 (0%)]#011Loss: 2.769455
Train Epoch: 2 [12800/50000 (26%)]#011Loss: 2.120270
Train Epoch: 2 [25600/50000 (51%)]#011Loss: 2.325724
Train Epoch: 2 [38400/50000 (77%)]#011Loss: 1.927383
Test set: Average loss: 2.6895, Accuracy: 3614/10000 (36%)
Train Epoch: 3 [0/50000 (0%)]#011Loss: 2.076034
Train Epoch: 3 [12800/50000 (26%)]#011Loss: 2.152596
Train Epoch: 3 [25600/50000 (51%)]#011Loss: 1.898087
Train Epoch: 3 [38400/50000 (77%)]#011Loss: 1.880090
Test set: Average loss: 2.2342, Accuracy: 3869/10000 (39%)
Train Epoch: 4 [0/50000 (0%)]#011Loss: 2.021207
Train Epoch: 4 [12800/50000 (26%)]#011Loss: 1.818099
Train Epoch: 4 [25600/50000 (51%)]#011Loss: 1.808489
Train Epoch: 4 [38400/50000 (77%)]#011Loss: 1.765604
Test set: Average loss: 2.0180, Accuracy: 4013/10000 (40%)
Train Epoch: 5 [0/50000 (0%)]#011Loss: 1.766563
Train Epoch: 5 [12800/50000 (26%)]#011Loss: 1.728524
Train Epoch: 5 [25600/50000 (51%)]#011Loss: 1.960890
Train Epoch: 5 [38400/50000 (77%)]#011Loss: 1.610332
Test set: Average loss: 1.8434, Accuracy: 4139/10000 (41%)
Train Epoch: 6 [0/50000 (0%)]#011Loss: 1.662343
Train Epoch: 6 [12800/50000 (26%)]#011Loss: 1.656353
Train Epoch: 6 [25600/50000 (51%)]#011Loss: 1.559448
Train Epoch: 6 [38400/50000 (77%)]#011Loss: 1.765801
Test set: Average loss: 1.8043, Accuracy: 4155/10000 (42%)
Train Epoch: 7 [0/50000 (0%)]#011Loss: 1.802094
Train Epoch: 7 [12800/50000 (26%)]#011Loss: 1.778468
Train Epoch: 7 [25600/50000 (51%)]#011Loss: 1.749434
Train Epoch: 7 [38400/50000 (77%)]#011Loss: 1.561363
Test set: Average loss: 1.6907, Accuracy: 4353/10000 (44%)
Train Epoch: 8 [0/50000 (0%)]#011Loss: 1.498955
Train Epoch: 8 [12800/50000 (26%)]#011Loss: 1.738591
Train Epoch: 8 [25600/50000 (51%)]#011Loss: 1.726524
Train Epoch: 8 [38400/50000 (77%)]#011Loss: 1.469194
Test set: Average loss: 1.6661, Accuracy: 4221/10000 (42%)
Train Epoch: 9 [0/50000 (0%)]#011Loss: 1.747864
Train Epoch: 9 [12800/50000 (26%)]#011Loss: 1.580819
Train Epoch: 9 [25600/50000 (51%)]#011Loss: 1.639743
Train Epoch: 9 [38400/50000 (77%)]#011Loss: 1.632315
Test set: Average loss: 1.8658, Accuracy: 4492/10000 (45%)
Train Epoch: 10 [0/50000 (0%)]#011Loss: 1.582177
Train Epoch: 10 [12800/50000 (26%)]#011Loss: 1.595542
Train Epoch: 10 [25600/50000 (51%)]#011Loss: 1.467178
Train Epoch: 10 [38400/50000 (77%)]#011Loss: 1.633402
Test set: Average loss: 1.7570, Accuracy: 4214/10000 (42%)
```

![image](https://github.com/ChrisLMills/Udacity-Image-Classification-using-AWS-SageMaker/assets/31799634/1b5574fd-538a-4910-bab3-812ad8c529c8)


### Round 2:
Hyperparameters:

`hyperparameters={"epochs": 20, "num_classes": 10, "batch_size":128, "lr":0.05, "trainable_layers":3, "weight_path":previous-model-output}`

Result:

```
Train Epoch: 1 [12800/50000 (26%)]#011Loss: 1.377892
Train Epoch: 1 [25600/50000 (51%)]#011Loss: 1.133370
Train Epoch: 1 [38400/50000 (77%)]#011Loss: 1.001098
Test set: Average loss: 1.1488, Accuracy: 6740/10000 (67%)
Train Epoch: 2 [0/50000 (0%)]#011Loss: 0.955137
Train Epoch: 2 [12800/50000 (26%)]#011Loss: 0.873655
Train Epoch: 2 [25600/50000 (51%)]#011Loss: 0.849849
Train Epoch: 2 [38400/50000 (77%)]#011Loss: 0.708235
Test set: Average loss: 1.3739, Accuracy: 7077/10000 (71%)
Train Epoch: 3 [0/50000 (0%)]#011Loss: 0.696587
Train Epoch: 3 [12800/50000 (26%)]#011Loss: 0.722772
Train Epoch: 3 [25600/50000 (51%)]#011Loss: 0.739430
Train Epoch: 3 [38400/50000 (77%)]#011Loss: 0.783713
Test set: Average loss: 1.2681, Accuracy: 7066/10000 (71%)
Train Epoch: 4 [0/50000 (0%)]#011Loss: 0.742920
Train Epoch: 4 [12800/50000 (26%)]#011Loss: 0.589553
Train Epoch: 4 [25600/50000 (51%)]#011Loss: 0.750581
Train Epoch: 4 [38400/50000 (77%)]#011Loss: 0.667175
Test set: Average loss: 1.3735, Accuracy: 7223/10000 (72%)
Train Epoch: 5 [0/50000 (0%)]#011Loss: 0.567792
Train Epoch: 5 [12800/50000 (26%)]#011Loss: 0.504891
Train Epoch: 5 [25600/50000 (51%)]#011Loss: 0.545125
Train Epoch: 5 [38400/50000 (77%)]#011Loss: 0.453202
Test set: Average loss: 2.0344, Accuracy: 7084/10000 (71%)
Train Epoch: 6 [0/50000 (0%)]#011Loss: 0.643031
Train Epoch: 6 [12800/50000 (26%)]#011Loss: 0.527561
Train Epoch: 6 [25600/50000 (51%)]#011Loss: 0.503914
VanishingGradient: IssuesFound
Overfit: InProgress
Overtraining: InProgress
PoorWeightInitialization: InProgress
Train Epoch: 6 [38400/50000 (77%)]#011Loss: 0.416939
Test set: Average loss: 4.9940, Accuracy: 7089/10000 (71%)
Train Epoch: 7 [0/50000 (0%)]#011Loss: 0.281261
Train Epoch: 7 [12800/50000 (26%)]#011Loss: 0.433155
Train Epoch: 7 [25600/50000 (51%)]#011Loss: 0.603710
Train Epoch: 7 [38400/50000 (77%)]#011Loss: 0.485727
Test set: Average loss: 1.3490, Accuracy: 7001/10000 (70%)
Train Epoch: 8 [0/50000 (0%)]#011Loss: 0.472302
Train Epoch: 8 [12800/50000 (26%)]#011Loss: 0.251676
Train Epoch: 8 [25600/50000 (51%)]#011Loss: 0.363861
Train Epoch: 8 [38400/50000 (77%)]#011Loss: 0.359998
VanishingGradient: IssuesFound
Overfit: InProgress
Overtraining: InProgress
PoorWeightInitialization: Error
Test set: Average loss: 1.5120, Accuracy: 7034/10000 (70%)
Train Epoch: 9 [0/50000 (0%)]#011Loss: 0.228926
VanishingGradient: IssuesFound
Overfit: InProgress
Overtraining: Error
PoorWeightInitialization: Error
Train Epoch: 9 [12800/50000 (26%)]#011Loss: 0.368150
VanishingGradient: IssuesFound
Overfit: IssuesFound
Overtraining: Error
PoorWeightInitialization: Error
Train Epoch: 9 [25600/50000 (51%)]#011Loss: 0.378665
Train Epoch: 9 [38400/50000 (77%)]#011Loss: 0.347986
Test set: Average loss: 1.6986, Accuracy: 7091/10000 (71%)
Train Epoch: 10 [0/50000 (0%)]#011Loss: 0.183391
Train Epoch: 10 [12800/50000 (26%)]#011Loss: 0.250780
Train Epoch: 10 [25600/50000 (51%)]#011Loss: 0.193979
Train Epoch: 10 [38400/50000 (77%)]#011Loss: 0.349006
Test set: Average loss: 1.8006, Accuracy: 7059/10000 (71%)
Train Epoch: 11 [0/50000 (0%)]#011Loss: 0.251324
Train Epoch: 11 [12800/50000 (26%)]#011Loss: 0.415836
Train Epoch: 11 [25600/50000 (51%)]#011Loss: 0.416842
Train Epoch: 11 [38400/50000 (77%)]#011Loss: 0.492213
Test set: Average loss: 1.8181, Accuracy: 6993/10000 (70%)
Train Epoch: 12 [0/50000 (0%)]#011Loss: 0.128385
Train Epoch: 12 [12800/50000 (26%)]#011Loss: 0.212389
Train Epoch: 12 [25600/50000 (51%)]#011Loss: 0.367021
Train Epoch: 12 [38400/50000 (77%)]#011Loss: 0.330452
Test set: Average loss: 3.6141, Accuracy: 7020/10000 (70%)
Train Epoch: 13 [0/50000 (0%)]#011Loss: 0.129347
Train Epoch: 13 [12800/50000 (26%)]#011Loss: 0.274606
Train Epoch: 13 [25600/50000 (51%)]#011Loss: 0.379741
Train Epoch: 13 [38400/50000 (77%)]#011Loss: 0.210673
Test set: Average loss: 4.6254, Accuracy: 7018/10000 (70%)
Train Epoch: 14 [0/50000 (0%)]#011Loss: 0.121250
Train Epoch: 14 [12800/50000 (26%)]#011Loss: 0.180425
Train Epoch: 14 [25600/50000 (51%)]#011Loss: 0.154921
Train Epoch: 14 [38400/50000 (77%)]#011Loss: 0.186153
Test set: Average loss: 2.6880, Accuracy: 6922/10000 (69%)
Train Epoch: 15 [0/50000 (0%)]#011Loss: 0.203109
Train Epoch: 15 [12800/50000 (26%)]#011Loss: 0.223032
Train Epoch: 15 [25600/50000 (51%)]#011Loss: 0.244422
Train Epoch: 15 [38400/50000 (77%)]#011Loss: 0.391352
Test set: Average loss: 2.6451, Accuracy: 7012/10000 (70%)
Train Epoch: 16 [0/50000 (0%)]#011Loss: 0.212482
Train Epoch: 16 [12800/50000 (26%)]#011Loss: 0.280434
Train Epoch: 16 [25600/50000 (51%)]#011Loss: 0.15630120
Train Epoch: 16 [38400/50000 (77%)]#011Loss: 0.223700
Test set: Average loss: 3.4299, Accuracy: 6948/10000 (69%)
Train Epoch: 17 [0/50000 (0%)]#011Loss: 0.182967
Train Epoch: 17 [12800/50000 (26%)]#011Loss: 0.222112
Train Epoch: 17 [25600/50000 (51%)]#011Loss: 0.166987
Train Epoch: 17 [38400/50000 (77%)]#011Loss: 0.173975
Test set: Average loss: 3.5416, Accuracy: 6975/10000 (70%)
Train Epoch: 18 [0/50000 (0%)]#011Loss: 0.140496
Train Epoch: 18 [12800/50000 (26%)]#011Loss: 0.185966
Train Epoch: 18 [25600/50000 (51%)]#011Loss: 0.172055
Train Epoch: 18 [38400/50000 (77%)]#011Loss: 0.108929
Test set: Average loss: 6.8919, Accuracy: 7039/10000 (70%)
Train Epoch: 19 [0/50000 (0%)]#011Loss: 0.133753
Train Epoch: 19 [12800/50000 (26%)]#011Loss: 0.067656
Train Epoch: 19 [25600/50000 (51%)]#011Loss: 0.145023
Train Epoch: 19 [38400/50000 (77%)]#011Loss: 0.239939
Test set: Average loss: 3.1812, Accuracy: 6878/10000 (69%)
Train Epoch: 20 [0/50000 (0%)]#011Loss: 0.299295
Train Epoch: 20 [12800/50000 (26%)]#011Loss: 0.126678
Train Epoch: 20 [25600/50000 (51%)]#011Loss: 0.361809
Train Epoch: 20 [38400/50000 (77%)]#011Loss: 0.184326
Test set: Average loss: 2.8532, Accuracy: 7038/10000 (70%)
```

![image](https://github.com/ChrisLMills/Udacity-Image-Classification-using-AWS-SageMaker/assets/31799634/495a3936-5281-4585-95de-45af2f6f41eb)

As we can see from this second plot, we have overfitting of the model to the training data. 
This is resulting in the low accuracy of 70%.

The plot for this round shows a classic high variance plot. We are overfitting to the training data. We need to use regularization methos such as:

Dropout and weight decay.

### Round 3:
Hyperparameters:

`hyperparameters={"epochs": 10, "num_classes": 10, "batch_size":128, "lr":0.01, "trainable_layers":4, "weight_path":weight_path}`

Dropout of 0.2 added to the fully connected layer in the model.
weight_decay=1e-5 added to the Adam optimizer.

```
Train Epoch: 1 [12800/50000 (26%)]#011Loss: 1.068382
Train Epoch: 1 [25600/50000 (51%)]#011Loss: 0.891764
Train Epoch: 1 [38400/50000 (77%)]#011Loss: 1.979000
Test set: Average loss: 12.2965, Accuracy: 6342/10000 (63%)
Train Epoch: 2 [0/50000 (0%)]#011Loss: 1.603551
Train Epoch: 2 [12800/50000 (26%)]#011Loss: 0.990714
Train Epoch: 2 [25600/50000 (51%)]#011Loss: 1.156443
Train Epoch: 2 [38400/50000 (77%)]#011Loss: 0.733944
Test set: Average loss: 1.1913, Accuracy: 6921/10000 (69%)
Train Epoch: 3 [0/50000 (0%)]#011Loss: 0.960824
Train Epoch: 3 [12800/50000 (26%)]#011Loss: 0.801614
Train Epoch: 3 [25600/50000 (51%)]#011Loss: 0.820299
VanishingGradient: Error
Overfit: InProgress
Overtraining: InProgress
PoorWeightInitialization: InProgress
Train Epoch: 3 [38400/50000 (77%)]#011Loss: 0.600789
Test set: Average loss: 1.0809, Accuracy: 7420/10000 (74%)
Train Epoch: 4 [0/50000 (0%)]#011Loss: 0.769856
Train Epoch: 4 [12800/50000 (26%)]#011Loss: 0.837615
Train Epoch: 4 [25600/50000 (51%)]#011Loss: 1.714671
Train Epoch: 4 [38400/50000 (77%)]#011Loss: 0.795265
Test set: Average loss: 1.6473, Accuracy: 7407/10000 (74%)
Train Epoch: 5 [0/50000 (0%)]#011Loss: 2.266783
Train Epoch: 5 [12800/50000 (26%)]#011Loss: 0.576756
Train Epoch: 5 [25600/50000 (51%)]#011Loss: 0.603325
Train Epoch: 5 [38400/50000 (77%)]#011Loss: 0.663793
VanishingGradient: Error
Overfit: Error
Overtraining: Error
PoorWeightInitialization: InProgress
VanishingGradient: Error
Overfit: Error
Overtraining: Error
PoorWeightInitialization: Error
Test set: Average loss: 1.4372, Accuracy: 7628/10000 (76%)
Train Epoch: 6 [0/50000 (0%)]#011Loss: 0.488719
Train Epoch: 6 [12800/50000 (26%)]#011Loss: 0.515446
Train Epoch: 6 [25600/50000 (51%)]#011Loss: 0.586988
Train Epoch: 6 [38400/50000 (77%)]#011Loss: 0.489115
Test set: Average loss: 1.5229, Accuracy: 7731/10000 (77%)
Train Epoch: 7 [0/50000 (0%)]#011Loss: 0.400257
Train Epoch: 7 [12800/50000 (26%)]#011Loss: 0.461708
Train Epoch: 7 [25600/50000 (51%)]#011Loss: 0.497665
Train Epoch: 7 [38400/50000 (77%)]#011Loss: 0.589339
Test set: Average loss: 1.3409, Accuracy: 7736/10000 (77%)
Train Epoch: 8 [0/50000 (0%)]#011Loss: 0.353389
Train Epoch: 8 [12800/50000 (26%)]#011Loss: 0.503264
Train Epoch: 8 [25600/50000 (51%)]#011Loss: 0.397709
Train Epoch: 8 [38400/50000 (77%)]#011Loss: 0.588397
Test set: Average loss: 1.8671, Accuracy: 7848/10000 (78%)
Train Epoch: 9 [0/50000 (0%)]#011Loss: 0.366575
Train Epoch: 9 [12800/50000 (26%)]#011Loss: 0.522114
Train Epoch: 9 [25600/50000 (51%)]#011Loss: 0.446773
Train Epoch: 9 [38400/50000 (77%)]#011Loss: 0.530452
Test set: Average loss: 1.7385, Accuracy: 7829/10000 (78%)
Train Epoch: 10 [0/50000 (0%)]#011Loss: 0.322812
Train Epoch: 10 [12800/50000 (26%)]#011Loss: 0.266484
Train Epoch: 10 [25600/50000 (51%)]#011Loss: 0.287044
Train Epoch: 10 [38400/50000 (77%)]#011Loss: 0.328061
Test set: Average loss: 1.9316, Accuracy: 7890/10000 (79%)
```

![image](https://github.com/ChrisLMills/Udacity-Image-Classification-using-AWS-SageMaker/assets/31799634/4f9053ed-d0cf-404e-8750-00c01f01fbae)

### Round 4

This round was to rerun to the training but 
Hyperparameters:

`hyperparameters={"epochs": 1, "num_classes": 10, "batch_size":128, "lr":0.001, "trainable_layers":5, "weight_path":weight_path}`

```
Train Epoch: 1 [12800/50000 (26%)]#011Loss: 0.523572
Train Epoch: 1 [25600/50000 (51%)]#011Loss: 0.563649
Train Epoch: 1 [38400/50000 (77%)]#011Loss: 0.465692

2024-04-22 03:46:13 Uploading - Uploading generated training modelTest set: Average loss: 1.5733, Accuracy: 8028/10000 (80%)
```

## Endpoint deployment

Due to going over the allocated credits of $25 during my multiple training runs, I was locked out of the AWS instance for this project. I didn't backup my .ipynb notebook, and this was lost. I was not able to deploy an endpoint for the trained model, nor run a prediction against it. However, these tasks are straightforward and covered in previous projects of mine. 

I could redo the notebook and retrain the model. However, unlike the Rolling Stones, time is not on my side and I must push on with the rest of the course. 

## Prediction

As above.
