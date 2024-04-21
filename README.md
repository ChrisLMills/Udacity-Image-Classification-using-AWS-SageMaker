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

In coming to understand how PyTorch manages back propogation, I came across this great discussion:

https://discuss.pytorch.org/t/how-does-loss-function-affect-model-during-training/187244

I was initially confused by the implied links from the model parameters being passed to the optimizer, the output from the model being passed to loss and then the loss and optimizer being called as separate steps:

`optimizer = optim.Adam(model.parameters(), lr=args.lr)`

`output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()`


## Testing def
## Hyperparamter Search
## Debugger and Profiler setup
## Model training
## Endpoint deployment
## Prediction
