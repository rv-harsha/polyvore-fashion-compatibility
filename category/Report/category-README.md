# Category Classification

## Models
- After testing resnet50, Mobilenet and Squeezenet, I found resnet50 to give higher accuracy as compared to other counterparts
- For this problem, I used a pre-trained resnet50 and added FC layers with dropout in the end to prevent overfitting. 
- I also trained a custom model but couldn't see much accuracy as compared with pre trained with custom model. 

### Compare the two models (pretrained vs. custom). What is the advantage of fine-tuning a pretrained model vs. creating a custom network? 

- We can quickly achieve good training performance with the learned weights from a pre trained model. But with custom model training time will be more and we may not achieve higher accuracy than pre trained ones. 
- Its faster, easy and efficient to train a pre trained model as we can freeze certain layers, train again which may help to increase accuracy sharply. 

### Did you require learning rates for each model?
- For pre-trained model and custom model, learning rates were applied. However, with the pre trained model, layers are usually frozen, and no parameters are learnt during the process. So if we add any extra FC layers in the end, we need to specify optimizer to train only those layers and optimize gradients for those layers. So learning rates should not have a significant impact for all layers of pre trained model when it is frozen. But for custom model, we need to apply learning rates.

## Model metrics

#### For 50 Epochs

- Training Loss: 1.4018, Acc: 59.2319
- Validation Loss: 1.2626 , Acc: 62.9144
- Time taken to complete training: 620.000000m 40.637893s
- Best Validation Accuracy: 63.3040
- Best Test Accuracy: 61.1271