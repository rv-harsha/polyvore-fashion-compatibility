import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn = nn.Sequential(

            # CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
            nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    
            
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=.25),          

            nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout2d(p=.25),

            nn.Conv2d(256, 128, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=.25),

            nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(p=.25),

            nn.Conv2d(64, 8, kernel_size=3,stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Dropout2d(p=.25),

        )
         
        self.fc = nn.Sequential(

            nn.Linear(8*15*15, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.25),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.25),

            nn.Linear(512, 153)
            )

        # self.relu = nn.ReLU() 

        # self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(64)                                     
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)         
        
        # self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(128)                                        
        # self.dropout2 = nn.Dropout2d(p=.25)
        # #self.maxpool2 = nn.MaxPool2d(kernel_size=2)     

        # self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1, padding=1)
        # self.batchnorm3 = nn.BatchNorm2d(256)                          
        # self.dropout3 = nn.Dropout2d(p=.25)
        # #self.maxpool3 = nn.MaxPool2d(kernel_size=2)     

        # self.cnn4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3,stride=1, padding=1)
        # self.batchnorm4 = nn.BatchNorm2d(64)                                   
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # self.dropout4 = nn.Dropout2d(p=.25)    

        # self.cnn5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.batchnorm5 = nn.BatchNorm2d(32)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2)     #Size now is 32/2 = 16
        # self.dropout5 = nn.Dropout2d(p=.25)
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        # self.fc1 = nn.Linear(in_features=32*7*7, out_features=1024)   #Flattened image is fed into linear NN and reduced to half size
        # self.droput = nn.Dropout(p=0.5)                 #Dropout used to reduce overfitting
        # # self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # # self.droput = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(in_features=512, out_features=256)
        # self.droput = nn.Dropout(p=0.5)
        # self.fc4 = nn.Linear(in_features=256, out_features=153)
        # self.droput = nn.Dropout(p=0.5)
        # self.fc5 = nn.Linear(in_features=50, out_features=2)    #You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.
       
        
    def forward(self,x):

        out = self.cnn(x)
        # print("\n Shape",out.shape)
        out = out.view(-1,8*15*15)
        # out = output.view(out.size()[0], -1)
        out = self.fc(out)

        # out = self.cnn1(x)
        # out = self.relu(out)
        # out = self.batchnorm1(out)
        # out = self.maxpool1(out)

        # out = self.cnn2(out)
        # out = self.relu(out)
        # out = self.batchnorm2(out)
        # out = self.dropout2(out)
        # #out = self.maxpool2(out)

        # out = self.cnn3(out)
        # out = self.relu(out)
        # out = self.batchnorm3(out)
        # out = self.dropout3(out)
        # #out = self.maxpool3(out)

        # out = self.cnn4(out)
        # out = self.relu(out)
        # out = self.batchnorm4(out)
        # out = self.maxpool4(out)
        # out = self.dropout4(out)

        # out = self.cnn5(out)
        # out = self.relu(out)
        # out = self.batchnorm5(out)
        # out = self.maxpool5(out)
        # out = self.dropout5(out)

        # print(out.shape)
        # #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        # out = out.view(-1,32*7*7)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        # #Then we forward through our fully connected layer 

        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.droput(out)

        # # out = self.fc2(out)
        # # out = self.relu(out)
        # # out = self.droput(out)

        # out = self.fc3(out)
        # out = self.relu(out)
        # out = self.droput(out)

        # out = self.fc4(out)
        return out

model=CNN()