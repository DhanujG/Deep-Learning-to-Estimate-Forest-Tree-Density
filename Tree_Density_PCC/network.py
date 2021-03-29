#Dhanuj Gandikota
#dhanujg@umich.edu | (734)-730-3226

# I will be using PyTorch to create my deep learning network
import torch
import torch.nn as nn
import torch.nn.functional as F
WEIGHTS_SEGMENTATION = 1e-4
WEIGHTS_BCELOSS = 1e-4


#--------------------------------------------------------TREE DENSITY CONVOLUTIONAL NEURAL NETWORK CLASS---------------------------------------
class TreeCounter(nn.Module):
    
    def __init__(self, class_weights=None):
        super(TreeCounter, self).__init__()        


        self.TreeNet = NN_Architecture()        
        self.TreeNet=self.TreeNet.cuda()



        if class_weights is not None:
            class_weights = torch.Tensor(class_weights)
            class_weights = class_weights.cuda()
        




        self.MSE_Loss_FN = nn.MSELoss().cuda()
        self.BCE_Loss_FN = nn.BCELoss(weight=class_weights).cuda() # binary cross entropy Loss
        self.CEL_Loss_FN = CrossEntropyLossMod().cuda()

        
        
    @property
    def Loss(self):
        return self.MSE_Loss + WEIGHTS_BCELOSS *self.CrossEntropy_Loss + WEIGHTS_SEGMENTATION *self.Segmentation_Loss




    def Loss_scores(self):
        return self.MSE_Loss, self.CrossEntropy_Loss, self.Segmentation_Loss
    



    def forward(self, image, densityVal, roiBox, roiLabel, segmentation): 

        DensityMap, Class_Density_Score,Segmentation_Pred = self.TreeNet(image,roiBox)

        Class_Density_Probability = F.softmax(Class_Density_Score,dim=1)        
        self.MSE_Loss, self.CrossEntropy_Loss, self.Segmentation_Loss = self.build_Loss(DensityMap, Class_Density_Probability, Segmentation_Pred, densityVal, roiLabel, segmentation)
    
        return DensityMap, Class_Density_Score, Segmentation_Pred
    


    def build_Loss(self, DensityMap, Class_Density_Score, Segmentation_Pred, orig_data, orig_Class_Label,segmentation):
        MSE_Loss = self.MSE_Loss_FN(DensityMap.squeeze(), orig_data.squeeze())  
        
        CrossEntropy_Loss = self.BCE_Loss_FN(Class_Density_Score, orig_Class_Label)
        Segmentation_Loss = self.CEL_Loss_FN(Segmentation_Pred, segmentation)  
        return MSE_Loss, CrossEntropy_Loss,Segmentation_Loss



    def test_forward(self, image, roiBox):                               
        DensityMap, Class_Density_Score,Segmentation_Pred = self.TreeNet(image,roiBox)            
            
        return DensityMap, Class_Density_Score, Segmentation_Pred


#--------------------------------------------------------CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE-----------------------------------------------
class NN_Architecture(nn.Module):

    def __init__(self, bn=False, Class_Number=10):
        super(NN_Architecture, self).__init__()
        
        self.Class_Number = Class_Number

        self.Initial_Layers = nn.Sequential(
                                        
                                        #Base Layer 1
                                        nn.Conv2d(1, 16, 9, stride = 1, padding= int((9 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(16, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True), 
                                         
                                         
                                                                                                    
                                        #Base Layer 2
                                        nn.Conv2d(16, 32, 7, stride = 1, padding= int((7 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True), 
                                         
                                         
                                         
                                        )
        
        self.High_Level_1_4 = nn.Sequential(
                                    
                                    #High Level Layer 1
                                    nn.Conv2d(32, 16, 9, stride = 1, padding= int((9 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(16, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  


                                    nn.MaxPool2d(2),

                                    
                                    #High Level Layer 2
                                    nn.Conv2d(16, 32, 7, stride = 1, padding= int((7 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                     
                                    nn.MaxPool2d(2),

                                    #High Level Layer 3
                                    nn.Conv2d(32, 32, 7, stride = 1, padding= int((7 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                     
                                    #High Level Layer 4
                                    nn.Conv2d(16, 32, 7, stride = 1, padding= int((7 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                     
                                    )
                
        self.Roi_Pooling = RoIPool([16, 16], 2.5)


        self.High_Level_PostRoi = nn.Sequential(
                                                #Classification Layer Following ROI Pooling
                                                nn.Conv2d(32, 16, 1, stride = 1, padding= int((1 - 1) // 2), dilation=1),
                                                nn.BatchNorm2d(16, eps=0.001, momentum=0, affine=True),
                                                nn.PReLU(),
                                                #nn.ReLU(inplace=True),  
                                    
                                     
                                                )
        
        self.Classification_Network = nn.Sequential(
                                    
                                    # Fully Connected Layer 1
                                    nn.Linear(16*16*16, 512),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True)


                                    # Fully Connected Layer 2
                                    nn.Linear(512, 256),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True)


                                    
                                    # Fully Connected Layer 3
                                    nn.Linear(256, self.Class_Number),
                                    nn.PReLU()
                                    #nn.ReLU(inplace=True)


                                    )
             





        # generate dense map
        self.DensityMap_1_4 = nn.Sequential(
                                    #DesnityMap Layer 1
                                    nn.Conv2d(32, 32, 7, stride = 1, padding= int((7 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                    
                                     nn.MaxPool2d(2),

                                     #DensityMap Layer 2
                                    nn.Conv2d(32, 64, 5, stride = 1, padding= int((5 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(64, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                    
                                     nn.MaxPool2d(2),

                                     #DensityMap Layer 3
                                    nn.Conv2d(64, 32, 5, stride = 1, padding= int((5 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU(),
                                    #nn.ReLU(inplace=True),  
                                    
                                    
                                     #DensityMap Layer 4
                                    nn.Conv2d(32, 32, 5, stride = 1, padding= int((5 - 1) // 2), dilation=1),
                                    nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                    nn.PReLU()
                                    #nn.ReLU(inplace=True),  
                                    
                                    )
        
        self.DensityMap_DULR_Module = nn.Sequential(
                                        #Special DULR Perspective Module, see below in code for more details
                                        convDU(Channel_Input_Output=32,kernel_size=(1,9)),
                                        convLR(Channel_Input_Output=32,kernel_size=(9,1)))


        self.DensityMap_5_8 = nn.Sequential(
                                        #Density Map Layer 5
                                        nn.Conv2d(64, 64, 3, stride = 1, padding= int((3 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(64, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True),  
                                    
                                    
                                        #Layer 6
                                        nn.Conv2d(64, 32, 3, stride = 1, padding= int((3 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True),  
                                    
                                        #Layer 7                                  
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),

                                        #Layer 8
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU())

        # generrate seg map
        self.SegmentationMap_1_4 = nn.Sequential(
                                        # Foreward and Backward Segmentation Map Layer 1
                                        nn.Conv2d(32, 32, 1, stride = 1, padding= int((1 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True),  
                                    
                                    
        	                            #Layer 2
                                        nn.Conv2d(32, 64, 3, stride = 1, padding= int((3 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(64, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True),  
                                    
                                    
                                        #Layer 3
                                        nn.Conv2d(64, 32, 3, stride = 1, padding= int((3 - 1) // 2), dilation=1),
                                        nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True),
                                        nn.PReLU(),
                                        #nn.ReLU(inplace=True),  
                                    
                                    


                                        #Layer 4
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU())


        self.SegmentationMap_output = nn.Sequential(
                                                #Generate Segmentation Map
                                                nn.Conv2d(8, 2, 1, stride = 1, padding= int((1 - 1) // 2), dilation=1),
                                                nn.BatchNorm2d(2, eps=0.001, momentum=0, affine=True),
                                                nn.PReLU(),
                                                #nn.ReLU(inplace=True),  
                                    
                                     
                                                )

        self.Segmentation_Density_Transition = nn.Sequential(
                                                #GTransition Layer feeding into DensityMap
                                                nn.Conv2d(8, 8, 1, stride = 1, padding= int((1 - 1) // 2), dilation=1),
                                                nn.BatchNorm2d(8, eps=0.001, momentum=0, affine=True),
                                                nn.PReLU(),
                                                #nn.ReLU(inplace=True),  
                                    
                                     
                                                )

        self.DensityMap_output = nn.Sequential(
                                                #Generate Density Map
                                                nn.Conv2d(8, 1, 1, stride = 1, padding= int((1 - 1) // 2), dilation=1),
                                                nn.BatchNorm2d(1, eps=0.001, momentum=0, affine=True),
                                                nn.PReLU(),
                                                #nn.ReLU(inplace=True),  
                                    
                                     
                                                )

        

        #Normalize and Set the Weight for the entire Network Accordingly
        Normalize_Weights(self.Initial_Layers, self.High_Level_1_4, self.High_Level_PostRoi, self.Classification_Network, self.DensityMap_1_4, self.DensityMap_DULR_Module, self.DensityMap_5_8, self.Segmentation_Density_Transition, self.DensityMap_output)
        Set_Weights(self.SegmentationMap_1_4,  self.SegmentationMap_output)
        

        
    def forward(self, image_data, roiBox):
        trees_init = self.Initial_Layers(image_data)

        trees_HighLevel = self.High_Level_1_4(trees_init)

        trees_Classif = self.Roi_Pooling(trees_HighLevel, roiBox)

        trees_Classif = self.High_Level_PostRoi(trees_Classif)

        trees_Classif = trees_Classif.view(trees_Classif.size(0), -1) 

        trees_Classif = self.Classification_Network(trees_Classif)

        trees_mapping = self.DensityMap_1_4(trees_init)

        trees_mapping = self.DensityMap_DULR_Module(trees_mapping) 

        trees_DensMap = torch.cat((trees_HighLevel,trees_mapping),1)

        trees_DensMap = self.DensityMap_5_8(trees_DensMap)

        trees_FBS = self.SegmentationMap_1_4(trees_mapping)

        trees_SegMap = self.SegmentationMap_output(trees_FBS)

        trees_FBS = self.Segmentation_Density_Transition(trees_FBS)

        trees_DensMap = self.DensityMap_output(trees_DensMap)



        return trees_DensMap, trees_Classif, trees_SegMap


#-------------------------------------------------------- ADDITIONAL FUNCTIONS--------------------------------------------------------------

class CrossEntropyLossMod(nn.Module):
    def __init__(self):

        super(CrossEntropyLossMod, self).__init__()

        self.NLL_Loss = nn.NLLLoss(weight=None, reduction='mean')

    def forward(self, inputs, targets):

        return self.NLL_Loss(F.log_softmax(inputs,dim=1), targets)


def Normalize_Weights(*Layer_Sets):

    for Component in Layer_Sets:
        deviation=0.01

        if isinstance(Component, list):

            for i in Component:
            
                Normalize_Weights(i, deviation)
        else:

            for i in Component.modules():       

                if isinstance(i, nn.Conv2d):        
                    i.weight.data.normal_(0.0, deviation)

                    if i.bias is not None:
                        i.bias.data.fill_(0.0)

                elif isinstance(i, nn.Linear):
                    i.weight.data.normal_(0.0, deviation)


def Set_Weights(*Layer_Sets):
    for Component in Layer_Sets:

        for i in Component.modules():

            if isinstance(i, nn.Conv2d):
                i.weight.data.normal_(0, math.sqrt(2. / (i.kernel_size[0] * i.kernel_size[1] * i.out_channels)))

                if i.bias is not None:
                    i.bias.data.zero_()

            elif isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()

            elif isinstance(i, nn.Linear):
                i.weight.data.normal_(0, math.sqrt(2. / i.weight.size(1)))
                i.bias.data.zero_()


#--------------------------------------------------------------------PERSPECTIVE 'DULR' MODULE ------------------------------------------------------------
#These layers are very special because their design come straight from the paper -> J. Gao, Q. Wang and X. Li, "PCC Net: Perspective Crowd Counting via Spatial Convolutional Network," doi: 10.1109/TCSVT.2019.2919139
#In this paper, this DULR Layer of the Neural Net was the novel 'Perspective Module' which was consisted of the two classes below making up a DULR (Down, Up, Left, Right) module.
# The code implementation of these Kernels remain the same in structure to the original paper such as to maintain the accuracy of these two layers


class convDU(nn.Module):

    def __init__(self,
        Channel_Input_Output=2048,
        kernel_size=(9,1)

        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(Channel_Input_Output, Channel_Input_Output, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.PReLU()
            )

    def forward(self, Feature):
        Num_Images, Num_Channels, Height, Width = Feature.size()

        Feature_Collection = []
        for i in range(Height):
            i_fea = Feature.select(2, i).reshape(Num_Images,Num_Channels,1,Width)
            if i == 0:
                Feature_Collection.append(i_fea)
                continue
            Feature_Collection.append(self.conv(Feature_Collection[i-1])+i_fea)
            
            


        for i in range(Height):
            loc = Height-i-1
            if loc == Height-1:
                continue
            Feature_Collection[loc] = self.conv(Feature_Collection[loc+1])+Feature_Collection[loc]
        
        Feature = torch.cat(Feature_Collection, 2)
        return Feature

class convLR(nn.Module):

    def __init__(self,
        Channel_Input_Output=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(Channel_Input_Output, Channel_Input_Output, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.PReLU()
            )

    def forward(self, Feature):
        Num_Images, Num_Channels, Height, Width = Feature.size()

        Feature_Collection = []
        for i in range(Width):
            i_fea = Feature.select(3, i).reshape(Num_Images,Num_Channels,Height,1)
            if i == 0:
                Feature_Collection.append(i_fea)
                continue
            Feature_Collection.append(self.conv(Feature_Collection[i-1])+i_fea)

        for i in range(Width):
            loc = Width-i-1
            if loc == Width-1:
                continue
            Feature_Collection[loc] = self.conv(Feature_Collection[loc+1])+Feature_Collection[loc]


        Feature = torch.cat(Feature_Collection, 3)
        return Feature


