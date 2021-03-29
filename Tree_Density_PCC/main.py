#Dhanuj Gandikota
#dhanujg@umich.edu | (734)-730-3226

#import base libraries needed
import os
import random
from random import randint
import time
import numbers
import numpy as np
#   Since our Image data has been transformed to JSON objects in MATLAB as well we will use EasyDict
from easydict import EasyDict
# I will be using PyTorch to create my deep learning network
import torch
#   use optim to create optimizer for SGD
#   StepLr helps Decays the learning rate of each parameter group by gamma every step_size epochs
from torch import optim
from torch.optim.lr_scheduler import StepLR
#   torch.nn is used in creating the layers
#   use the negative log liklihood loss in our training
from torch import nn
from torch.nn import NLLLoss2d
#   use torchvision to interact with image data
import torchvision.transforms as common_transforms
import torchvision.utils as vutils
#   I will use Pytorch's Dataloader
from torch.utils.data import DataLoader
#   I am using TensorBoard to output the model performance, thought it was cool!
from tensorboardX import SummaryWriter
#A small set of helper functions found in Github of Paper of Inspiration - J. Gao, Q. Wang and X. Li, "PCC Net: Perspective Crowd Counting via Spatial Convolutional Network," doi: 10.1109/TCSVT.2019.2919139.
from PCNN_Func import * 



#This is the file where I wrote all my Torch Neural Network Dimensions and Architecture
from network import *



#---------------------MODELING SETUP------------------------------------------------------

#Set up our Model TreeVision Parameters as 'dict' object, may be exported later into JSON or seperate files
TreeVision = EasyDict()
#   Define Dataset Parameters for TreeVuion
TreeVision.DATASET = EasyDict()
TreeVision.DATASET.IMAGE_DIM = (576, 768)
TreeVision.DATASET.WD = './data/trees'
TreeVision.DATASET.MEAN_STD = ([0.444637000561], [0.226200059056])#Obtained in MATLAB
TreeVision.DATASET.SCALE_FACTOR = 1
#   Define Network Training Parameters for TreeVision
TreeVision.TRAINING = EasyDict()
TreeVision.TRAINING.ORIG_IMG_DIM = (512,680)
TreeVision.TRAINING.RESUME = './models'
TreeVision.TRAINING.TRAIN_SUBSET_SIZE = 6 #image batch size
TreeVision.TRAINING.VAL_SUBSET_SIZE = 1 #image batch size
TreeVision.TRAINING.WEIGHT_BCELOSS = 1e-4
TreeVision.TRAINING.MODEL_SEED = 500
TreeVision.TRAINING.LR = 1e-4
TreeVision.TRAINING.LR_DECAY = 1
TreeVision.TRAINING.LR_DECAY_EPOCH_LEVEL = 1 
TreeVision.TRAINING.LR_SEGMENTATION = 1e-2
TreeVision.TRAINING.WEIGHTS_SEGMENTATION = 1e-4
TreeVision.TRAINING.ROIBOX_NUM = 20 
TreeVision.TRAINING.ROIBOX_RATIO = 0.25
TreeVision.TRAINING.NUM_EPOCH = 30
TreeVision.TRAINING.INSTANCE_NAME = 'TreeVision_CNN' + ' 4/24/2020'
TreeVision.TRAINING.INSTANCE_WD = './experiments'
TreeVision.TRAINING.GPU_ID = [1] #GPU Allocation

#Define Model Summary Output location/txt
TensorSummary= SummaryWriter(TreeVision.TRAINING.INSTANCE_WD + '/' + TreeVision.TRAINING.INSTANCE_NAME)
Output = (TreeVision.TRAINING.INSTANCE_WD + '/' + TreeVision.TRAINING.INSTANCE_NAME + '/' + TreeVision.TRAINING.INSTANCE_NAME + '.txt')
#os.makedirs(TreeVision.TRAINING.INSTANCE_WD)



#Set Model Seeds for Consistant results
np.random.seed(TreeVision.TRAINING.MODEL_SEED)
torch.manual_seed(TreeVision.TRAINING.MODEL_SEED)
torch.cuda.manual_seed(TreeVision.TRAINING.MODEL_SEED)


####------------------LOAD DATA---------------------------------------------------------------

#Create a function to return back a tensor array of [Image, DensityVal, TreeCount, RoiBox, RoiBox Labels, Segmentations]
# for each set of files dictated by extensionpath (train or validation data)
def create_data_set(extensionpath = 'train_data'):
    #####since we have segmentations, we should chain together some transformations

    #   convert images in PIL to tensor objects
    tensor_image_conversion = common_transforms.ToTensor()
    #   normalize images
    normalize_transforms = Compose_img([tensor_image_conversion, common_transforms.Normalize(*TreeVision.DATASET.MEAN_STD)])

    #   we will crop the data into our traning input sizes and perform a horizontal flip (simulate multiple viewing angles)
    training_transforms = Compose_img([RandomCrop_img(TreeVision.TRAINING.ORIG_IMG_DIM), RandomHorizontallyFlip_img()])


    ####making sure the densities are evenly weighted across classes
    files = [filename for filename in os.listdir(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'img') if os.path.isfile(os.path.join(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'img',filename))]
    data_size = len(files)

    #   we need to obtain the ground truth density count statistics for trees from our MATLAB Data Files, we will set classes = 10
    dens_min = sys.maxsize
    dens_max = 0
    i = 0
    tree_counts = np.zeros(data.size)
    class_dens = np.zeros(10)
    for datafile in files:
        #read in the density MATLAB object 
        tree_dens = (pd.read_csv(os.path.join(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'den',os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()).astype(np.float32, copy=False)
        #sum up the count values in the dens file
        temp_counts = np.sum(tree_dens)
        #fill in counts matrix
        tree_counts [i] = temp_count/(tree_dens[0]*tree_dens[1])
        i= i+1
        #set the max,min den values
        dens_max = max(temp_count/(tree_dens[0]*tree_dens[1]), dens_max)
        dens_min  = max(temp_count/(tree_dens[0]*tree_dens[1]), dens_min)

    #   update the tree classes normalized by the max,min dens, we will set classes = 10
    tree_classes = np.round(tree_counts/(dens_max- dens.min)/float(10))
    for j in tree_classes:
        #increment class dens to get total class dens overall
        class_dens[(int(min(9, j)))] = class_dens[(int(min(9, j)))] + 1

    #calculate the weights for each density class given their density counts
    tree_class_weights = ((1 - class_dens) / class_dens) / class_dens


    ###We will now grab and load the data

    #create torch object to hold full data
    full_data = torch.empty(6)

    for filepath in files:
        #read in image
        image = Image.open(os.path.join(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'img',fname))
        if Image.mode == 'RGB':
            image = Image.convert('L')

        #read in segmentation
        segmentation = Image.open(os.path.join(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'seg',fname.split('.')[0]+'.png'))

        #read in density
        density = (pd.read_csv(os.path.join(TreeVision.DATASET.WD + '/' + extensionpath + '/' + 'den',os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()).astype(np.float32, copy=False)
        density = Image.fromarray(density)

        #crop our data using training_transformation from earlier
        image, density, segmentation = training_tranforms(image, density, segmentation)
        # normalize the image transformations
        image = normalize_transforms(image)

        #convert segmentations to torch object
        seg = torch.from_numpy(np.array(segmentation).astype(np.uint8)).long()

        #calculate tree_count for image
        count = density.sum()

        #create ROI data
        height = TRAINING.ORIG_IMG_DIMM[0]
        width = TRAINING.ORIG_IMG_DIM[1]
        ROI_Labels = torch.zeros(TRAINING.ROIBOX_NUM,10)
        ROI_Dim = torch.zeros((TRAINING.ROIBOX_NUM,5))

        #create randomed roi boxes of dimensionality greater than 1/4 original
        for box in range(0, TRAINING.ROIBOX_NUM):
            box_height, hox_width = 0
            l1, l2, h1, h2 = 0

            while (box_width <= (TreeVision.TRAINING.ROIBOX_RATIO)*width) or (box_height <= (TreeVision.TRAINING.ROIBOX_RATIO)*height):
                #pick random axis for roi box
                l1 = random.randint(0,width-2)
                l2 = random.randint(l1,width-2)
                h1 = random.randint(0,height-2)
                h2 = random.randint(h1,height-2)
                box_height = h2 - h1
                box_width = l2 - l1

            #set ROI dimensions
            ROI_Dim[box][0] = int(0)
            ROI_Dim[box][1] = l1
            ROI_Dim[box][2] = h1
            ROI_Dim[box][3] = l2
            ROI_Dim[box][4] = h2

            #Assign ROI into their own classes

            ROI_ClassLabel = int(min(np.round((density[h1:h2-1, l1:l2-1].sum) / (box_width *  box_height * (dens_max- dens.min)/float(10))), 9))
            ROI_Labels[box][ROI_ClassLabel] = 1
        #convert to long
        ROI_Dim = ROI_Dim.long()
        
        #append all the data to full_data
        full_data.cat(Tensor(image, density, count, ROI_Dim, ROI_Labels, segmentation), dim = 1)

    return full_data, tree_class_weights


#Obtain the training data and use DataLoader to create our Training and Validation Sets
Training_Data, train_weights = create_data_set(extensionpath = 'train_data')
Validation_Data, valid_weights = create_data_set(extensionpath = 'test_data')

Training_loader = DataLoader(Training_Data, batch_size = TreeVision.TRAINING.TRAIN_SUBSET_SIZE, num_workers = 8, shuffle=True, drop_last=True)
Validation_loader = DataLoader(Validation_Data, batch_size = TreeVision.TRAINING.VAL_SUBSET_SIZE, num_workers = 8, shuffle=True, drop_last=True)

#  transform to return images to normal for after
denormalize_transforms = Compose_img([Denormalize_label(*TreeVision.DATASET.MEAN_STD), common_transforms.ToPILImage()]), 





####---------------------TRAIN AND BUILD MODEL FUNC-------------------------------


def TrainMeToCountTrees(Training_loader, NN_object, Optimizer, Epoch_Num, iter_tensorboard): 

    i = 0

    for values in Training_loader:

       

        #read in our train_data values
        image, densityVal, treecount, roiBox, roiLabel, segmentation = values

        #lets convert our main image data to cuda
        image = Variable(image).cuda()
        densityVal = Variable(densityVal).cuda()
        segmentation = Variable(segmentation).cuda()

        #set the base [0] dimension of our ROI to contain the training image index
        for index_train in range(0, TreeVision.TRAINING.TRAIN_SUBSET_SIZE):
            roiBox[index_train, : , 0] = index_image
        
        #we will create views of our marked ROI in training such that access/edits are still easy
        sight = TRAINING.TRAIN_SUBSET_SIZE * TreeVision.TRAINING.ROIBOX_NUM
        roiBox = roiBox.view(sight, 5)
        roiBox = Variable(roiBox).cuda().float()
        roiLabel = roiLabel.view(sight,10)
        roiLabel = Variable(roiLabel).cuda()

        #set gradient to zero before backpropagation
        optimizer.zero_grad()

        #Run our Neural Net!!!!!
        DensityMap_pred, Class_pred, Segmentation_Pred = NN_object(image, densityVal, roiBox, roiLabel, segmentation)

        #set loss = net.loss and forward setp with optimizer
        Training_Loss = NN_object.loss
        Training_Loss.backward()

        optimizer.step()

        #write out our loss results to the tensorboardx writer every 5 iterations
        if (i + 1)%5 == 0:
            iter_tensorboard+=1

            #return back our three chosen loss metrics from the paper.
            MSE_loss, CrossEntropy_Loss, Segmentation_Loss = NN_object.Loss_scores()

            #print out our training scores
            print('Epoch #: ' + str(epoch+1) + ' | Iter: ' + str(i+1))
            print('MSE Loss: ' + str(MSE_loss.item()) + ' | CrossEntropy: ' + str(CrossEntropy_Loss.item) + ' | Segm Loss: ' + str(Segmentation_Loss))

            #commit our values to the tensorboardx writer
            writer.add_scalar('Overall_loss', Training_Loss.item(), i_tb)
            writer.add_scalar('MSE_loss', MSE_loss.item(), i_tb)
            writer.add_scalar('CrossEntropy_Loss', CrossEntropy_Loss.item(), i_tb)
            writer.add_scalar('Segmentation_Loss', Segmentation_Loss.item(), i_tb)
        
        i+=1
    
    #save our torch trained network model!
    torch.save(NN_object.state_dict(), (TreeVision.TRAINING.INSTANCE_WD + '/' + 'treedensity_run12/'  + '/'+ 'Epoch_'+ str(epoch+1) + '.pth'))

    return iter_tensorboard, ((TreeVision.TRAINING.INSTANCE_WD + '/' + 'treedensity_run12/'  + '/'+ 'Epoch_'+ str(epoch+1) + '.pth'))



#####--------------------VALIDATE MODEL FUNC-------------------------------

#we will use this global dictionary to record the best scores of our top validated model
Best_Model_Info = {'Model': '','Best_Epoch': -1, 'Lowest_MAE': 0, 'Lowest_MSE': 0,'Lowest_Loss': 0, 'Lowest_Seg_Loss' : 0, 'Lowest_CrossEntropy_Loss' : 0 }


def ValMyCountingTrees(Validation_Loader, Model, Epoch_Num):

    #create neural net object for the validation and load in our training model
    NNet = TreeCounter(class_weights = train_weights)
    NNet.load_state_dict(torch.load(Model))

    NNet.cuda()
    NNet.eval()

    #define empty arrays to hold our Loss Scores for the validation set
    Validation_Loss, MSE_loss, CrossEntropy_Loss, Segmentation_Loss = []

    #We will be using Mean Squared Error and the Mean Average Error to judge our Validation Performance
    MSE, MAE = 0.0

    i = 0
    for values in Validation_loader:

        #read in our validation_data values
        image, densityVal, treecount, roiBox, roiLabel, segmentation = values

        #lets convert our main image data to cuda
        image = Variable(image).cuda()
        densityVal = Variable(densityVal).cuda()
        segmentation = Variable(segmentation).cuda()
        roiBox = Variable(roiBox[0]).cuda().float()
        roiLabel = Variable(roiLabel[0]).cuda()

        #Run our Neural Net!!!!!
        DensityMap_pred, Class_pred, Segmentation_Pred = NNet(image, densityVal, roiBox, roiLabel, segmentation)

        #return back our three chosen loss metrics from the paper.
        ValidLoss = NNet.loss
        MSEloss, CrossEntropyLoss, SegLoss = NNet.Loss_scores()

        #Push these losses into our main arrays
        Validation_Loss.append(ValidLoss.item())
        MSE_loss.append(MSEloss.item())
        CrossEntropy_Loss.append(CrossEntropyLoss.item())
        Segmentation_Loss.append(SegLoss.item())

        #Convert our objects to Numpy for easy calculations later, in CPU
        segmentation = segmentation.data.cpu().numpy()
        Segmentation_Pred = Segmentation_Pred.cpu().max(1)[1].squeeze_(1).data.numpy()
        image = image.cpu().data

        #Compute our Total Counts and Predicted Count
        Actual_Count = np.sum(densityVal.data.cpu().numpy())
        Predicted_Count = np.sum(DensityMap_pred.data.cpu().numpy())

        #Add the difference between the scores to our MSE and MAE
        MSE = MSE + (Actual_Count - Predicted_Count)*(Actual_Count - Predicted_Count)
        MAE = MAE + abs(Actual_Count - Predicted_Count)

        #Here I implement a cool algorithm written to display the output validation image
        # and compare it to the original image in the TensorboardX Summary Writer using Torchvision's vutils

        image_grid = []

        if (i==0):
            for index, data in enumerate(zip(image, DensityMap_pred.data.cpu().numpy(), densityVal.data.cpu().numpy(), Segmentation_Pred, segmentation)):
                #end loop after displaying 20 images
                if index > 20:
                    break
                
                
                #Obtain the original image from the data
                Original_Image = pil_to_tensor(denormalize_transforms(data[0]/255.))

                #create matrix of the original density of image 
                #All are repeated three times to spread across each image generated in the grid (image, density map, segmentation)
                Original_Image_Density = torch.from_numpy(data[2]/(data[2].max()+1e-10)).repeat(3,1,1)
                #do the same except now for the Predicted Density of the image, and the segmentation/predicted segmentations
                Predicted_Image_Density = torch.from_numpy(data[1]/(data[1].max()+1e-10)).repeat(3,1,1)
                #Also spread the segmentations and predicted segmentations
                Predicted_Image_Segmentation = torch.from_numpy(data[3]).repeat(3,1,1)
                Original_Image_Segmentation = torch.from_numpy(data[4]).repeat(3,1,1)

                #Push these values into the image grid
                image_grid.extend([Original_Image,Original_Image_Density, Original_Image_Segmentation, Predicted_Image_Segmentation])

            #Now we will make a grid out of these image data
            image_grid = vutils.make_grid(torch.stack(x, 0), nrow=5, padding=5)
            #covert to numpy image
            image_grid = (image_grid.numpy()*255).astype(np.uint8)
            #Send this grid to the summarywriter
            writer.add_image('Validation_Epoch_' + str(epoch+1), image_grid)
        i+=1
    

    Validation_num = Validation_Data.get_num_samples()



    #Now we can calculate our final Validation Scores and Loss Values

    Validation_Loss = np.mean(Validation_Loss)
    Validation_MSE_Loss = np.mean(MSE_Loss)
    Validation_CrossEntropy_Loss = np.mean(CrossEntropy_Loss)
    Validation_Segmentation_Loss = np.mean(Segmentation_Loss)

    #Average out MAE and MSE
    MAE = MAE/ Validation_num
    MSE = np.sqrt(MSE/Validation_num)

    #Check if our Scores are the best scores so far!
    #Use MAE as deciding metric for best model
    if (Best_Model_Info['Lowest_MAE'] > MAE):
        #Update our Best Model Variable
        Best_Model_Info['Model'] = Model
        Best_Model_Info['Best_Epoch'] = Epoch_Num + 1
        Best_Model_Info['Lowest_MAE'] = MAE
        Best_Model_Info['Lowest_MSE'] = MSE
        Best_Model_Info['Best_Loss'] = Validation_Loss
        Best_Model_Info['Best_Seg_Loss'] = Validation_Segmentation_Loss
        Best_Model_Info['Best_CrossEntropy_Loss'] = Validation_CrossEntropy_Loss

    
    #Print out our Epoch Results!
    print ( '_'*20)
    print ('Model Validation')
    print('\n')
    print ('Current_Epoch %d | MAE: %1f | MSE: %1f | Validation Loss: %3f') % (Epoch_Num, MAE, MSE, Validation_Loss)
    print ('                 | MSE Loss: %3f | CrossEntropy Loss: %3f |  Segmentation Loss: %3f' ) % (Validation_MSE_Loss, Validation_CrossEntropy_Loss, Validation_Segmentation_Loss)
    print('\n')
    print ('Best_Epoch %d | MAE: %1f | MSE: %1f | Validation Loss: %3f') % (Best_Model_Info['Best_Epoch'], Best_Model_Info['Lowest_MAE'], Best_Model_Info['Lowest_MSE'], Best_Model_Info['Best_Loss'])
    print ('                  CrossEntropy Loss: %3f |  Segmentation Loss: %3f' ) % (Best_Model_Info['Best_CrossEntropy_Loss'], Best_Model_Info['Best_Seg_Loss'])


    #finally we will send our results to the summarywriter
    writer.add_scalar('Validation_Loss', Validation_Loss, Epoch_Num + 1)
    writer.add_scalar('MAE', MAE, Epoch_Num + 1)
    writer.add_scalar('MSE', MSE, Epoch_Num + 1)
    writer.add_scalar('MSE_Loss', Validation_MSE_Loss, Epoch_Num + 1)
    writer.add_scalar('CrossEntropy_Loss', Validation_CrossEntropy_Loss, Epoch_Num + 1)
    writer.add_scalar('Segmentation_Loss', Validation_Segmentation_Loss, Epoch_Num + 1)
    
    return


#####----------------MAIN, WE CAN RUN THE ENTIRE PIPELINE------------------------------------------------------------------


def main():

    #run with cuda
    use_cuda = True
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True


    #Create our NN object
    NNet = TreeCounter(class_weights = train_weights)
    NNet.train()


    #We will use the Adam Optimization algotihm with follows a L2 penality that is changed in decoupled weight decay regulaization
    Optimizer = optim.Adam([
                            {'params': [param for name, param in Net.named_parameters() if 'seg' in name], 'lr': cfg.TRAIN.SEG_LR},
                            {'params': [param for name, param in Net.named_parameters() if 'seg' not in name], 'lr': cfg.TRAIN.LR}
                          ])


    
    iter_tensorboard = 0


    #We can now train and validate our Neural Network, yay!
    for Epoch in range(0, TreeVision.TRAINING.NUM_EPOCH):

        #train the Neural Net and obtain the tensorboard iter
        iter_tensorboard , Model = TrainMeToCountTrees(Training_loader, NNet, Optimizer, Epoch, iter_tensorboard)

        #validate the model performance, check the output for all the values and the summaryWriter!
        ValMyCountingTrees(Validation_Loader, Model, Epoch)



if __name__ == '__main__':
    main()


