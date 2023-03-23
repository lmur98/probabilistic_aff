import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.empty_cache()
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from common.warmup_lr import GradualWarmupScheduler
#from data.dataset_UMD import UMD_dataset
from data.dataset import IIAff_dataset, my_collate
from models.baseline import baseline
from models.eval_one_epoch import evaluate_one_epoch
from models.train_one_epoch import train_one_epoch
from common.metrics import init_stats, init_loss_stats

#A. Training on the GPU and tensorboard
gpu_device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
aff_labels = {0: 'background', 1: 'contain', 2: 'cut', 3: 'display', 4: 'engine', 5: 'grasp', 6: 'hit', 7: 'pound', 8: 'support', 9:'w-grasp'}
#aff_labels = {0: 'background', 1: 'grasp', 2: 'cut', 3: 'scoop', 4: 'contain', 5: 'pound', 6: 'support', 7: 'wrap-grasp'}    


#D. Define model, move to device and define optimizer, learning rate, etc..
n_epochs = 31
resume  = False
Ens_models = 12

for k in range(Ens_models):
    print('MODEL', k, 'of', Ens_models)
    writer = SummaryWriter(log_dir=os.path.join('/home/lmur/documents/metrics_Deep_Ensembles/Rx101', 'Ensembles_Rx101_' + str(k)))

    #B. Read the dataset
    general_dir = '/home/lmur/documents/iit_aff_DATASET' #'/home/lmur/Documents/iit_aff/UMD_DATASET/part-affordance-dataset' #
    train_data = IIAff_dataset(dataset_path = general_dir, mode = 'training')
    test_data = IIAff_dataset(dataset_path = general_dir, mode = 'testing')
    print('The len of the training dataset is:', train_data.__len__(), ', test:', test_data.__len__())

    #C. Create the dataloaders
    train_dataloader = DataLoader(train_data, batch_size = 2, shuffle = True, collate_fn = my_collate)
    test_dataloader = DataLoader(test_data, batch_size = 2, shuffle = True, collate_fn = my_collate)
    print('The number of batches in the train datast is:', len(train_dataloader), 'testing:', len(test_dataloader))

    
    model = baseline()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print('We found a dropout layer', m)
 
    model.to(gpu_device) #Move the model to the GPU
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum = 0.9) # momentum = 0.9, weight_decay = 0.0001

    scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15*3092 - 500, 20*3092 - 500, 25*3092 - 500, 30*3092 - 500], gamma=0.9)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=500, after_scheduler=scheduler_steplr)

    if resume:
        load_path = '/home/lmur/documents/saved_models/20_ResneXT_101_dropout_0_3_encoder_FC'
        checkpoint = torch.load(load_path)
        init_epoch = checkpoint['start_epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        scheduler_warmup.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        init_epoch = 0


    for epoch in range(init_epoch, n_epochs):
        print('EPOCH', epoch)
        ##--Train one epoch
        train_one_epoch(model, optimizer, train_dataloader, gpu_device, epoch, writer, scheduler_warmup)

        ##---Evaluate one epoch on the validation set
        evaluate_one_epoch(model, test_dataloader, gpu_device, epoch, writer)
    
        if epoch == 0:
            torch.save({'start_epoch': epoch + 1, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler_steplr.state_dict()},
                        os.path.join('/home/lmur/documents/saved_models_Ensembles_Rx101', 'Ensembles_Rx101_' + str(k) + 'Ensemble'))
            print('SAVING MODEL EPOCH', epoch)
    
    #EL MODELO QUE SE LLAMA _only_ENCODER ES EL BUENO!!!!
    #Baseline 1: Mask-RCNN with lr = 0.01 and no weight decay and no momentum. We achieve the same results as the paper
    #Baseline 2: As baseline 1 but with dropout in the 2 FC layers before the bbox head. p = 0.2
    #Baseline 3: As baseline 2 but dropout also in the final FC predicted layers ->RETRAIN
    #Baseline 4: All masks and box heads with FC layers, p = 0.2
    #Baseline 5: Dropout at the end of each resnet Layer_bottleneck and in the mask and bounding box parts of the decoder