from models.baseline import data_to_GPU, object_detection_loss
from common.metrics import init_loss_stats, compute_epoch_loss
import torch


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, lr_scheduler):
    model.train() #CHECK THIS
    stats_train_loss = init_loss_stats()
    for batch, data in enumerate(data_loader, 0):
        if (batch % 200) == 0:
            print('---Epoch---', epoch, 'batch', batch, 'lr',lr_scheduler.get_last_lr()[0])
        #Send the data to the GPU!
        images_cuda, targets_cuda = data_to_GPU(data, device)
    
        losses = model(images_cuda, targets_cuda)
        total_loss = compute_all_losses(losses)
        optimizer.zero_grad() # Zero out the gradients
        total_loss.backward() # Backward pass: gradient of loss wr. each model parameter
        optimizer.step() # Update parameters of model by gradients
        n_iteration = epoch * 3092 + batch
        lr_scheduler.step(n_iteration)

        accumulate_metrics_batch(losses, total_loss, writer, epoch, batch, stats_train_loss)
        
        writer.add_scalar('Batch/Learning rate', lr_scheduler.get_last_lr()[0], n_iteration)
        del losses, total_loss

    epoch_loss = compute_epoch_loss(stats_train_loss)
    writer.add_scalar('Epoch/Total_loss_TRAIN', epoch_loss['loss_total'], epoch)


def compute_all_losses(loss_dict):
    w_class, w_box, w_objectness, w_box_reg, w_mask = 1.0, 1.0, 1.0, 1.0, 1.0 #w_aff * loss_dict['loss_affordances'] + 
    
    total_loss = w_class * loss_dict['loss_classifier'] + \
                 w_box * loss_dict['loss_box_reg'] + w_objectness * loss_dict['loss_objectness'] + \
                 w_box_reg * loss_dict['loss_rpn_box_reg'] + w_mask * loss_dict['loss_mask']
    return total_loss

def accumulate_metrics_batch(losses, total_loss, writer, epoch, batch, loss_stats):
    training_n_batchs = 3092
    n_iter = epoch * training_n_batchs + batch

    #We print and we accumulate the losses of the batch
    writer.add_scalar('Batch/Individual_losses/Loss_mask', losses['loss_mask'].item(), n_iter)
    loss_stats['loss_mask'].append(losses['loss_mask'].item())
    writer.add_scalar('Batch/Individual_losses/Loss_class', losses['loss_classifier'].item(), n_iter)
    loss_stats['loss_classifier'].append(losses['loss_classifier'].item())
    writer.add_scalar('Batch/Individual_losses/Loss_box_reg', losses['loss_box_reg'].item(), n_iter)
    loss_stats['loss_box_reg'].append(losses['loss_box_reg'].item())
    writer.add_scalar('Batch/Individual_losses/Loss_objecte', losses['loss_objectness'].item(), n_iter)
    loss_stats['loss_objectness'].append(losses['loss_objectness'].item())
    writer.add_scalar('Batch/Individual_losses/Loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), n_iter)
    loss_stats['loss_rpn_box_reg'].append(losses['loss_rpn_box_reg'].item())

    #Also with the total loss, of course :)
    writer.add_scalar('Batch/Total_loss', total_loss.item(), n_iter)
    loss_stats['loss_total'].append(total_loss.item())
    

