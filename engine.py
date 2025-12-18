"""
Train and eval functions used in main.py
"""
from typing import Iterable, Optional
from einops import rearrange
import torch
import numpy
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from networks.nce_loss import ContrastiveLoss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, nce_loss: torch.nn.Module,
                    data_loader: Iterable, num_cilps:int, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False
                    ):
    # TODO fix this for finetuning
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    #criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        
        samples, targets = data['imgs'], data['labels']
        masks, has_masks = data['masks'], data['has_masks'] # mask:[b,numclips*duration,14,14], has_masks[b,num_clips*duration]
        batch_size = samples.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        if mixup_fn is not None:
            # batch size has to be an even number
            if batch_size == 1:
                continue
            if batch_size % 2 != 0:
                    samples, targets = samples[:-1], targets[:-1]
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            if epoch > 30:
                outputs, feats, patch_pred = model(samples, train=True, bfg=True) # patch_pred:[bnt*14*14,2]
            else:
                outputs, feats, patch_pred = model(samples, train=False, bfg=False)
            # outputs = outputs.reshape(batch_size, num_cilps, -1).mean(dim=1) 
            targets = targets.repeat_interleave(num_cilps)
            ce_targets = torch.cat([targets, targets, targets], dim=0)
            if epoch > 30:
                ce_loss = criterion(outputs, ce_targets)
                nceloss = nce_loss(feats, ce_targets)
            else:
                ce_loss = criterion(outputs, targets)
                nceloss = nce_loss(feats, targets)         
            
            # patch loss
            has_masks = has_masks.reshape(has_masks.shape[0], num_cilps, -1) #[b,n,t]
            b, n, t = has_masks.shape
            patch_pred = patch_pred.view(b, n, t, 14, 14, 2)
            masks = masks.view(b, n, t, 14, 14)

            has_mask_expanded = has_masks.view(b, n, t, 1, 1, 1).expand(-1, -1, -1, 14, 14, 2)
            
            valid_pred = patch_pred[has_mask_expanded.bool()].view(-1, 2)
            valid_mask = masks[has_masks.bool()].view(-1)
            patch_loss = criterion(valid_pred, valid_mask)

            loss = ce_loss + 0.5 * nceloss + 0.5 * patch_loss

        loss_value = loss.item()
        ce_loss_value = ce_loss.item()
        patch_loss_value = patch_loss.item()

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(ce_loss=ce_loss_value)
        metric_logger.update(patch_loss=patch_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_nosbi(data_loader, model, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1, temperature=0.1):
    criterion = torch.nn.CrossEntropyLoss()
    nce_criterion = ContrastiveLoss(temperature)
    to_np = lambda x: x.data.cpu().numpy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    logits = []
    binary_label = []
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    total_nce_loss = 0.
    total_count = 0.
    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=amp):

            output, feats, _ = model(images)

        output = output.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        output_np = to_np(output[:,1])

        # target = target.repeat_interleave(num_clips * model.clip_size)

        
        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
            output_ = concat_all_gather(output)
            target_ = concat_all_gather(target)
            output_np_ = to_np(output_[:,1])
            logits.append(output_np_)
            binary_label.append(target_.detach().cpu())
        else:
            outputs.append(output)
            targets.append(target)
            logits.append(output_np)
            binary_label.append(target.detach().cpu())
        batch_size = images.shape[0]

        acc1 = accuracy(output, target, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), images.size(0))
        
        nce_target = target.repeat_interleave(num_clips)
        total_nce_loss += nce_criterion(feats, nce_target) * batch_size * num_clips
        total_count += batch_size * num_clips

    # import pdb;pdb.set_trace()

    acc_outputs = numpy.stack(logits,0).reshape(-1,1)
    acc_label = numpy.stack(binary_label,0).reshape(-1,1)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    auc_score = roc_auc_score(acc_label, acc_outputs)
    
    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())
    
    nce_loss = total_nce_loss / total_count

    print('* Acc@1 {top1.global_avg:.3f} AUC {auc} ce_loss {losses.global_avg:.3f} nce_loss {cl}'
          .format(top1=metric_logger.acc1,auc=auc_score,losses=metric_logger.loss, cl=nce_loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_nosbi(data_loader, model, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1):
    criterion = torch.nn.CrossEntropyLoss()
    to_np = lambda x: x.data.cpu().numpy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    logits = []
    binary_label = []
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=amp):

            output, feats, _ = model(images)

        output = output.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        output_np = to_np(output[:,1])

        # target = target.repeat_interleave(num_clips * model.clip_size)

        
        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
            output_ = concat_all_gather(output)
            target_ = concat_all_gather(target)
            output_np_ = to_np(output_[:,1])
            logits.append(output_np_)
            binary_label.append(target_.detach().cpu())
        else:
            outputs.append(output)
            targets.append(target)
            logits.append(output_np)
            binary_label.append(target.detach().cpu())
        batch_size = images.shape[0]

        acc1 = accuracy(output, target, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), images.size(0))

    # import pdb;pdb.set_trace()

    acc_outputs = numpy.stack(logits,0).reshape(-1,1)
    acc_label = numpy.stack(binary_label,0).reshape(-1,1)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    auc_score = roc_auc_score(acc_label, acc_outputs)
    
    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())


    print('* Acc@1 {top1.global_avg:.3f} AUC {auc} ce_loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1,auc=auc_score,losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_feats(data_loader, model, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1):

    # switch to evaluation mode
    model.eval()
    feats_list = []
    for images, target in tqdm(data_loader):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=amp):

            _, feats, _ = model(images)
            feats_list.append(feats.cpu().numpy())
    return numpy.vstack(feats_list)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    #output = torch.cat(tensors_gather, dim=0)
    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')

    return output
