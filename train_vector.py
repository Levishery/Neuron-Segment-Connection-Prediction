import os
import sys
import time
import numpy as np
import json
import argparse
import pdb
import h5py
import pickle
from cloudvolume import CloudVolume
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from utils import get_scheduler, setup_logger
from models import GroupFreeVectorDetector, get_loss, get_loss_vector
from models import APCalculator, parse_predictions, parse_groundtruths, parse_predictions_to_vector, get_candidates, \
    get_recall_record, get_candidate_masks, get_candidates_with_terminal_thresh, get_candidates_surface_points, Traverse_end_point_sphere
from growvector.model_util_vector import VectorDatasetConfig
DC = VectorDatasetConfig()

DEBUG = 1

def parse_option():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--DBSCAN_eps', type=int, default=300, help='DBSCAN eps in nm; -1 denote no clustering')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')
    parser.add_argument('--image_model_path', type=str, default='None', help='image model path (if use image feature)')
    parser.add_argument('--preserve_scale', action='store_true', help='whether to preserve the pc scale')
    parser.add_argument('--test_baseline', action='store_true', help='surface baseline test')
    parser.add_argument('--terminal_classification', action='store_true', help='whether to predict terminal class')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='xyz_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--vector_loss_coef', default=1, type=float, help='Loss weight for vector loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--yaw_delta', default=1.0, type=float, help='delta for smoothl1 loss in yaw loss')
    parser.add_argument('--pitch_delta', default=1.0, type=float, help='delta for smoothl1 loss in pitch loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')

    # Data
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size per GPU during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 50000]')
    parser.add_argument('--data_root', default='data', help='data root path')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=40, help='print frequency')
    parser.add_argument('--visualize_freq', type=int, default=4000, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=5, help='val frequency')

    # others
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--num_gpus", type=int, default=2, help='device number for DataParallel')
    parser.add_argument("--test_index", type=int, default=5, help='device number for DataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--prediction_thresholds', type=float, default=[0.3], nargs='+',
                        help='A list of vector preserving thresholds')
    parser.add_argument('--test_only', action='store_true', help='test only setting')
    parser.add_argument('--visualize_only', action='store_true', help='visualize_only only setting')
    parser.add_argument('--candidate_record_path', default='None', type=str, help='path to record candidate')
    parser.add_argument('--candidate_scan_area', default='cone', type=str, help='difine the scan area as cone / cylinder/ sphere')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()
    args.image_model_path = None if args.image_model_path == 'None' else args.image_model_path
    return args


def is_data_parallel_state_dict(state_dict):
    for key in state_dict.keys():
        if key[:7] == 'module.':
            return True
        else:
            return False


def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    if is_data_parallel_state_dict(checkpoint['model']):
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint['model'] = new_state_dict
    if args.image_model_path is not None and not args.test_only:
        model.load_state_dict(checkpoint['model'], strict=False)
        print('Load matched modules for image module training.')
    else:
        model.load_state_dict(checkpoint['model'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('Parameter not match. Load existing parameters and reset optimizer.')
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        # state['save_path'] = 'current.pth'
        # torch.save(state, os.path.join(args.log_dir, 'current.pth'))
        print("not saving checkpoint")
        pass


def get_loader(args):
    # Init datasets and dataloaders
    train_loader, val_loader, test_loader = None, None, None
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'vector':
        from growvector.growvector_detection_dataset import GrowDetectionVotesDataset
        from growvector.model_util_vector import VectorDatasetConfig

        DATASET_CONFIG = VectorDatasetConfig()
        if args.visualize_only:
            VIS_DATASET = GrowDetectionVotesDataset('vis', num_points=args.num_point,
                                                     augment=False,
                                                     data_root=args.data_root, preserve_scale=args.preserve_scale,
                                                     terminal_classification=args.terminal_classification)
            vis_sampler = torch.utils.data.SequentialSampler(VIS_DATASET)
            vis_loader = torch.utils.data.DataLoader(VIS_DATASET,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.num_workers,
                                                     worker_init_fn=my_worker_init_fn,
                                                     pin_memory=True,
                                                     sampler=vis_sampler,
                                                     drop_last=False)
            return vis_loader, vis_loader, vis_loader, vis_loader, DATASET_CONFIG
        if not args.test_only:
            TRAIN_DATASET = GrowDetectionVotesDataset('train', num_points=args.num_point,
                                                        augment=False,
                                                        data_root=args.data_root, preserve_scale=args.preserve_scale,
                                                        terminal_classification=args.terminal_classification)
            VAL_DATASET = GrowDetectionVotesDataset('val', num_points=args.num_point,
                                                        augment=False,
                                                        data_root=args.data_root, preserve_scale=args.preserve_scale,
                                                    terminal_classification=args.terminal_classification, test_index=args.test_index)
            print(f"train_len: {len(TRAIN_DATASET)}, val_len: {len(VAL_DATASET)}, test_len: {len(TEST_DATASET)}")

            train_sampler = torch.utils.data.RandomSampler(TRAIN_DATASET)
            train_loader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    worker_init_fn=my_worker_init_fn,
                                                    pin_memory=True,
                                                    sampler=train_sampler,
                                                    drop_last=True)

            val_sampler = torch.utils.data.SequentialSampler(VAL_DATASET)
            val_loader = torch.utils.data.DataLoader(VAL_DATASET,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    worker_init_fn=my_worker_init_fn,
                                                    pin_memory=True,
                                                    sampler=val_sampler,
                                                    drop_last=False)
            n_data = len(train_loader.dataset)
            logger.info(f"length of training dataset: {n_data}")
            n_data = len(val_loader.dataset)
            logger.info(f"length of validation dataset: {n_data}")
            
            
        TEST_DATASET = GrowDetectionVotesDataset('test', num_points=args.num_point,
                                                    augment=False,
                                                    data_root=args.data_root, preserve_scale=args.preserve_scale,
                                                    terminal_classification=args.terminal_classification, test_index=args.test_index)

        
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    test_sampler = torch.utils.data.SequentialSampler(TEST_DATASET)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)

    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")
    return train_loader, val_loader, test_loader, None, DATASET_CONFIG


def get_model(args, DATASET_CONFIG):
    model = GroupFreeVectorDetector(num_yaw_bin=DATASET_CONFIG.num_yaw_bin,
                              num_pitch_bin=DATASET_CONFIG.num_pitch_bin,
                              input_feature_dim=0,
                              width=args.width,
                              bn_momentum=args.bn_momentum,
                              sync_bn=True if args.syncbn else False,
                              num_proposal=args.num_target,
                              sampling=args.sampling,
                              dropout=args.transformer_dropout,
                              activation=args.transformer_activation,
                              nhead=args.nhead,
                              num_decoder_layers=args.num_decoder_layers,
                              dim_feedforward=args.dim_feedforward,
                              self_position_embedding=args.self_position_embedding,
                              cross_position_embedding=args.cross_position_embedding,
                              image_model_path=args.image_model_path,
                              terminal_classification=args.terminal_classification,
                              dbscan_eps=args.DBSCAN_eps)

    criterion = get_loss_vector
    return model, criterion


def main(args):
    train_loader, val_loader, test_loader, vis_loader, DATASET_CONFIG = get_loader(args)

    model, criterion = get_model(args, DATASET_CONFIG)
    logger.info(str(model))
    summary_writer = SummaryWriter(args.log_dir)
    # optimizer
    if args.optimizer == 'adamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad],
                "lr": args.decoder_learning_rate,
            },
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.test_only:
        scheduler = get_scheduler(optimizer, 0, args)
    else:
        scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = model.cuda()
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model, optimizer, scheduler)

    if args.num_gpus > 1:
        gpu_device_ids = list(range(args.num_gpus))
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    if args.test_baseline:
        print('Testig all surface point baseline.')
        test_baseline_surface(test_loader, DATASET_CONFIG)
    if args.test_only:
        if args.visualize_only:
            visualize_only(vis_loader, DATASET_CONFIG, model, criterion, args)
        else:
            test_one_epoch(test_loader, DATASET_CONFIG, model, args, summary_writer, args.start_epoch)

    else:
        # get_pc_scale_stat(train_loader)
        for epoch in range(args.start_epoch, args.max_epoch + 1):

            tic = time.time()

            train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, args, summary_writer)

            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}, lr_decoder {:.5f}'.format(epoch, (time.time() - tic),
                                                                   optimizer.param_groups[0]['lr'],
                                                                   optimizer.param_groups[1]['lr']))

            save_checkpoint(args, epoch, model, optimizer, scheduler)
            if epoch % args.val_freq == 0:
                evaluate_one_epoch(val_loader, DATASET_CONFIG, model,
                                   criterion, args, summary_writer, epoch)
                # test_one_epoch(test_loader, DATASET_CONFIG, model, args, summary_writer, epoch)

        evaluate_one_epoch(val_loader, DATASET_CONFIG, args.ap_iou_thresholds, model, criterion, args, summary_writer, epoch)
        save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))
    return os.path.join(args.log_dir, f'ckpt_epoch_last.pth')


def sigmoid(x):
    ''' Numpy function for softmax'''
    s = 1 / (1 + np.exp(-x))
    return s


def save_visualize(pc_batch, center_batch, yaw_class_batch, yaw_residual_batch, pitch_class_batch, pitch_residual_batch,
                   instance_obj_mask_batch, batch_idx, epoch, save_dir, prefix='GT', show_num=4, score_to_class=False,
                   seg_id_batch=None, terminal_score_batch=None):

    target_dir = os.path.join(save_dir, 'visualize', str(epoch))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if score_to_class:
        yaw_class_batch = torch.argmax(yaw_class_batch, -1)  # B,num_proposal
        yaw_residual_batch = torch.gather(yaw_residual_batch, 2, yaw_class_batch.unsqueeze(-1))  # B,num_proposal,1
        pitch_class_batch = torch.argmax(pitch_class_batch, -1)  # B,num_proposal
        pitch_residual_batch = torch.gather(pitch_residual_batch, 2, pitch_class_batch.unsqueeze(-1))  # B,num_proposal,1
        instance_obj_mask_batch = sigmoid(instance_obj_mask_batch.numpy())[:, :, 0]
        if terminal_score_batch is not None:
            terminal_score_batch = sigmoid(terminal_score_batch.numpy())

    for i in range(show_num):
        point_cloud = np.asarray(pc_batch[i, :])
        instance_num = instance_obj_mask_batch[i, :].shape[0]
        center_data = np.zeros([instance_num, 3])
        vector_data = np.zeros([instance_num, 3])
        score_data = np.zeros([instance_num, 1])
        terminal_score_data = np.zeros([instance_num, 1])
        for j in range(instance_num):
            center_data[j, :] = center_batch[i, j, :]
            yaw = DC.class2angle(yaw_class_batch[i, j], yaw_residual_batch[i, j], type='yaw')
            pitch = DC.class2angle(pitch_class_batch[i, j], pitch_residual_batch[i, j], type='pitch')
            vector_data[j, :] = DC.Euler2vector(yaw, pitch)
            score_data[j, :] = float(instance_obj_mask_batch[i, j])
            if terminal_score_batch is not None:
                terminal_score_data[j, :] = float(terminal_score_batch[i, j])
        if seg_id_batch is not None:
            filename = os.path.join(target_dir, prefix + '_' + str(seg_id_batch[i].item()) + '.h5')
        else:
            filename = os.path.join(target_dir, prefix + '_' + str(int(batch_idx)) + '_' + str(i) + '.h5')
        with h5py.File(filename, 'w') as f:
            # 写入 point_cloud 数据
            f.create_dataset('point_cloud', data=point_cloud)
            # 写入 vectors 数据
            f.create_dataset('center', data=center_data)
            f.create_dataset('vector', data=vector_data)
            f.create_dataset('score', data=score_data)
            f.create_dataset('terminal_score', data=terminal_score_data)


def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, config, summary_writer):
    stat_dict = {}  # collect statistics
    model.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        inputs = {'point_clouds': batch_data_label['point_clouds'], 'scale': batch_data_label['scale'], 'centroid': batch_data_label['centroid']}
        # Forward pass
        end_points = model(inputs)
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     vector_loss_coef=config.vector_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     yaw_delta=config.yaw_delta,
                                     pitch_delta=config.pitch_delta,
                                     terminal_classification=config.terminal_classification
                                     )

        optimizer.zero_grad()
        loss.backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate statistics and print out
        stat_dict['grad_norm'] = grad_total_norm
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
                 for key in sorted(stat_dict.keys()) if 'loss' not in key]))
            logger.info(f"grad_norm: {stat_dict['grad_norm']}")
            logger.info(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
                                 for key in sorted(stat_dict.keys()) if
                                 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'last_' in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
            for ihead in range(config.num_decoder_layers - 2, -1, -1):
                logger.info(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
                                     for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

            for key in sorted(stat_dict.keys()):
                if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key:
                    summary_writer.add_scalar('loss/' + key, stat_dict[key] / config.print_freq,
                                              batch_idx + 1 + (epoch - 1) * len(train_loader))
                if 'proposal_' in key:
                    summary_writer.add_scalar('proposal/' + key, stat_dict[key] / config.print_freq,
                                              batch_idx + 1 + (epoch - 1) * len(train_loader))
                if 'last_' in key:
                    summary_writer.add_scalar('last/' + key, stat_dict[key] / config.print_freq,
                                              batch_idx + 1 + (epoch - 1) * len(train_loader))
                for ihead in range(config.num_decoder_layers - 2, -1, -1):
                    if f'{ihead}head_' in key:
                        summary_writer.add_scalar(f'{ihead}head/' + key, stat_dict[key] / config.print_freq,
                                                  batch_idx + 1 + (epoch - 1) * len(train_loader))

            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0
        # visualize
        if (batch_idx + 1) % config.visualize_freq == 0:
            # save GT
            save_visualize(batch_data_label['point_clouds'].cpu(), batch_data_label['center_label'].cpu(),
                       batch_data_label['yaw_class_label'].cpu(), batch_data_label['yaw_residual_label'].cpu(),
                       batch_data_label['pitch_class_label'].cpu(), batch_data_label['pitch_residual_label'].cpu(),
                       batch_data_label['box_label_mask'].cpu(), batch_idx, epoch, save_dir=config.log_dir)
            # save prediction
            save_visualize(batch_data_label['point_clouds'].cpu(), end_points['last_center'].detach().cpu(),
                       end_points['last_yaw_scores'].detach().cpu(), end_points['last_yaw_residuals'].detach().cpu(),
                       end_points['last_pitch_scores'].detach().cpu(), end_points['last_pitch_residuals'].detach().cpu(),
                       end_points['last_objectness_scores'].detach().cpu(), batch_idx, epoch, save_dir=config.log_dir, score_to_class=True, prefix='pred')


def evaluate_one_epoch(val_loader, DATASET_CONFIG, model, criterion, config, summary_writer, epoch):
    stat_dict = {}

    if config.num_decoder_layers > 0:
        prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    model.eval()  # set model to eval mode (for bn and dp)
    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    for batch_idx, batch_data_label in enumerate(val_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds'], 'scale': batch_data_label['scale'], 'centroid': batch_data_label['centroid']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     vector_loss_coef=config.vector_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     yaw_delta=config.yaw_delta,
                                     pitch_delta=config.pitch_delta)
        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

    logger.info(f'Eval: ' + ''.join(
        [f'{key} {stat_dict[key] / (len(val_loader)):.4f} \t'
         for key in sorted(stat_dict.keys()) if 'loss' not in key]))
    logger.info(''.join([f'{key} {stat_dict[key] / (len(val_loader)):.4f} \t'
                         for key in sorted(stat_dict.keys()) if
                         'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
    logger.info(''.join([f'{key} {stat_dict[key] / (len(val_loader)):.4f} \t'
                         for key in sorted(stat_dict.keys()) if 'last_' in key]))
    logger.info(''.join([f'{key} {stat_dict[key] / (len(val_loader)):.4f} \t'
                         for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
    for ihead in range(config.num_decoder_layers - 2, -1, -1):
        logger.info(''.join([f'{key} {stat_dict[key] / (len(val_loader)):.4f} \t'
                             for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

    for key in sorted(stat_dict.keys()):
        if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key:
            summary_writer.add_scalar('val_loss/' + key, stat_dict[key] / (len(val_loader)), epoch)
        if 'proposal_' in key:
            summary_writer.add_scalar('val_proposal/' + key, stat_dict[key] / (len(val_loader)), epoch)
        if 'last_' in key:
            summary_writer.add_scalar('val_last/' + key, stat_dict[key] / (len(val_loader)), epoch)
        for ihead in range(config.num_decoder_layers - 2, -1, -1):
            if f'{ihead}head_' in key:
                summary_writer.add_scalar(f'val_{ihead}head/' + key, stat_dict[key] / (len(val_loader)), epoch)


def get_pc_scale_stat(train_loader):
    scale_list = []
    for batch_idx, batch_data_label in tqdm(enumerate(train_loader)):
        m_ = batch_data_label['scale']
        for m in m_:
            scale_list.append(m)
    print(np.mean(np.array(scale_list)))
    pdb.set_trace()


def test_baseline_surface(test_loader, DATASET_CONFIG):
    candidate_dict = {}
    vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)
    [cord_start_box2, cord_end_box2] = test_loader.dataset.test_bbox
    padding = [90, 90, 30]
    vol_ffn1_data = vol_ffn1[cord_start_box2[0] / 4 - padding[0]:cord_end_box2[0] / 4 + 1 + padding[0],
                          cord_start_box2[1] / 4 - padding[1]:cord_end_box2[1] / 4 + 1 + padding[1], cord_start_box2[2] - padding[2]:cord_end_box2[2] + 1 + padding[2]].astype(np.uint64)
    radius = DATASET_CONFIG.biological_dist[0]
    block_shape = np.asarray(
        [2 * (radius // 16), 2 * (radius // 16), 2 * (radius // 40)]).astype(np.int16)
    mask = np.zeros(block_shape + [1, 1, 1], dtype=np.bool)
    center = np.asarray(mask.shape) // 2
    resolution = np.asarray([16, 16, 40])
    sphere_mask = Traverse_end_point_sphere(mask, radius, center, resolution)

    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):
        bsize = len(batch_data_label['point_clouds'])
        for j in range(bsize):
            seg_id = int(batch_data_label['seg_id'][j].detach().cpu())
            candidates = get_candidates_surface_points(batch_data_label['point_clouds'][j], batch_data_label['scale'][j], batch_data_label['centroid'][j],
                                                       padding, sphere_mask, seg_id, vol_ffn1_data, test_loader.dataset.test_bbox)
            if seg_id in candidate_dict:
                # the segments are splitted into shorter parts
                candidate_dict[seg_id] = candidate_dict[seg_id] | candidates
            else:
                candidate_dict[seg_id] = candidates
    start_dict_, end_dict_ = {}, {}
    recall, candidate_num = get_recall_record(candidate_dict, start_dict_, end_dict_, test_loader.dataset.baseline_recall_record)
    print(f'Sphere radius {radius}: Recall {recall}, Candiate Number {candidate_num}')


def visualize_only(vis_loader, DATASET_CONFIG, model, criterion, config):
    model.eval()  # set model to eval mode (for bn and dp)

    for batch_idx, batch_data_label in enumerate(vis_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds'], 'scale': batch_data_label['scale'], 'centroid': batch_data_label['centroid']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     vector_loss_coef=config.vector_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     yaw_delta=config.yaw_delta,
                                     pitch_delta=config.pitch_delta)
        seg_id = batch_data_label['seg_id'][0]
        # print(f'loss: {loss}, seg_id: {seg_id}')
        angle_loss = end_points['last_yaw_cls_loss'] + end_points['last_pitch_cls_loss']
        print(f'{seg_id}')
        if seg_id == 7608426969:
            with torch.no_grad():
                end_points = model(inputs, return_offset=True)
            attention_mask = end_points['attention_mask']
        save_visualize(batch_data_label['point_clouds'].cpu(), end_points['last_center'].detach().cpu(),
                       end_points['last_yaw_scores'].detach().cpu(), end_points['last_yaw_residuals'].detach().cpu(),
                       end_points['last_pitch_scores'].detach().cpu(),
                       end_points['last_pitch_residuals'].detach().cpu(),
                       end_points['last_objectness_scores'].detach().cpu(), batch_idx, 0, save_dir=config.log_dir,
                       score_to_class=True, show_num=len(batch_data_label['seg_id']), prefix='pred',
                       seg_id_batch=batch_data_label['seg_id'], terminal_score_batch=end_points['last_terminal_scores'].detach().cpu())
        save_visualize(batch_data_label['point_clouds'].cpu(), batch_data_label['center_label'].cpu(),
                       batch_data_label['yaw_class_label'].cpu(), batch_data_label['yaw_residual_label'].cpu(),
                       batch_data_label['pitch_class_label'].cpu(), batch_data_label['pitch_residual_label'].cpu(),
                       batch_data_label['box_label_mask'].cpu(), batch_idx, 0, save_dir=config.log_dir,
                       show_num=len(batch_data_label['seg_id']), seg_id_batch=batch_data_label['seg_id'],
                       terminal_score_batch=batch_data_label['terminal_class_label'].cpu())


def test_one_epoch(test_loader, DATASET_CONFIG, model, config, summary_writer, epoch, get_terminal_thresh=False, save_visualization=True):
    write_csv = True if config.candidate_record_path is not 'None' else False
    if write_csv:
        os.makedirs(os.path.join(config.log_dir, config.candidate_record_path, str(epoch)))
    vol_ffn1 = CloudVolume('file:///h3cstore_nt/fafb-ffn1', cache=True)
    [cord_start_box2, cord_end_box2] = test_loader.dataset.test_bbox
    padding = [90, 90, 30]
    vol_ffn1_data = vol_ffn1[cord_start_box2[0] / 4 - padding[0]:cord_end_box2[0] / 4 + 1 + padding[0],
                          cord_start_box2[1] / 4 - padding[1]:cord_end_box2[1] / 4 + 1 + padding[1], cord_start_box2[2] - padding[2]:cord_end_box2[2] + 1 + padding[2]].astype(np.uint64)

    biological_radius = DATASET_CONFIG.biological_radius
    biological_dist = DATASET_CONFIG.biological_dist
    biological_move_back = DATASET_CONFIG.biological_move_back
    terminal_thresh = DATASET_CONFIG.terminal_thresh
    biological_parameters = list(zip([biological_radius[0] for k in range(len(biological_dist))], biological_dist)) \
                            + list(zip(biological_radius[1:], [biological_dist[0] for k in range(len(biological_radius)-1)]))
    # if config.num_decoder_layers > 0:
    #     prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
    # else:
    #     prefixes = ['proposal_']  # only proposal

    # use only the last layer
    prefixes = ['last_']

    model.eval()  # set model to eval mode (for bn and dp)
    batch_pred_map_cls_dict = {k: [] for k in prefixes}

    time_start = time.time()
    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):
        import pdb; pdb.set_trace()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds'], 'scale': batch_data_label['scale'], 'centroid': batch_data_label['centroid']}
        with torch.no_grad():
            end_points = model(inputs)

        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        if save_visualization:
            save_visualize(batch_data_label['point_clouds'].cpu(), end_points['last_center'].detach().cpu(),
                       end_points['last_yaw_scores'].detach().cpu(), end_points['last_yaw_residuals'].detach().cpu(),
                       end_points['last_pitch_scores'].detach().cpu(),
                       end_points['last_pitch_residuals'].detach().cpu(),
                       end_points['last_objectness_scores'].detach().cpu(), batch_idx, epoch, save_dir=config.log_dir,
                       score_to_class=True, show_num=len(batch_data_label['seg_id']), prefix='pred', seg_id_batch=batch_data_label['partition_id'])
        for prefix in prefixes:
            batch_pred_map_cls = parse_predictions_to_vector(end_points, DATASET_CONFIG, prefix)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)

    print(f'Inference time for {len(test_loader)} segments: {time.time() - time_start}')
    # candidate evaluation
    parameters_list_1 = []
    for prediction_thresh in config.prediction_thresholds:
        parameters_list_1 = parameters_list_1 + [[prediction_thresh, biological_radius, biological_dist] for [biological_radius, biological_dist] in biological_parameters]
    parameters_list = []
    for move_back in biological_move_back:
        parameters_list = parameters_list + [[move_back, prediction_thresh, biological_radius, biological_dist] for
                                             [prediction_thresh, biological_radius, biological_dist] in parameters_list_1]
    recall_dict = {}
    candidate_num_dict = {}
    for parameters in parameters_list:
        for prefix in batch_pred_map_cls_dict.keys():
            candidate_masks, mask_indexes = get_candidate_masks(parameters, DATASET_CONFIG, test_loader.dataset.data_path, scan_area=config.candidate_scan_area)
            candidate_dict = {}
            candidate_dict_all_vectors = {}
            terminal_score_dcit = {}
            start_dict = {}
            end_dict = {}
            full_prefix = prefix + '_'.join([str(x) for x in parameters])
            print(full_prefix)
            for i in tqdm(range(len(batch_pred_map_cls_dict[prefix]))):
                bsize = len(batch_pred_map_cls_dict[prefix][i])
                for j in range(bsize):
                    instance = batch_pred_map_cls_dict[prefix][i][j]
                    candidates, starts, ends, candidates_per_vector, terminal_score = get_candidates(instance, parameters, candidate_masks, mask_indexes, vol_ffn1_data, padding, test_loader.dataset.test_bbox)
                    if instance['seg_id'] in candidate_dict:
                        # the segments are splitted into shorter parts
                        candidate_dict[instance['seg_id']] = candidate_dict[instance['seg_id']] | candidates
                        start_dict[instance['seg_id']] = start_dict[instance['seg_id']] + starts
                        end_dict[instance['seg_id']] = end_dict[instance['seg_id']] + ends
                        candidate_dict_all_vectors[instance['seg_id']] = candidate_dict_all_vectors[instance['seg_id']] + candidates_per_vector
                        terminal_score_dcit[instance['seg_id']] = terminal_score_dcit[instance['seg_id']] + terminal_score
                    else:
                        candidate_dict[instance['seg_id']] = candidates
                        start_dict[instance['seg_id']] = starts
                        end_dict[instance['seg_id']] = ends
                        candidate_dict_all_vectors[instance['seg_id']] = candidates_per_vector
                        terminal_score_dcit[instance['seg_id']] = terminal_score

            if get_terminal_thresh:
                thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                recall_dict_terminal = {}
                candidate_num_dict_terminal = {}
                for thresh in thresh_list:
                    candidate_dict_ = get_candidates_with_terminal_thresh(candidate_dict_all_vectors, terminal_score_dcit, thresh)
                    start_dict_, end_dict_ = {}, {}
                    recall, candidate_num = get_recall_record(candidate_dict_, start_dict_, end_dict_, test_loader.dataset.baseline_recall_record)
                    recall_dict_terminal[thresh] = recall
                    candidate_num_dict_terminal[thresh] = candidate_num
                    logger.info(f'Terminal thresh {thresh}: Recall {recall}, Candiate Number {candidate_num}')
                pickle_filename = os.path.join(os.path.join(config.log_dir, config.candidate_record_path, str(epoch)),
                                               'terminal_thresh_curve.pkl')
                with open(pickle_filename, 'wb') as f:
                    pickle.dump([recall_dict_terminal, candidate_num_dict_terminal], f)
                print(f' {pickle_filename} saved!')

            if config.terminal_classification:
                candidate_dict = get_candidates_with_terminal_thresh(candidate_dict_all_vectors, terminal_score_dcit,
                                                                      terminal_thresh)
                start_dict_, end_dict_ = {}, {}
                recall_dict[full_prefix], candidate_num_dict[full_prefix] = get_recall_record(candidate_dict, start_dict_, end_dict_,
                                                          test_loader.dataset.baseline_recall_record, logger=logger)
            else:
                record_path = os.path.join(config.log_dir, config.candidate_record_path, str(epoch), full_prefix)
                recall_dict[full_prefix], candidate_num_dict[full_prefix] = get_recall_record(candidate_dict, start_dict,
                                    end_dict, test_loader.dataset.baseline_recall_record, record_path, write_csv=write_csv)

            # log record
            logger.info(f'Test candidate recall: ' + '_'.join([f'{full_prefix} {recall_dict[full_prefix]:.4f} \t']))
            logger.info(f'Test candidate number: ' + '_'.join([f'{full_prefix} {candidate_num_dict[full_prefix]:.4f} \t']))

            summary_writer.add_scalar('Test/' + 'recall_' + full_prefix, recall_dict[full_prefix], epoch)
            summary_writer.add_scalar('Test/' + 'candidate_num_' + full_prefix, candidate_num_dict[full_prefix], epoch)
    if config.candidate_record_path is not 'None':
        pickle_filename = os.path.join(os.path.join(config.log_dir, config.candidate_record_path, str(epoch)), 'recall_candidate_curve.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump([recall_dict, candidate_num_dict], f)
            print(f"{pickle_filename} saved successfully !!")


if __name__ == '__main__':
    opt = parse_option()
    # torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    LOG_DIR = os.path.join(opt.log_dir, 'group_free',
                           f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(opt.log_dir, 'group_free',
                               f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    opt.log_dir = LOG_DIR
    os.makedirs(opt.log_dir, exist_ok=True)

    logger = setup_logger(output=opt.log_dir, distributed_rank=0, name="group-free")
    path = os.path.join(opt.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(opt)))

    ckpt_path = main(opt)
