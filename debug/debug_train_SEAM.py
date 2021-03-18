import importlib

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from seam.tool import imutils
from seam.tool import pyutils, torchutils
from seam.tool import visualization
from seam.voc12 import data


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


@hydra.main(config_path='../conf', config_name="train_seam")
def run_app(cfg: DictConfig) -> None:
    model = getattr(importlib.import_module(cfg.network), 'Net')()

    print(model)

    train_dataset = data.VOC12ClsDataset(cfg.train_list, voc12_root=cfg.voc12_root,
                                         cls_label_path=cfg.cls_label_path,
                                         transform=transforms.Compose([
                                             imutils.RandomResizeLong(448, 768),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                    hue=0.1),
                                             np.asarray,
                                             model.normalize,
                                             imutils.RandomCrop(cfg.crop_size),
                                             imutils.HWC_to_CHW,
                                             torch.from_numpy
                                         ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // cfg.batch_size * cfg.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[1], 'lr': 2 * cfg.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[3], 'lr': 20 * cfg.lr, 'weight_decay': 0}
    ], lr=cfg.lr, weight_decay=cfg.wt_dec, max_step=max_step)

    if cfg.weights[-7:] == '.params':
        import network.resnet38d

        assert 'resnet38' in cfg.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(cfg.weights)
    else:
        weights_dict = torch.load(cfg.weights, map_location=torch.device('cpu'))

    model.load_state_dict(weights_dict, strict=False)
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')

    timer = pyutils.Timer("Session started: ")
    for ep in range(cfg.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])
            cam1 = F.interpolate(visualization.max_norm(cam1), scale_factor=scale_factor, mode='bilinear',
                                 align_corners=True) * label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1), scale_factor=scale_factor, mode='bilinear',
                                    align_corners=True) * label

            cam2, cam_rv2 = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))
            # loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
            #            with torch.no_grad():
            #                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            loss = loss_cls + loss_er + loss_ecr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter + 1) * cfg.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                img_8 = img1[0].numpy().transpose((1, 2, 0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2, 0, 1))
                h = H // 4
                w = W // 4
                p1 = F.interpolate(cam1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2, (h, w), mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                # MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item()}
                itr = optimizer.global_step - 1
                print(loss_dict)
        else:
            print('')
            timer.reset_stage()

    torch.save(model.state_dict(), cfg.session_name + '.pth')


if __name__ == "__main__":
    run_app()
