import os, csv, time, yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from argparse import ArgumentParser
from tqdm import tqdm
from .dataset import build_dataloader
from .model import ResNet18Classifier
from .metrics import MetricPack
from .utils import set_seed, AverageMeter


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    losses = AverageMeter()
    metrics = MetricPack(num_classes=model.head.out_features, device=device)
    pbar = tqdm(loader, desc='train', ncols=70, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = model.loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = model.loss(logits, labels)
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        metrics.update(logits.detach(), labels)
        pbar.set_postfix(loss=f"{losses.avg:.4f}")

    return losses.avg, metrics.compute()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = AverageMeter()
    metrics = MetricPack(num_classes=model.head.out_features, device=device)
    for imgs, labels in tqdm(loader, desc='val'):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = model.loss(logits, labels)
        losses.update(loss.item(), imgs.size(0))
        metrics.update(logits, labels)
    return losses.avg, metrics.compute()

def main(cfg):
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ld, val_ld, test_ld, classes = build_dataloader(cfg['data_root'], cfg['batch_size'], cfg['num_workers'], cfg['strong_aug'])

    model = ResNet18Classifier(
        num_classes=len(classes),
        pretrained=cfg['pretrained'],
        dropout_p=cfg['dropout'],
        label_smoothing=cfg['label_smoothing'],
    ).to(device)

    base_lr = cfg['lr'] * (cfg['batch_size']/64)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=cfg['weight_decay'])

    warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=cfg['warmup'])
    cosine = CosineAnnealingLR(optimizer, T_max=cfg['epochs'] - cfg['warmup'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg['warmup']])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

    os.makedirs(cfg['out_dir'], exist_ok=True)
    best_top1, best_path = 0.0, os.path.join(cfg['out_dir'], 'best.pt')

    log_csv = os.path.join(cfg['out_dir'], 'log.csv')
    if not os.path.exists(log_csv):
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_top1', 'val_top5', 'val_f1', 'lr'])

    epoch_times = []
    epoch_bar = tqdm(range(cfg['epochs']), desc='epochs', dynamic_ncols=True, position=0, leave=True)

    for epoch in epoch_bar:
        t0 = time.time()
        scheduler.step()

        train_loss, train_m = train_one_epoch(model, train_ld, optimizer, device, scaler)
        val_loss, val_m = validate(model, val_ld, device) 

        cur_lr = optimizer.param_groups[0]['lr']

        epoch_sec = time.time() - t0
        epoch_times.append(epoch_sec)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remain_sec = avg_epoch * (cfg['epochs'] - (epoch + 1))
        eta_str = tqdm.format_interval(remain_sec)

        epoch_bar.set_postfix(epoch_sec=f'{epoch_sec:.1f}', eta=eta_str, lr=f'{cur_lr:.2e}')

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, train_loss, val_loss,
                val_m['top1'], val_m.get('top5', 0.0), val_m['macro_f1'], cur_lr
            ])

        # log_line = (f"epoch={epoch+1}/{cfg['epochs']} "
        #             f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        #             f"val_top1={val_m['top1']:.4f} val_top5={val_m.get('top5',0.0):.4f} val_f1={val_m['macro_f1']:.4f}")
        # print()
        # print(log_line)
        
        epoch_bar.write(
            f"epoch={epoch+1}/{cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_top1={val_m['top1']:.4f} val_top5={val_m.get('top5',0.0):.4f} "
            f"val_f1={val_m['macro_f1']:.4f} lr={cur_lr:.2e}"
        )

        if val_m['top1'] > best_top1:
            best_top1 = val_m['top1']
            torch.save({'model': model.state_dict(), 'classes': classes, 'cfg': cfg}, best_path)

    print(f"Best val top1: {best_top1:.4f}; saved to: {best_path}")
    print(f"Log saved to: {log_csv}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)