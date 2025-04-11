import os
import random
import time, datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import yaml
from src import config, data
from src.checkpoints import CheckpointIO


t0 = time.time()
# Arguments
parser = argparse.ArgumentParser(
    description='Train SCONet.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
root_path = os.path.dirname(os.path.realpath(__file__))
cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))

seed = cfg['training']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
n_epoch = cfg['training']['n_epoch']
num_classes = cfg['model']['decoder_kwargs']['num_classes']

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(out_dir, 'config.yaml'), 'w') as file:
    documents = yaml.dump(cfg, file, default_flow_style=False)

# Dataset and loaders
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
learning_rate = cfg['training']['learning_rate']
weight_decay = cfg['training']['weight_decay']
optimizer_name = cfg['training']['optimizer']
momentum = cfg['training']['momentum']
Optimizer = getattr(optim, optimizer_name)

if optimizer_name in ['Adam', 'AdamW']:
    optimizer = Optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, 0.999))
else:
    optimizer = Optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer,device=device)
pre_trained = False
first_checkpoint = False
try:
    load_dict = checkpoint_io.load('model.pt')

except FileExistsError:
    try:
        load_dict = checkpoint_io.load(cfg['generation']['model_file'])
    except FileExistsError:
        pre_trained_model = cfg['training']['pre_trained_model_file']
        if pre_trained_model != "none" :
            load_dict = checkpoint_io.load( pre_trained_model )
            pre_trained = True
        else : load_dict = dict()
        first_checkpoint = True

if pre_trained :
    epoch_it = it = 0
else :
    it = load_dict.get('it', 0)
    epoch_it = load_dict.get('epoch_it', 0)

metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))


# Shorthands
check_val_every = cfg['training']['check_val_every']
compute_metric_on_training_set = cfg['training']['compute_metric_on_training_set']
classes = cfg['data']['classes']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)


while True:
    epoch_it += 1
    epoch_loss_train = []
    epoch_metric_train = []
    epoch_loss_val = []
    epoch_metric_val = []

    for batch in train_loader:
        it += 1
        eval_train_dict = trainer.train_step(batch)
        if compute_metric_on_training_set:
            epoch_metric_train += [eval_train_dict[model_selection_metric]]
        epoch_loss_train += [eval_train_dict['loss']]


    if compute_metric_on_training_set:
        epoch_metric_train = np.array(epoch_metric_train)
        for i in range(num_classes):
            logger.add_scalar('train/' + model_selection_metric + '_' + classes[i], epoch_metric_train.mean(axis=0)[i], epoch_it)
        logger.add_scalar('train/' + model_selection_metric, epoch_metric_train.mean(), epoch_it)
    logger.add_scalar('train/loss', sum(epoch_loss_train)/len(epoch_loss_train), epoch_it)

    # Save checkpoint
    print('Saving checkpoint')
    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                       loss_val_best=metric_val_best)

    # Eval on val set
    if (check_val_every>0 and (epoch_it % check_val_every) == 0):
        for val_data in val_loader:
            eval_val_dict = trainer.eval_step(val_data)
            epoch_metric_val += [eval_val_dict[model_selection_metric]]

            epoch_loss_val += [eval_val_dict['loss'].item()]
        epoch_metric_val = np.array(epoch_metric_val)
        metric_val = epoch_metric_val.mean()
        for i in range(num_classes):
            logger.add_scalar('val/' + model_selection_metric + '_' + classes[i], epoch_metric_val.mean(axis=0)[i], epoch_it)
        logger.add_scalar('val/'+model_selection_metric, metric_val, epoch_it)
        logger.add_scalar('val/loss', sum(epoch_loss_val)/len(epoch_loss_val), epoch_it)

        if ((first_checkpoint) or (model_selection_sign * (metric_val - metric_val_best) > 0)):
            metric_val_best = metric_val
            print(f'New best model ({model_selection_metric} {metric_val_best:.4f})')
            checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)
            first_checkpoint = False

    t = datetime.datetime.now()
    print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
            % (epoch_it, it, eval_train_dict['loss'], time.time() - t0, t.hour, t.minute))

    if n_epoch > 0 and epoch_it > n_epoch:
        print('Max number of epoch reached. Exiting.')
        checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                            loss_val_best=metric_val_best)

        # Save details about training.
        f= open(os.path.join(out_dir, 'training_details.txt'),"w+")
        f.write('Total number of parameters: %d\n' % nparameters)
        f.write('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d\n'
            % (epoch_it, it, eval_train_dict['loss'], time.time() - t0, t.hour, t.minute))
        f.write('Current best validation metric (%s): %.8f\n'
            % (model_selection_metric, metric_val_best))
        f.close()

        exit(3)

