import argparse
import os
import time
import pickle
import sys
from datetime import datetime
import pandas as pd
import matplotlib
import numpy as np
import torch
from data_loading_sym import RotMNIST_AE_Dataloader, PartialMNIST_AE_Dataloader
from collections import OrderedDict
from modules_sym import PartEqMod
import wandb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from utils import config_to_str, CustomDataParallel
import torch.nn.functional as F
from cyclic_distributions import cyclic_classifier

import pytorch_lightning as pl

from schedulers import construct_scheduler

# Configuration ---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# General
parser.add_argument("--model_ind", type=int, required=True)  # ID
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--input_sz", type=int, default=28)
parser.add_argument("--in_channels", default=1, type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--wandb_key", type=str, default="")

# Dataset
parser.add_argument("--dataset", type=str, default="PartMNIST")
parser.add_argument("--dataset_root", type=str,
                    default="/MNIST")
parser.add_argument("--customdata_train_path", type=str,
                    default="datasets/mnist_all_rotation_normalized_float_train_valid.amat")
parser.add_argument("--customdata_test_path", type=str,
                    default="datasets/mnist_all_rotation_normalized_float_test.amat")
parser.add_argument("--save_oriented", action='store_true', default=False)

# Output
parser.add_argument("--out_root", type=str,
                    default="/saves/")

# Invariant Autoencoder
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--batch_sz", type=int, default=240)  # Batch size pre-training
parser.add_argument("--only_inv_ae", action='store_true', default=False)  # True for standard IE-AE
parser.add_argument("--hidden_dim", default=128, type=int)  # Size of the networks
parser.add_argument("--emb_dim", default=32, type=int)  # Dimension of latent spaces
parser.add_argument("--ae_epochs", default=3, type=int)  # Inv AE training epochs
parser.add_argument("--train", action='store_true', default=False)  # Loads train dataset
parser.add_argument("--test", action='store_true', default=False)  # Loads test dataset
parser.add_argument("--loss_weight", type=float, default=1./16)

# Scheduler
parser.add_argument("--scheduler_type", type=str, default="cosine")
parser.add_argument("--scheduler_factor", type=float, default=0.1)
parser.add_argument("--scheduler_warmup_epochs", type=int, default=5)

# Pretrained Inv AE
parser.add_argument("--pretrained", action='store_true', default=False)  # Call if passing a saved Inv AE model
parser.add_argument("--pretrained_path", type=str,
                    default="./")  # Path to the Inv AE model

# Cyclic groups
parser.add_argument("--n_cyclic_groups", type=int, default=8)  # Number of cyclic groups to classify in
parser.add_argument("--discrete_groups", action='store_true', default=False)  # Cyclic symmetries

# Theta Network
parser.add_argument("--use_one_layer", action='store_true', default=False)
parser.add_argument("--lr_theta", default=0.0001, type=float)
parser.add_argument("--hidden_dim_theta", default=64, type=int)  # Size of theta network
parser.add_argument("--emb_dim_theta", default=100, type=int)  # Size of embedding space in theta network
parser.add_argument("--pseudolabels_batchsize", default=1000, type=int)  # Batch size for pseudolabels computation
parser.add_argument("--batchsize_theta", default=100, type=int)  # Batch size for theta network learning
parser.add_argument("--theta_epochs", default=150, type=int)  # Theta network training epochs
parser.add_argument("--n_nearest_neig", default=30, type=int)
parser.add_argument("--estimator", type=str,
                    default="moments_outliers")
parser.add_argument("--pretrained_theta", action='store_true', default=False)

# Visualizations/Debugs
parser.add_argument("--scores", type=str)

# Logging
parser.add_argument("--wandb_mode", type=str, default="online")
parser.add_argument("--log_every", type=int, default=100)


def main():

    config = parser.parse_args()

    # Set seed
    if config.seed == -1:
        config.seed = np.random.randint(0, 100000)
    pl.seed_everything(config.seed)

    # Set detect anomaly if debug
    if config.debug:
        torch.autograd.set_detect_anomaly(True)  # Check for anomalies in the backward pass

    # Check for correct estimators
    if config.discrete_groups:
        assert config.estimator == "cyclic", "Discrete groups only supports cyclic estimators"
    if config.estimator == "cyclic":
        assert config.discrete_groups, "Cyclic estimators only supported for discrete groups"

    # Setup ------------------------------------------------------------------------

    config.out_dir = config.out_root + str(config.model_ind) + "/"
    config.dataloader_batch_sz = int(config.batch_sz)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"exp_{timestamp}"
    os.makedirs("saves/"+folder_name)
    config.out_dir = "saves/"+folder_name+"/"

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    print("Config: %s" % config_to_str(config))

    # Initialize wandb
    if config.debug:
        os.environ['WANDB_MODE'] = 'dryrun'
    if config.wandb_key:
        wandb.login(key=config.wandb_key)
    wandb.init(
        project="unsup-equiv",
        config=config,
        entity="ck-experimental",
        mode=config.wandb_mode
    )

    # Model ------------------------------------------------------------------------
    if config.pretrained:
        # Don't train Inv AE when a pretrained model is passed
        net = PartEqMod(hparams=config)
        state_dict = torch.load(config.pretrained_path)

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "")  # remove "model."
            new_state_dict[name] = v

        if config.pretrained_theta:
            # If the pretrained model contains also a trained theta (e.g. to run an evaluation), then load theta too
            keys_to_load = {k: v for k, v in new_state_dict.items()}
        else:
            keys_to_load = {k: v for k, v in new_state_dict.items() if "theta_function" not in k}
        print("Loading pretrained model")
        net.load_state_dict(keys_to_load, strict=False)
        net = net.cuda()
        net = CustomDataParallel(net)
        net.eval()

        if isinstance(net, torch.nn.DataParallel):
            print("Model is wrapped with DataParallel.")
            print(f"Distributed on GPUs: {net.device_ids}")
        else:
            print("Model is not wrapped with DataParallel.")
    else:
        net = PartEqMod(config)

        net = net.cuda()
        net = CustomDataParallel(net)
        net.train()
        if isinstance(net, torch.nn.DataParallel):
            print("Model is wrapped with DataParallel.")
            print(f"Distributed on GPUs: {net.device_ids}")
        else:
            print("Model is not wrapped with DataParallel.")

    params_ae = list(net.encoder.parameters()) + list(net.decoder.parameters()) + list(
        net.projection_head.parameters())
    optimiser_ae = torch.optim.Adam(params_ae, lr=config.lr)

    print("Model: %s" % net)

    # Train ------------------------------------------------------------------------

    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    identity = identity.cuda()

    if config.dataset == "PartMNIST":
        print("Loading custom Partial Rot MNIST datasets (.pkl files)")
        main_dataloader = PartialMNIST_AE_Dataloader(config, train=config.train, test=config.test, shuffle=True)
    if config.dataset == "RotMNIST":
        print("Loading RotMNIST or MNIST benchmarks (.amat files)")
        main_dataloader = RotMNIST_AE_Dataloader(config, train=config.train, test=config.test, shuffle=True)

    train_dataloader = main_dataloader[0]

    # Schedulers
    config.iters_per_train_epoch = len(train_dataloader)
    config.total_train_iters = config.iters_per_train_epoch * config.ae_epochs
    ae_scheduler = construct_scheduler(optimiser_ae, config)


    print("Starting Training")
    print("Pre-training Inv AE")
    best_validation = 10000.
    train_iterations = 0
    best_validation_accuracy = 0.

    for ae_e in range(config.ae_epochs):
        if config.pretrained:
            # Don't train Inv AE when a pretrained Inv AE model is passed
            print("Pretrained autoencoder passed. Skipping Inv AE training.")
            break

        print("Sub-epoch", ae_e)
        net.train()

        for x, label in train_dataloader:
            net.zero_grad()

            x = x.cuda()

            # Calculations for encoding in Inv AE
            emb, v = net.encoder(x)
            rot = net.get_rotation_matrix(v)

            # Calculations for decoding in Inv AE
            canonical_rep = net.decoder(emb)
            recon = net.rot_img(canonical_rep, rot)

            # Reconstruction loss
            recon_loss = F.mse_loss(recon, x)

            if config.only_inv_ae:
                # Only IE-AE arch. No backpropagation through group loss
                with torch.no_grad():
                    net.eval()
                    group_loss = F.l1_loss(rot, identity.repeat(rot.shape[0], 1, 1))
                    net.train()

                ae_loss = recon_loss  # Reconstruction loss

                # Backpropagation
                optimiser_ae.zero_grad()
                ae_loss.backward()  # Only reconstruction loss
                optimiser_ae.step()
            else:
                # Ours
                group_loss = F.l1_loss(rot, identity.repeat(rot.shape[0], 1, 1))
                ae_loss = recon_loss + config.loss_weight * group_loss  # add group loss to ae loss

                # Backpropagation
                optimiser_ae.zero_grad()
                ae_loss.backward()
                optimiser_ae.step()

            train_iterations += 1

            if ((train_iterations % config.log_every) == 0) or (train_iterations == 1):
                print("Model ind %d epoch %d batch %d: ae "
                      "loss %f reconstruction loss %f group loss %f time %s" %
                      (config.model_ind, ae_e, train_iterations,
                       ae_loss.item(), recon_loss.item(), group_loss.item(), datetime.now()))

                # Log to wandb
                wandb.log({
                    "pretrain/train/ae_loss": ae_loss.item(),
                    "pretrain/train/reconst_loss": recon_loss.item(),
                    "pretrain/train/group_loss": group_loss.item(),
                    "pretrain/epoch": ae_e,
                    "pretrain/lr": optimiser_ae.param_groups[0]['lr'], # Log current learning rate
                }, step=train_iterations)
            ae_scheduler.step()

        # Validation
        net.eval()
        val_dataloader = main_dataloader[1]
        print("Validation step")
        running_ae_val_loss = []
        running_group_loss = []
        running_reconstr_loss = []
        for x, label in val_dataloader:
            net.zero_grad()

            x = x.cuda()
            # No gradients in validation
            with torch.no_grad():
                # Calculations for encoding in Inv AE
                emb, v = net.encoder(x)
                rot = net.get_rotation_matrix(v)

                # Calculations for decoding in Inv AE
                canonical_rep = net.decoder(emb)
                recon = net.rot_img(canonical_rep, rot)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, x)
                group_loss = F.l1_loss(rot, identity.repeat(rot.shape[0], 1, 1))

                if config.only_inv_ae:
                    ae_loss = recon_loss  # Only recon
                else:
                    ae_loss = recon_loss + config.loss_weight * group_loss  # Ours

            # End of torch.no_grad()
            running_reconstr_loss.append(recon_loss.item())
            running_group_loss.append(group_loss.item())
            running_ae_val_loss.append(ae_loss.item())
        val_reconstr_loss = np.mean(running_reconstr_loss)
        val_group_loss = np.mean(running_group_loss)
        val_ae_loss = np.mean(running_ae_val_loss)
        print(f"Validation loss: Invariant AE {val_ae_loss:.4f}, Recon {val_reconstr_loss:.4f}, "
              f"Group {val_group_loss:.4f}")
        print(f"Previous best validation loss: {best_validation:.4f}")

        # Train a classifier with the embeddings to evaluate using accuracy
        # Freeze weights
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        # Define the linear classifier
        if "MNIST" in config.dataset:
            num_classes = 10
        else:
            raise ValueError("Define number of classes for accuracy evaluation of this dataset")

        # Classifier and optimizer
        classifier = torch.nn.Linear(config.emb_dim, num_classes).cuda()
        optimiser_classifier = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0008)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Train the linear classifier with the net embeddings
        for _ in range(3):  # Train just for a few epochs to evaluate quality of embeddings
            for x, label in train_dataloader:
                net.zero_grad()
                classifier.zero_grad()

                x = x.cuda()
                label = label.long().cuda()

                with torch.no_grad():
                    emb, _ = net.encoder(x)

                logits = classifier(emb).float()
                classification_loss = criterion(logits, label)

                classification_loss.backward()
                optimiser_classifier.step()

        # Validation accuracy
        correct = 0
        total = 0
        for x, label in val_dataloader:
            x = x.cuda()
            label = label.long().cuda()
            with torch.no_grad():
                emb, _ = net.encoder(x)

            logits = classifier(emb).float()

            _, predicted = torch.max(logits.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        accuracy_val = 100. * correct / total
        print(f'Accuracy of the linear classifier on the validation set: {accuracy_val:.4f}%')

        # Unfreeze weights
        for param in net.parameters():
            param.requires_grad = True
        net.train()

        # Save model state
        net_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
        optimizer_state_dict = optimiser_ae.state_dict()
        torch.save(net_state_dict, config.out_dir + "last_pretrained.pt")
        torch.save(optimizer_state_dict, config.out_dir + "last_pretrained_opt.pt")
        # Save weights to wandb
        wandb.save(config.out_dir + "last_pretrained.pt", policy="now")
        wandb.save(config.out_dir + "last_pretrained_opt.pt", policy="now")

        if val_reconstr_loss < best_validation:
            best_validation = val_reconstr_loss
            wandb.log({"pretrain/val/best_ae_loss": val_ae_loss.item()}, step=train_iterations)
            print("Saving model with best reconstruction loss.")
            torch.save(net_state_dict, config.out_dir + "best_recon_pretrained.pt")
            # Save weights to wandb
            wandb.save(config.out_dir + "best_recon_pretrained.pt", policy="now")

        if accuracy_val > best_validation_accuracy:
            best_validation_accuracy = accuracy_val

        # Log to wandb
        wandb.log({
            "pretrain/val/ae_loss": val_ae_loss.item(),
            "pretrain/val/reconst_loss": val_reconstr_loss.item(),
            "pretrain/val/group_loss": val_group_loss.item(),
            "pretrain/val/accuracy_val": accuracy_val,
            "pretrain/val/best_val_acc": best_validation_accuracy,
            "pretrain/val/epoch": ae_e,
        }, step=train_iterations)

    # After pre-training, load best
    if not config.pretrained:
        net = PartEqMod(hparams=config)
        state_dict = torch.load(config.out_dir + "best_recon_pretrained.pt")

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "")  # remove "model."
            new_state_dict[name] = v
        keys_to_load = {k: v for k, v in new_state_dict.items()}
        print("Loading best model")
        net.load_state_dict(keys_to_load, strict=False)
        net = net.cuda()
        net = CustomDataParallel(net)

        if isinstance(net, torch.nn.DataParallel):
            print("Model is wrapped with DataParallel.")
            print(f"Distributed on GPUs: {net.device_ids}")
        else:
            print("Model is not wrapped with DataParallel.")

    # Save Symmetry Standardized datasets after pretraining
    net.eval()
    if config.save_oriented:
        print("Computing Sym-Std Datasets.")
        partitions = [(True, False), (False, True)]  # Two datasets: train and test
        for _train, _test in partitions:
            dataset_df = pd.DataFrame()
            if config.dataset == "PartMNIST":
                print("Loading custom Partial Rot MNIST datasets (.pkl files)")
                main_dataloader = PartialMNIST_AE_Dataloader(config, train=_train, test=_test, shuffle=True,
                                                             no_val_split=True)
            if config.dataset == "RotMNIST":
                print("Loading RotMNIST or MNIST benchmarks (.amat files)")
                main_dataloader = RotMNIST_AE_Dataloader(config, train=_train, test=_test, shuffle=True,
                                                         no_val_split=True)

            iterators_ae = (d for d in main_dataloader)
            for tup in zip(*iterators_ae):
                net.zero_grad()
                imgs_curr = tup[0][0]  # only one here
                x = imgs_curr.cuda()
                labels = tup[0][1]
                # Calculations
                with torch.no_grad():
                    emb, v = net.encoder(x)
                    rot = net.get_rotation_matrix(v)
                    oriented = net.rot_img(x, rot, rot_inverse=True)
                oriented = oriented.squeeze().detach().cpu().reshape(x.size(0),
                                                                     config.input_sz*config.input_sz*config.in_channels)
                imgDF = pd.DataFrame(oriented.numpy())
                imgDF["labels"] = labels.squeeze().cpu().numpy()
                dataset_df = pd.concat([dataset_df, imgDF])

            if _train:
                print("Saving sym-std train dataset")
                dataset_name = "invariant_dataset_train.pkl"
                dataset_df.to_pickle(config.out_dir+dataset_name)
                wandb.save(config.out_dir + dataset_name, policy="now")
            else:
                print("Saving sym-std test dataset")
                dataset_name = "invariant_dataset_test.pkl"
                dataset_df.to_pickle(config.out_dir + dataset_name)
                wandb.save(config.out_dir + dataset_name, policy="now")

    # Self-supervised learning of partial symmetries in data

    if "MNIST" in config.dataset:
        num_classes = 10

    if config.dataset == "RotMNIST":
        print("Loading RotMNIST or MNIST benchmarks (.amat files)")
        pseudolab_dataloader = RotMNIST_AE_Dataloader(config, train=config.train, test=config.test, shuffle=True,
                                                      custom_batchsize=config.pseudolabels_batchsize,
                                                          return_index=True, no_val_split=True)
        pseudolab_dataloader = pseudolab_dataloader[0]
        # Pass no_val_split = True as we want to create labels for the complete training dataset (no val split)
    if config.dataset == "PartMNIST":
        print("Loading custom Partial Rot MNIST datasets (.pkl files)")
        pseudolab_dataloader = PartialMNIST_AE_Dataloader(config, train=config.train, test=config.test, shuffle=True,
                                                      custom_batchsize=config.pseudolabels_batchsize,
                                                          return_index=True, no_val_split=True)
        pseudolab_dataloader = pseudolab_dataloader[0]
        # Pass no_val_split = True as we want to create labels for the complete training dataset (no val split)

    print("Generating pseudolabels")
    # Empty tensors for printing and debug purposes
    all_labels = None
    rotations_tensor = None
    rd_indices = None
    b_i = 0

    # Store embeddings
    embeddings_list = []

    net.eval()  # Eval mode for pseudolabels computation
    for x, label, b_indices in pseudolab_dataloader:
        net.zero_grad()
        x = x.cuda()

        # Calculations
        with torch.no_grad():
            emb, v = net.encoder(x)

        # Store embeddings
        embeddings_list.append(emb.detach().cpu())

        rot = net.get_rotation_matrix(v)
        degrees_rot = net.get_degrees(rot).squeeze()
        if (not config.estimator == "cyclic") or ((not config.estimator == "gaussian")):
            degrees_rot = torch.abs(degrees_rot)  # Take abs value
        else:
            pass  # No abs value for gaussian and cyclic estimators

        if rotations_tensor is None:
            rotations_tensor = degrees_rot.detach().cpu()
        else:
            rotations_tensor = torch.cat((rotations_tensor, degrees_rot.detach().cpu()), dim=0)

        if rd_indices is None:
            rd_indices = b_indices.detach().cpu()
        else:
            rd_indices = torch.cat((rd_indices, b_indices.detach().cpu()), dim=0)

        if all_labels is None:
            all_labels = label.cpu()
        else:
            all_labels = torch.cat((all_labels, label.cpu()), dim=0)

        if ((b_i % 100) == 0):
            # Some optional prints for checking
            """print("True labels")
            print(labels.squeeze()[:10])
            print("Nearest neighbours labels")
            print(labels.squeeze()[indices][:10])"""
        b_i += 1

    all_embeddings = torch.cat(embeddings_list, dim=0)

    # Now, compute the pairwise distance matrix
    norm_all_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)  # Normalize embeddings for cosine
    cosine_sim = torch.mm(norm_all_embeddings, norm_all_embeddings.t())
    distance_matrix = 1. - cosine_sim

    # Exclude each embedding from being its own neighbor (cosine distance max distance is 2)
    distance_matrix[range(norm_all_embeddings.size(0)), range(norm_all_embeddings.size(0))] = 2.

    # Check for correct size
    total_samples = len(pseudolab_dataloader.dataset)
    assert distance_matrix.shape == (total_samples, total_samples),\
        f"Expected shape {(total_samples, total_samples)} but got {distance_matrix.shape}"

    # Nearest Neighbours
    k = config.n_nearest_neig
    _, indices = torch.topk(distance_matrix, k, dim=-1, largest=False)

    # Compute k-nn's transformations for each embedding
    target_rots = rotations_tensor.squeeze()[indices]
    target_rots = target_rots.cuda().float()
    assert target_rots.shape == (total_samples, k)\
        , f"Expected shape {(total_samples, k)} but got {target_rots.shape}"

    # Estimations
    means = torch.mean(target_rots, dim=1, keepdim=True)
    stds = torch.std(target_rots, dim=1, keepdim=True)

    # IQR outlier detection for estimating without outliers
    outliers = target_rots > (means + 2 * stds)
    # Replace outliers with zeros
    outliers = outliers.cuda()
    zero_tensor = torch.tensor(0.).cuda()
    target_rots_no_outliers = torch.where(outliers, zero_tensor, target_rots)
    # Calculate the sum of non-outliers and the number of non-outliers for each row
    sum_no_outliers = torch.sum(target_rots_no_outliers, dim=1)
    count_no_outliers = torch.sum((target_rots_no_outliers != 0.).float(), dim=1)
    # Now, calculate the mean of each row excluding outliers
    means_no_outliers = sum_no_outliers / count_no_outliers.float()

    # Computation of different estimators, with and without outliers depending on choice
    if config.estimator == "mle":
        expected_values = torch.max(target_rots, dim=1).values
    elif config.estimator == "mle_outliers":
        expected_values = torch.max(target_rots_no_outliers, dim=1).values
    elif config.estimator == "moments":
        expected_values = 2*means
    elif config.estimator == "moments_outliers":
        expected_values = 2*means_no_outliers
    elif config.estimator == "gaussian_half_normal":
        # Std dev of the half-normal distribution
        half_normal_std_dev = torch.std(target_rots, unbiased=True, dim=1)
        # Std dev of the original normal distribution
        expected_values = half_normal_std_dev / torch.sqrt(torch.tensor(1 - 2 / np.pi))
    elif config.estimator == "gaussian_half_normal_outliers":
        # Std dev of the half-normal distribution
        half_normal_std_dev = torch.std(target_rots_no_outliers, unbiased=True, dim=1)
        # Std dev of the original normal distribution
        expected_values = half_normal_std_dev / torch.sqrt(torch.tensor(1 - 2 / np.pi))
    elif config.estimator == "gaussian":
        # Std dev of the normal distribution
        expected_values = torch.std(target_rots, unbiased=True, dim=1)
    elif config.estimator == "gaussian_no_outliers":
        # Std dev of the normal distribution
        expected_values = torch.std(target_rots_no_outliers, unbiased=True, dim=1)
    elif config.estimator == "cyclic":
        expected_values = cyclic_classifier(target_rots, n_cyclic_groups=config.n_cyclic_groups)
        expected_values = expected_values.long()
    else:
        raise ValueError("Estimator not implemented")

    expected_values = expected_values.squeeze()
    pseudolabels = expected_values.detach().cpu()
    pseudolabels = pseudolabels.squeeze()
    # Some prints for manual checks
    print("True labels")
    print(all_labels.squeeze()[:10])
    print("Nearest neighbours labels")
    print(all_labels.squeeze()[indices][:10])
    print("Pseudolabels")
    print(expected_values[:10])
    print("Nearest neighbors rotations")
    print(target_rots[:10])

    # Load the new dataloader with pseudolabels added
    if config.dataset == "RotMNIST":
        print("Loading RotMNIST or MNIST benchmarks (.amat files)")
        _, inverse_indices = rd_indices.sort()
        # Reorder the pseudolabels to match original data
        pseudolabels = pseudolabels[inverse_indices]

        # Load data with pseudolabels ordered
        theta_train_dataloader, theta_val_dataloader = RotMNIST_AE_Dataloader(config, train=config.train,
                                                                              test=config.test, shuffle=True,
                                                      custom_batchsize=config.batchsize_theta,
                                                      theta=True, pseudolabels=pseudolabels)

    if config.dataset == "PartMNIST":
        print("Loading custom Partial Rot MNIST datasets (.pkl files)")
        _, inverse_indices = rd_indices.sort()
        # Reorder the pseudolabels to match original data
        pseudolabels = pseudolabels[inverse_indices]

        # Load data with pseudolabels ordered
        theta_train_dataloader, theta_val_dataloader = PartialMNIST_AE_Dataloader(config, train=config.train,
                                                                                  test=config.test, shuffle=True,
                                                      custom_batchsize=config.batchsize_theta,
                                                      theta=True, pseudolabels=pseudolabels)

    # Free memory now that we have created the datasets with the pseudolabels
    pseudolabels = None
    all_labels = None
    rotations_tensor = None
    target_rots = None
    embeddings_list = None

    optimiser_theta = torch.optim.Adam(list(net.theta_function.parameters()), lr=config.lr_theta)

    # Schedulers in SSL symmetries
    config.total_train_iters = config.iters_per_train_epoch * config.theta_epochs
    scheduler_theta = construct_scheduler(optimiser_theta, config)

    # Self-supervised learning in theta network using the pseudolabels
    net.train()
    best_loss = 10000.
    net.theta_function._initialize_weights()
    for ae_th in range(config.theta_epochs):
        # Training theta
        b_i = 0
        epoch_loss = []
        print("Theta Network Epoch ",ae_th)
        for x, label, pseudolabels in theta_train_dataloader:
            net.zero_grad()
            x = x.cuda()
            pseudolabels = pseudolabels.cuda()

            # Network calculations
            degrees_theta = net.theta_function(x)
            degrees_theta = degrees_theta.squeeze()

            # Theta loss
            if config.estimator == "cyclic":
                pseudolabels = pseudolabels.long()
                # Classification loss
                theta_losses = F.cross_entropy(degrees_theta, pseudolabels)
            else:
                # Regression loss
                theta_losses = F.mse_loss(degrees_theta, pseudolabels)

            # Backpropagation
            optimiser_theta.zero_grad()
            theta_losses.backward()
            optimiser_theta.step()

            epoch_loss.append(theta_losses.item())
            if ((b_i % 100) == 0):
                # Show stats for theta pass
                print("Model ind %d epoch %d batch %d: ae "
                      "theta loss %f time %s" %
                      (config.model_ind, ae_th, b_i, theta_losses.item(),
                       datetime.now()))
                print("True labels")
                print(label.squeeze()[:10])
                print("Theta predictions sample")
                print(degrees_theta.squeeze()[:10])
                print("Pseudolabels")
                print(pseudolabels[:10])

                sys.stdout.flush()
            b_i += 1
            train_iterations += 1

            if ((train_iterations % config.log_every) == 0) or (train_iterations == 1):
                # Log to wandb
                wandb.log({
                    "theta/train/theta_loss": torch.mean(theta_losses).item(),
                    "theta/train/epoch": ae_th,
                    "theta/lr": optimiser_theta.param_groups[0]['lr'],  # Log current learning rate
                }, step=train_iterations)
            scheduler_theta.step()

        epoch_loss = np.mean(epoch_loss)

        # Validation step
        val_running_loss = []
        net.eval()  # Eval mode for validation
        for x, label, pseudolabels in theta_val_dataloader:
            net.zero_grad()
            x = x.cuda()
            pseudolabels = pseudolabels.cuda()

            # Theta network calculations
            with torch.no_grad():
                degrees_theta = net.theta_function(x)
                degrees_theta = degrees_theta.squeeze()

            # Theta loss
            if config.estimator == "cyclic":
                pseudolabels = pseudolabels.long()
                # Classification loss
                theta_losses = F.cross_entropy(degrees_theta.squeeze(), pseudolabels)
            else:
                # Regression loss
                theta_losses = F.mse_loss(degrees_theta.squeeze(), pseudolabels)

            val_running_loss.append(theta_losses.item())
        val_loss = np.mean(val_running_loss)

        wandb.log({
            "theta/val/theta_loss": val_loss,
            "theta/val/epoch": ae_th,
        }, step=train_iterations)

        # Prints
        print("Model ind %d epoch %d: "
              "train loss %f validation loss %f time %s" %
              (config.model_ind, ae_th,
               epoch_loss, val_loss,
               datetime.now()))
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Previous best validation loss: {best_loss:.4f}")
        sys.stdout.flush()

        # Checkpointing
        is_best = False
        if val_loss < best_loss:
            is_best = True
            best_loss = val_loss

        if ae_th % config.save_freq == 0:
            net_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
            print("Saving model periodically.")
            torch.save(net_state_dict,
                       config.out_dir + "last_model_theta.pt")
            # Save weights to wandb
            wandb.save(config.out_dir + "last_model_theta.pt", policy="now")

        if is_best:
            net_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
            print("Saving model with best validation loss")
            torch.save(net_state_dict,
                       config.out_dir + "best_model_theta.pt")
            # Save weights to wandb
            wandb.save(config.out_dir + "best_model_theta.pt", policy="now")
            with open(os.path.join(config.out_dir, "best_config.pickle"),
                      'wb') as outfile:
                pickle.dump(config, outfile)

            with open(os.path.join(config.out_dir, "best_config.txt"),
                      "w") as text_file:
                text_file.write("%s" % config)
        net.train()

    # Save the configuration
    with open(os.path.join(config.out_dir, "config.pickle"),
              'wb') as outfile:
        pickle.dump(config, outfile)

    with open(os.path.join(config.out_dir, "config.txt"),
              "w") as text_file:
        text_file.write("%s" % config)

    if config.scores:  # Plot visualizations for evaluations
        # For plotting
        if "MNIST" in config.dataset:
            num_classes = 10

        if config.dataset == "PartMNIST":
            test_dataloader = PartialMNIST_AE_Dataloader(config, train=False, test=True, shuffle=True,
                                                         no_val_split=True)
            test_dataloader = test_dataloader[0]
        if config.dataset == "RotMNIST":
            test_dataloader = RotMNIST_AE_Dataloader(config, train=False, test=True, shuffle=True,
                                                     no_val_split=True)
            test_dataloader = test_dataloader[0]

        # Visualizations of inputs, canonicals and reconstructions
        # Visualize a random sample from the dataset
        sample_size = 20
        fig, ax = plt.subplots(nrows=3, ncols=sample_size, figsize=(20, 7))
        fig_2, ax_2 = plt.subplots(nrows=2, ncols=sample_size, figsize=(20, 7))
        for x, label in test_dataloader:
            x = x.cuda()
            label = label.long().cuda()
            assert sample_size < x.shape[0]

            with torch.no_grad():
                # Encoder pass
                emb, v = net.encoder(x)
                rot = net.get_rotation_matrix(v)
                degrees_rot = net.get_degrees(rot)

                # Canonical and recon pass
                canonical_rep = net.decoder(emb)
                recon = net.rot_img(canonical_rep, rot)

                # Standarized versions
                oriented = net.rot_img(x, rot, rot_inverse=True)

            # Visualization of canonicals and reconstruction of a random sample
            for j in range(sample_size):
                ax[0, j].imshow(x[j].cpu().permute(1,2,0).squeeze())
                ax[0, j].set_xticks([])
                ax[0, j].set_yticks([])
                ax[1, j].imshow(canonical_rep[j].detach().permute(1,2,0).cpu().squeeze())
                ax[1, j].set_xticks([])
                ax[1, j].set_yticks([])
                ax[2, j].imshow(recon[j].detach().permute(1,2,0).cpu().squeeze())
                ax[2, j].set_xticks([])
                ax[2, j].set_yticks([])

            # Visualization of symmetry-standardized inputs
            for j in range(sample_size):
                ax_2[0, j].imshow(x[j].permute(1,2,0).cpu().squeeze())
                ax_2[0, j].set_xticks([])
                ax_2[0, j].set_yticks([])

                img_or = oriented[j].detach().permute(1,2,0).squeeze().cpu()
                ax_2[1, j].imshow(img_or)
                ax_2[1, j].set_xticks([])
                ax_2[1, j].set_yticks([])

            break
        # Save figures
        fig.savefig(config.out_dir + "Canonicals_all.png")
        wandb.save(config.out_dir + "Canonicals_all.png", policy="now")

        fig_2.savefig(config.out_dir + "Standarized_all.png")
        wandb.save(config.out_dir + "Standarized_all.png", policy="now")

        # Visualize for each digit
        sample_size = 20
        for target_digit in range(10):
            break_loop = False
            count = 0

            fig, ax = plt.subplots(nrows=3, ncols=sample_size, figsize=(20, 7))
            fig_2, ax_2 = plt.subplots(nrows=2, ncols=sample_size, figsize=(20, 7))

            for x, label in test_dataloader:
                if break_loop:
                    break
                x = x.cuda()
                label = label.long().cuda()

                with torch.no_grad():
                    # Encoder pass
                    emb, v = net.encoder(x)
                    rot = net.get_rotation_matrix(v)
                    degrees_rot = net.get_degrees(rot)

                    # Canonical and recon pass
                    canonical_rep = net.decoder(emb)
                    recon = net.rot_img(canonical_rep, rot)

                    # Standarized versions
                    oriented = net.rot_img(x, rot, rot_inverse=True)

                # Plot digits
                for j in range(x.shape[0]):
                    if int(label[j].item()) == target_digit:
                        ax[0, count].imshow(x[j].cpu().permute(1,2,0).squeeze())
                        ax[0, count].set_xticks([])
                        ax[0, count].set_yticks([])
                        ax[1, count].imshow(canonical_rep[j].detach().permute(1,2,0).cpu().squeeze())
                        ax[1, count].set_xticks([])
                        ax[1, count].set_yticks([])
                        ax[2, count].imshow(recon[j].detach().cpu().permute(1,2,0).squeeze())
                        ax[2, count].set_xticks([])
                        ax[2, count].set_yticks([])

                        ax_2[0, count].imshow(x[j].cpu().permute(1,2,0).squeeze())
                        ax_2[0, count].set_xticks([])
                        ax_2[0, count].set_yticks([])
                        img_or = oriented[j].detach().permute(1,2,0).cpu().squeeze()
                        ax_2[1, count].imshow(img_or)
                        ax_2[1, count].set_xticks([])
                        ax_2[1, count].set_yticks([])

                        count += 1

                        if count == sample_size:
                            fig.savefig(config.out_dir + "Canonical_" + str(target_digit) + ".png")
                            wandb.save(config.out_dir + "Canonical_" + str(target_digit) + ".png", policy="now")

                            fig_2.savefig(config.out_dir + "Standardized_" + str(target_digit) + ".png")
                            wandb.save(config.out_dir + "Standardized_" + str(target_digit) + ".png", policy="now")
                            plt.close()
                            break_loop = True
                            break

        thetas_dict = {i: [] for i in range(num_classes)}
        labels_dict = {i: [] for i in range(num_classes)}
        all_thetas_dict = {i: [] for i in range(num_classes)}

        all_transforms = []
        all_transforms_dict = {i: [] for i in range(num_classes)}
        # Define true symmetries of each dataset
        if config.scores == "MNISTRot60":
            true_thetas_dict = {0: 60., 1: 60., 2: 60., 3: 60., 4: 60.,
                                5: 60., 6: 60., 7: 60., 8: 60., 9: 60.}
        elif config.scores == "MNISTRot60-90":
            true_thetas_dict = {0: 60., 1: 60., 2: 60., 3: 60., 4: 60.,
                                5: 90., 6: 90., 7: 90., 8: 90., 9: 90.}
        elif config.scores == "MNISTMultiple":
            true_thetas_dict = {0: 0, 1: 18, 2: 36, 3: 54, 4: 72,
                                5: 90, 6: 108, 7: 126, 8: 144, 9: 162}
        elif config.scores == "MNISTMultiple_gaussian":
            std_dev_dict = {0: 0, 1: 9, 2: 18, 3: 27, 4: 36,
                            5: 45, 6: 54, 7: 63, 8: 72, 9: 81}
            true_thetas_dict = std_dev_dict
        elif config.scores == "MNISTRot":
            true_thetas_dict = {0: 180., 1: 180., 2: 180., 3: 180., 4: 180.,
                                5: 180., 6: 180., 7: 180., 8: 180., 9: 180.}
        elif config.scores == "MNIST":
            true_thetas_dict = {0: 0., 1: 0., 2: 0., 3: 0., 4: 0.,
                                5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        elif config.scores == "MNISTC2C4":
            true_thetas_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1,  # class 0 is C1, class 1 is C2...
                                5: 3, 6: 3, 7: 3, 8: 3, 9: 3}
        else:
            ValueError("Dataset not implemented")

        net.eval()
        for x, label in test_dataloader:
            x = x.cuda()
            label = label.long().cuda()

            with torch.no_grad():
                # Encoder pass
                emb, v = net.encoder(x)
                rot = net.get_rotation_matrix(v)
                degrees_rot = net.get_degrees(rot)

                # Theta function
                degrees_theta = net.theta_function(x).squeeze()
                all_transforms.extend(degrees_rot.cpu().numpy())

                # Loop through each class and gather rotations
                for lab in range(num_classes):
                    mask = (label == lab)

                    # Extract values
                    sub_thetas = degrees_theta[mask].cpu().numpy()
                    sub_labels = label[mask].cpu().numpy()
                    sub_degrees = degrees_rot[mask].cpu().numpy()

                    # Save the values
                    mean_degrees = np.nanmean(sub_thetas)  # Beware of nan values when the mask is all False
                    thetas_dict[lab].append(mean_degrees)
                    labels_dict[lab].extend(sub_labels.tolist())
                    all_thetas_dict[lab].extend(sub_thetas.tolist())
                    all_transforms_dict[lab].extend(sub_degrees.tolist())

        # Calculate scores for the symmetry level prediction (continuous distributions case)
        if not config.discrete_groups:
            final_results = {}
            symmetry_level_list= []
            with open(config.out_dir + 'results.txt', 'w') as f:
                for lab in range(num_classes):
                    f.write(f"\n====== Results for class {lab} ======")

                    # Nan values
                    nan_values = np.sum(np.isnan(thetas_dict[lab]))
                    f.write("\nNan values: " + str(nan_values))

                    # Calculate mean of thetas, per-digit
                    mean_value = np.nanmean(thetas_dict[lab])
                    f.write(f'\nMean Value of Symmetry Predictions:{lab}: {mean_value}')

                    # Calculate MAE
                    if len(all_thetas_dict[lab]) > 0:
                        differences = np.array(all_thetas_dict[lab]) - true_thetas_dict[lab]
                        if not np.all(np.isfinite(differences)):
                            print(f"Warning: Some non-finite values detected for lab={lab} after subtraction.")
                            differences = np.nan_to_num(
                                differences)  # Replace NaNs and infinite values with 0

                        mae_value = np.mean(np.abs(differences))
                        symmetry_level_list.append(mae_value)
                        f.write(f'\nMAE for class {lab} : {mae_value}')

                        # Calculate standard deviation
                        std_dev = np.std(differences)
                        f.write(f'\nstd dev for class {lab} : {std_dev}\n')

                        # Save to dict
                        final_results[lab] = (mean_value, std_dev)
                    else:
                        f.write(f'\nMAE for class {lab} : Not Computed due to empty array')
                        f.write(f'\nstd dev for class {lab} : Not Computed due to empty array\n')

                sys.stdout.flush()
            # Save the final result in a csv
            df_results = pd.DataFrame.from_dict(final_results, orient="index", columns=["Mean", "Standard Deviation"])
            df_results.to_csv(config.out_dir + "final_results.csv")
            wandb.save(config.out_dir + "final_results.csv", policy="now")

            # Calculate MAE between mean predicted symmetry level and true symmetry level
            symmetry_level_means = list(df_results["Mean"].values)
            print("Mean per-class predicted symmetry level:")
            print(symmetry_level_means)
            symmetry_level_error = mean_absolute_error(symmetry_level_means, list(true_thetas_dict.values()))
            print("Total MAE in predicting the symmetry level:", symmetry_level_error)
            wandb.log({"symmetry_error": float(symmetry_level_error)}, step=train_iterations + 1, commit=True)
            wandb.save(config.out_dir + 'results.txt')
        # Calculate scores for symmetry level prediction (discrete case)
        if config.discrete_groups:
            final_results = {}
            with open(config.out_dir + 'results.txt', 'w') as f:
                for lab in range(num_classes):
                    f.write(f"\n====== Results for class {lab} ======")

                    # Nan values
                    nan_values = np.sum(np.isnan(thetas_dict[lab]))
                    f.write("\nNan values: " + str(nan_values))

                    # Count correct predictions
                    true_thetas_array = np.repeat(true_thetas_dict[lab], len(all_thetas_dict[lab]))
                    predicted_classes = np.argmax(all_thetas_dict[lab], axis=1)
                    correct_predictions = np.sum(predicted_classes == true_thetas_array)
                    total_predictions = len(true_thetas_array)

                    # Calculate accuracy
                    if total_predictions > 0:
                        accuracy = correct_predictions / total_predictions
                        f.write(f'\nAccuracy for class {lab}: {accuracy}')

                        # Save to dict
                        final_results[lab] = accuracy
                    else:
                        f.write('\nAccuracy for class {lab}: Not Computed due to empty array\n')

                sys.stdout.flush()
            # Save the final result in a csv
            df_results = pd.DataFrame.from_dict(final_results, orient="index", columns=["Accuracy"])
            df_results.to_csv(config.out_dir + "final_results.csv")
            wandb.save(config.out_dir + "final_results.csv", policy="now")
            # Calculate total accuracy
            symmetry_level_error = np.sum(df_results["Accuracy"].values) / num_classes
            print("Total accuracy in predicting the symmetry level:", symmetry_level_error)
            wandb.log({"symmetry_error": float(symmetry_level_error)}, step=train_iterations + 1, commit=True)
            wandb.save(config.out_dir + 'results.txt')

        # Print Histogram for Psi
        fig, ax = plt.subplots(figsize=(12, 8))
        df = pd.DataFrame()
        df["psi"] = all_transforms
        kde = df["psi"].plot.kde(ax=ax, label="_nolegend_", lw=2)
        # Get x and y values of the KDE curve
        x, y = kde.get_lines()[0].get_data()
        ax.fill_between(x, y, color="skyblue", alpha=0.5)
        plt.xlim(-180, 180)
        ax.set_xlabel("Angle of Rotation (º)", fontsize=16)
        plt.yticks([])
        ax.set_ylabel("")
        # Adjust fontsize for tick labels
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(bottom=0)
        plt.title("Density of Group Action Estimator (Test Set)", fontsize=16)
        plt.savefig(config.out_dir + "histogram.png")
        wandb.save(config.out_dir + "histogram.png", policy="now")
        plt.close()

        # Print Histogram for Psi per label
        for lab in range(num_classes):
            fig, ax = plt.subplots(figsize=(12, 8))
            df = pd.DataFrame()
            df["psi"] = all_transforms_dict[lab]
            kde = df["psi"].plot.kde(ax=ax, label="_nolegend_", lw=2)
            # Save the psi values for later visualization purposes
            df.to_csv(config.out_dir + "rotations_class_" + str(lab) + ".csv", index=False)
            wandb.save(config.out_dir + "rotations_class_" + str(lab) + ".csv", policy="now")
            # Get x and y values of the KDE curve
            x, y = kde.get_lines()[0].get_data()
            ax.fill_between(x, y, color="skyblue", alpha=0.5)
            plt.xlim(-180, 180)
            ax.set_xlabel("Angle of Rotation (º)", fontsize=16)
            plt.yticks([])
            ax.set_ylabel("")
            # Adjust fontsize for tick labels
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            ax.set_ylim(bottom=0)
            plt.title("Density of Group Action Estimator (Test Set)", fontsize=16)
            plt.savefig(config.out_dir + "histogram" + str(lab) + ".png")
            wandb.save(config.out_dir + "histogram" + str(lab) + ".png", policy="now")
            plt.close()


if __name__ == "__main__":
    main()