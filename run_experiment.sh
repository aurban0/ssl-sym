#!/bin/bash

function run_MNIST() {
    python src/train_sym.py --scores MNIST --no_contrastive --visualize_pseudolabels --customdata_train_path src/datasets/mnist_train.amat --customdata_test_path src/datasets/mnist_test.amat --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset RotMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 100 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 140 --pseudolabels_batchsize 264 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_RotMNIST60() {
    python src/train_sym.py --scores MNISTRot60 --no_contrastive --customdata_train_path src/datasets/mnist60_train.pkl --customdata_test_path src/datasets/mnist60_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_RotMNIST60_IE_AE() {
    python src/train_sym.py --scores MNISTRot60 --only_inv_ae --customdata_train_path src/datasets/mnist60_train.pkl --customdata_test_path src/datasets/mnist60_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_RotMNIST6090() {
    python src/train_sym.py --scores MNISTRot60-90 --no_contrastive --customdata_train_path src/datasets/mnist60_90_train.pkl --customdata_test_path src/datasets/mnist60_90_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 128 --batchsize_theta 100 --lr_theta 0.005 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 60 --estimator moments_outliers
}

function run_RotMNIST6090_IE_AE() {
    python src/train_sym.py --scores MNISTRot60-90 --only_inv_ae --save_oriented --customdata_train_path src/datasets/mnist60_90_train.pkl --customdata_test_path src/datasets/mnist60_90_test.pkl --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 128 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_MNISTMultiple() {
    python src/train_sym.py --scores MNISTMultiple --no_contrastive --customdata_train_path src/datasets/mnist_multiple_train.pkl --customdata_test_path src/datasets/mnist_multiple_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 256 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.06 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_MNISTMultiple_IE_AE() {
    python src/train_sym.py --scores MNISTMultiple --only_inv_ae --customdata_train_path src/datasets/mnist_multiple_train.pkl --customdata_test_path src/datasets/mnist_multiple_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 256 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_RotMNIST() {
    python src/train_sym.py --scores MNISTRot --no_contrastive --customdata_train_path src/datasets/mnist_all_rotation_normalized_float_train_valid.amat --customdata_test_path src/datasets/mnist_all_rotation_normalized_float_test.amat --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset RotMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_RotMNIST_IE_AE() {
    python src/train_sym.py --scores MNISTRot --only_inv_ae --customdata_train_path /src/datasets/mnist_all_rotation_normalized_float_train_valid.amat --customdata_test_path src/datasets/mnist_all_rotation_normalized_float_test.amat --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset RotMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 150 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.001 --hidden_dim_theta 32 --emb_dim_theta 128 --n_nearest_neig 45 --estimator moments_outliers
}

function run_MNISTMultipleGaussian() {
    python src/train_sym.py --scores MNISTMultiple_gaussian --no_contrastive --customdata_train_path src/datasets/mnist_multiple_gaussian_train.pkl --customdata_test_path src/datasets/mnist_multiple_gaussian_test.pkl --save_oriented --train --model_ind 5 --input_sz 28 --in_channels 1 --dataset PartMNIST --dataset_root src/datasets --batch_sz 100 --ae_epochs 400 --lr 0.001 --emb_dim 200 --hidden_dim 64 --loss_weight 0.03125 --temperature 0.5 --con_weight 0.01 --aug_cl crop_blur_mild --lambda_consistency 1. --theta_epochs 200 --pseudolabels_batchsize 100 --batchsize_theta 100 --lr_theta 0.005 --hidden_dim_theta 64 --emb_dim_theta 128 --n_nearest_neig 45 --estimator gaussian_half_normal_outliers
}

case "$1" in
    -MNIST)
        run_MNIST
        ;;
    -RotMNIST60)
        run_RotMNIST60
        ;;
    -RotMNIST60IEAE)
        run_RotMNIST60_IE_AE
        ;;
    -RotMNIST6090)
        run_RotMNIST6090
        ;;
    -RotMNIST6090IEAE)
        run_RotMNIST6090_IE_AE
        ;;
    -MNISTMultiple)
        run_MNISTMultiple
        ;;
    -MNISTMultipleIEAE)
        run_MNISTMultiple_IE_AE
        ;;
    -RotMNIST)
        run_RotMNIST
        ;;
    -RotMNISTIEAE)
        run_RotMNIST_IE_AE
        ;;
    -MNISTMultipleGaussian)
        run_MNISTMultipleGaussian
        ;;
    # ...
    *)
        echo "Usage: $0 -MNIST | -RotMNIST60 | -RotMNIST60IEAE | -RotMNIST6090 | -RotMNIST6090IEAE | -MNISTMultiple | -MNISTMultipleIEAE | -RotMNIST | -RotMNISTIEAE | -MNISTMultipleGaussian"
        exit 1
        ;;
esac
