########################################################Zero-Shot Classification########################################################
########################################################################################################################################
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-cls.txt" \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --num_workers 1

CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-cls.txt" \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
#                                            --num_workers 1
########################################################################################################################################

##########################################################Zero-Shot Retrieval###########################################################
########################################################################################################################################
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-retr.txt" \
                                           --recall_k 1 5 10 \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"

CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-retr.txt" \
                                           --recall_k 1 5 10 \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
########################################################################################################################################

######################################################Linear Probing Classification#####################################################
########################################################################################################################################
# Linear Probing: CIFAR-10
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/vtab/cifar10 \
                                           --task linear_probe \
                                           --feature_root "clip-test-features" \
                                           --dataset_root ./clip-benchmark/cifar10/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test

# Linear Probing: CIFAR-100
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/vtab/cifar100 \
                                           --task linear_probe \
                                           --feature_root "./clip-test-features/" \
                                           --dataset_root ./clip-benchmark/cifar100/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test

# Linear Probing: IN-1K
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/imagenet1k \
                                           --task linear_probe \
                                           --feature_root "./clip-test-features/" \
                                           --dataset_root ./clip-benchmark/imagenet1k/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test

# Linear Probing: CIFAR-10
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/vtab/cifar10 \
                                           --task linear_probe \
                                           --feature_root "clip-test-features" \
                                           --dataset_root ./clip-benchmark/cifar10/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test

# Linear Probing: CIFAR-100
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/vtab/cifar100 \
                                           --task linear_probe \
                                           --feature_root "./clip-test-features/" \
                                           --dataset_root ./clip-benchmark/cifar100/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test

# Linear Probing: IN-1K
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset wds/imagenet1k \
                                           --task linear_probe \
                                           --feature_root "./clip-test-features/" \
                                           --dataset_root ./clip-benchmark/imagenet1k/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --batch_size 64 --fewshot_lr 0.1 --fewshot_epochs 10 \
                                           --batch_size 512 --train_split train --test_split test
########################################################################################################################################

##########################################################Robustness Assessment#########################################################
########################################################################################################################################
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-robu.txt" \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --num_workers 1

CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset "task-zero-shot-robu.txt" \
                                           --dataset_root "./clip-benchmark/{dataset_cleaned}" \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
                                           --num_workers 1
########################################################################################################################################

########################################################Language Compositionality#######################################################
########################################################################################################################################
CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.5.pt \
                                           --model ViT-B-16 \
                                           --dataset sugar_crepe \
                                           --dataset_root ./clip-benchmark/sugar_crepe/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"

CUDA_VISIBLE_DEVICES=0 clip_benchmark eval --pretrained ./clip-ckpts/clip-pgs-0.3.pt \
                                           --model ViT-B-16 \
                                           --dataset sugar_crepe \
                                           --dataset_root ./clip-benchmark/sugar_crepe/ \
                                           --output "./clip-test-results/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
########################################################################################################################################
