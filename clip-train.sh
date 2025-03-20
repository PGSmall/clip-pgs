######################################### CLIP-PGS-0.3 #########################################
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 8 --master_port 29500 -m training.main \
    --accum-freq 1 \
    --save-frequency 8 \
    --zeroshot-frequency 1 \
    --save-most-recent \
    --train-num-samples 10968539 \
    --train-data="/data/CC12M/{00000..01242}.tar" \
    --warmup 10000 \
    --imagenet-val="/data/ImageNet1k/val" \
    --batch-size=512 \
    --epochs=32 \
    --lr=1e-3 \
    --workers=8 \
    --model ViT-B-16 \
    --seed 0 \
    --force-patch-dropout 0.05 \
    --target-mask-ratio 0.5 \
    --min-mask-ratio 0.3 \
    --use-ed \
    --use-otn \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing

######################################### CLIP-PGS-0.5 #########################################
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 8 --master_port 29500 -m training.main \
    --accum-freq 1 \
    --save-frequency 8 \
    --zeroshot-frequency 1 \
    --save-most-recent \
    --train-num-samples 10968539 \
    --train-data="/data/CC12M/{00000..01242}.tar" \
    --warmup 10000 \
    --imagenet-val="/data/ImageNet1k/val" \
    --batch-size=512 \
    --epochs=32 \
    --lr=1e-3 \
    --workers=8 \
    --model ViT-B-16 \
    --seed 0 \
    --force-patch-dropout 0.05 \
    --target-mask-ratio 0.5 \
    --min-mask-ratio 0.5 \
    --use-ed \
    --use-otn \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing
