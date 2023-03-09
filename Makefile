dataset = CIFAR10
augnum = 10
transform_method = flip_crop
corruption_level = 1
corruption_method = gaussian_noise

DATASET_DIR = dataset/external/${dataset}
AUG_DATASET_DIR = dataset/processed/${dataset}/test_batch_aug_${augnum}_${transform_method}

.PHONY: aug_data
.PHONY: aug

.PHONY: train

aug_data:
	make aug transform_method=${transform_method} augnum=${augnum} dataset=CIFAR10 
	make aug transform_method=${transform_method} augnum=${augnum} dataset=CIFAR10-C
	make aug transform_method=${transform_method} augnum=${augnum} dataset=ImageNet
	make aug transform_method=${transform_method} augnum=${augnum} dataset=LSUN
	make aug transform_method=${transform_method} augnum=${augnum} dataset=SVHN

aug: ${AUG_DATASET_DIR}

${AUG_DATASET_DIR}: ${DATASET_DIR}
	python src/data/make_aug_dataset.py \
	--dataset ${dataset} \
	--transform-method ${transform_method} \
	--augnum ${augnum} \
	--corruption-level ${corruption_level} \
	--corruption-method ${corruption_method} \