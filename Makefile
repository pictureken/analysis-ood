#dataset params
dataset = CIFAR10
augnum = 10
transform_method = flip_crop
corruption_level = 1
corruption_method = gaussian_noise

#cifar10 train params
train_batch_size = 128
test_batch_size = 100
num_workers = 2
epoch = 4000
learning_late = 1e-4
gpu_device = cuda:0
model = ResNet18
noise_level = 0
train_transform_method = flip_crop
model_size = 1

#dataset dir
DATASET_DIR = dataset/external/${dataset}
AUG_DATASET_DIR = dataset/processed/${dataset}/test_batch_aug_${augnum}_${transform_method}

#cifar10 train dir
CIFAR10_DATASET_DIR = dataset/raw/cifar-10-batches-py
PRE_TRAINED_MODEL_DIR = models/${model}/labelnoise${noise_level}/${train_transform_method}/*${model_size}.pt

#dataset phony
.PHONY: aug_data
.PHONY: aug

#cifar10 train phony
.PHONY: model
.PHONY: train

.PHONY: eval

# generate aug dataset
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

# train
model:
	make models/${model}/labelnoise${noise_level}/${train_transform_method}
	@for i in {1..5}; do make train model_size=$$i; done

models/${model}/labelnoise${noise_level}/${train_transform_method}:
	mkdir -p models/${model}/labelnoise${noise_level}/${train_transform_method}

train: ${PRE_TRAINED_MODEL_DIR}

${PRE_TRAINED_MODEL_DIR}: ${CIFAR10_DATASET_DIR}
	python src/models/train_cifar10_model.py \
	--train-batch-size ${train_batch_size} \
	--test-batch-size ${test_batch_size} \
	--num-workers ${num_workers} \
	--epoch ${epoch} \
	--learning-late ${learning_late} \
	--gpu-device ${gpu_device} \
	--model ${model} \
	--model-size ${model_size} \
	--noise-level ${noise_level} \
