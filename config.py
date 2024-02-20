# Dataset
imagenet_root = '/data/ljc/datasets/imagenet64/processed'

# Training
epochs = 20
lr = 1e-3
test_ep_gap = 1

bs_imagenet = 2048

# Model size
SF_T = {}
SF_T['stage_blocks'] = [2, 2, 6, 2]
SF_T['embed_dim'] = 96

SF_S = {}
SF_S['stage_blocks'] = [2, 2, 18, 2]

SF_SIZE = SF_T