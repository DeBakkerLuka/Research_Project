# some training parameters
EPOCHS = 1
MODEL="VGG19" # Standard is VGG19, options are : resnet152, inceptionv3, vgg16 or vgg19
image_height = 224
image_width = 224
channels = 3
save_model_dir = "weights_recog"
save_model_name = "Final_Model_VGG19_onepiece.h5"
dataset_dir = "Split_dataset/"
train_dir = dataset_dir + "train"
test_dir = dataset_dir + "test"
volumes_dir = '../../Data/Volumes'