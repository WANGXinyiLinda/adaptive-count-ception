from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras.callbacks import ModelCheckpoint, TensorBoard

from data import *
from model import *
from utilities import *

# train on VGG
TRAIN = True 
batch_size = 4
epochs = 100
vgg_train_val_split = 0.9
vgg_num_train = int(vgg_train_val_split*200)

print("num vgg train data:", vgg_num_train)
print("num vgg val data: ", 200-vgg_num_train)

datasetfilename = str(scale) + "-" + str(patch_size) + "-" + str(framesize) + "-" + str(stride) + "-" + kernel + "-cell2-dataset.p"

img_file_path = []
for filename in glob.iglob('cells/*cell.png'):
    xml = filename.split("cell.png")[0] + "dots.png"
    img_file_path.append([filename, xml])

# load data
np_dataset_x, np_dataset_y, np_dataset_c = data_process(datasetfilename, img_file_path, verbose = False)
# split train-val-test
np_dataset_x_train = np_dataset_x[:vgg_num_train]
np_dataset_y_train = np_dataset_y[:vgg_num_train]
np_dataset_c_train = np_dataset_c[:vgg_num_train]

np_dataset_x_valid = np_dataset_x[vgg_num_train:]
np_dataset_y_valid = np_dataset_y[vgg_num_train:]
np_dataset_c_valid = np_dataset_c[vgg_num_train:]

# train model from scratch
model = build_model(train_conv=train)

bestcheck = ModelCheckpoint(filepath="model-best.h5", verbose=1, save_weights_only=True, save_best_only=True)
every10check = ModelCheckpoint(filepath="model-cp.{epoch:02d}-{val_loss:.2f}.h5", verbose=1, period=10, save_weights_only=True)
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit(np_dataset_x_train, np_dataset_y_train, epochs=epochs, batch_size = batch_size,
                    validation_data = (np_dataset_x_valid, np_dataset_y_valid), 
                    callbacks=[bestcheck, every10check, tbCallBack])

# load pretrained
model = build_model()
model.load_weights("model-best.h5", by_name=True)

pred = model.predict(np_dataset_x_test, batch_size=1)
plot_map(pred[0], "prediction0.png")
plot_map(np_dataset_y_test[0], "groundtruth0.png")
preds = sum_count_map(pred, ef)
tests = np.concatenate(np_dataset_c_test)
# order = np.argsort(tests)
# print(preds[order])
# print(tests[order])
print("Test MSE:", np.mean((preds-tests)**2))
print("Test MAE:", np.mean(np.abs(preds-tests)))
