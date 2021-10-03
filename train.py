from tensorflow.python.keras.saving.model_config import model_from_config
from model import built_model
from utils import load_train
from keras.optimizers import Adam

def train():
    # ==========================
    # input_shape 输入图片大小
    # stride 原图片采样间隔
    # batch_size epochs learn_rate
    #============================
    input_shape = (33, 33, 1)
    stride = 14
    batch_size = 64
    epochs=100
    learning_rate=0.001

    # 定义模型
    srcnn_model = built_model(input_shape=input_shape)
    srcnn_model.load_weights(r'model\srcnn_weight.hdf5')
    srcnn_model.summary()

    # 加载数据
    X_train, Y_train = load_train(image_size=input_shape[0], stride=stride)
    print(X_train.shape, Y_train.shape)
    optimizer = Adam(lr=learning_rate)
    srcnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    srcnn_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)
    srcnn_model.save(r'model/srcnn.h5')

if __name__ == '__main__':
    train()
