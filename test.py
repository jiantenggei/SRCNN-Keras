from model import built_model
import os
from utils import load_test,psnr
import cv2
def test():
    input_shape = (None, None, 1)
    scale = 3
    srcnn_model = built_model(input_shape=input_shape)
    srcnn_model.load_weights(r'model\srcnn_weight.hdf5')

    X_pre_test, X_test, Y_test = load_test(scale=scale)

    predicted_list = []

    for img in X_test:
        img = img.reshape(1,img.shape[0],img.shape[1],1)
        predicted=srcnn_model.predict(img)
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],1))
    n_img = len(predicted_list)
    dirname = './result'
    for i in range(n_img):
        imgname = 'image{:02}'.format(i)
        cv2.imwrite(os.path.join(dirname,imgname+'_original.bmp'), X_pre_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_input.bmp'), X_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_answer.bmp'), Y_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_predicted.bmp'), predicted_list[i])
          # 计算峰值信噪比
        answer = psnr(X_test[i],predicted_list[i])
        print(imgname+"_psnr:",answer)

if __name__ == '__main__':
    test()