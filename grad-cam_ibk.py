# from keras import backend as K

# if 'tensorflow' == K.backend():
#     import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
# import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import time
import datetime
import os
from keras.models import model_from_json

# path_img = 'img_0691_21(1).jpg'
#path_img = 'img_0656_11.jpg'
#path_img = '29.jpg'
path_img = sys.argv[1]
hw_height = 200 # 画像の縦サイズ
hw_width = 200  # 画像の横サイズ
classes = 2     # クラス数

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img = cv2.imread(path)
    #dst = cv2.resize(img, dsize=None, fx=0.78125, fy = 0.78125)
    #dst = cv2.resize(img, dsize=(200, 200))
    dst = cv2.resize(img, dsize=(hw_height, hw_width))  # 画像をhwにリサイズ
    #img_path = sys.argv[1]
    #img_path = "img_0691_21.jpg"
    #img_path = path
    #img = image.load_img(img_path, target_size=(224, 224))
    #img = image.load_img(img_path, target_size=(hw_height, hw_width))
    x = image.img_to_array(img)
    x = image.img_to_array(dst)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        # 自作勾配を登録するデコレーター
        # 今回は_GuidedBackProp関数を"GuidedBackProp"として登録
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            '''逆伝搬してきた勾配のうち、順伝搬/逆伝搬の値がマイナスのセルのみ0にして逆伝搬する'''
            dtype = op.inputs[0].dtype
            # grad : 逆伝搬してきた勾配
            # tf.cast(grad > 0., dtype) : gradが0以上のセルは1, 0以下のセルは0の行列
            # tf.cast(op.inputs[0] > 0., dtype) : 入力のが0以上のセルは1, 0以下のセルは0の行列
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    '''指定レイヤーのチャンネル方向最大値に対する入力の勾配を計算する関数の作成'''
    input_img = model.input # モデルのインプット
    # 入力層の次の層以降をレイヤー名とインスタンスの辞書として保持
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    # 引数で指定したレイヤー名のインスタンスの出力を取得 shape=(?, 14, 14, 512)
    layer_output = layer_dict[activation_layer].output
    # チャンネル方向に最大値を取る shape=(?, 14, 14)
    max_output = K.max(layer_output, axis=3)
    # 指定レイヤーのチャンネル方向最大値に対する入力の勾配を計算する関数
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    ''' ReLU関数の勾配を"name"勾配に置き換える'''
    # with内のReLUは"name"に置き換えられる
    g = tf.get_default_graph()
#    g = tf.compat.v1.get_default_graph
    with g.gradient_override_map({'Relu': name}):

        # ▽▽▽▽▽ 疑問4 : 新規モデルをreturnしているのに、引数のモデルのreluの置き換えが必要なのか? ▽▽▽▽▽

        # activationを持っているレイヤーのみ抜き出して配列化
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # kerasのRelUをtensorflowのReLUに置き換え
        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # △△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△

        # 新しくモデルをインスタンス化
        # 自作モデルを使用する場合はこちらを修正
        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):
    '''
    Parameters
    ----------
    input_model : model
        評価するKerasモデル
    image : tuple等
        入力画像(枚数, 縦, 横, チャンネル)
    category_index : int
        入力画像の分類クラス
    nb_classes = 2
    layer_name : str
        最後のconv層の後のactivation層のレイヤー名.
        最後のconv層でactivationを指定していればconv層のレイヤー名.
        batch_normalizationを使う際などのようなconv層でactivationを指定していない場合は、
        そのあとのactivation層のレイヤー名.

    Returns
    ----------
    cam : tuple
        Grad-Camの画像
    heatmap : tuple
        ヒートマップ画像
    '''
    # 分類クラス数
    #nb_classes = 1000
    #nb_classes = 2
    nb_classes = classes

    # ----- 1. 入力画像の予測クラスを計算 -----

    # 入力のcategory_indexが予想クラス

    # ----- 2. 予測クラスのLossを計算 -----

    # 入力データxのcategory_indexで指定したインデックス以外を0にする処理の定義
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    # 引数のinput_modelの出力層の後にtarget_layerレイヤーを追加
    # modelのpredictをすると予測クラス以外の値は0になる
    #x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)  # ここらへん3行が変更されてる
    #model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    # Original
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)

    # 予測クラス以外の値は0なのでsumをとって予測クラスの値のみ抽出
    loss = K.sum(model.output)
    # 引数のlayer_nameのレイヤー(最後のconv層)のoutputを取得する
    #print("model.layer:")
    #print(model.layers)
    #print("model.layers[0].layers:")
    #print(model.layers[0].layers)
    #print("[l for l in model.layers[0].layers]):")
    #print([l for l in model.layers[0].layers])
    #print("layer_name:")
    #print(layer_name)
    #print("[l.name for l in model.layers[0].layers]:")
    #print([l.name for l in model.layers[0].layers])
    #conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output

    # ----- 3. 予測クラスのLossから最後のconv層への逆伝搬(勾配)を計算 -----

    # 予想クラスの値から最後のconv層までの勾配を計算する関数を定義
    # 定義した関数の
    # 入力 : [判定したい画像.shape=(1, 224, 224, 3)]、
    # 出力 : [最後のconv層の出力値.shape=(1, 14, 14, 512), 予想クラスの値から最後のconv層までの勾配.shape=(1, 14, 14, 512)]
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    # 定義した勾配計算用の関数で計算し、データの次元を整形
    # 整形後
    # output.shape=(14, 14, 512), grad_val.shape=(14, 14, 512)
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # ----- 4. 最後のconv層のチャンネル毎に勾配を平均を計算して、各チャンネルの重要度(重み)とする -----

    #weights.shape=(512, )
    #cam.shape=(14, 14)
    # ※疑問点1：camの初期化はzerosでなくて良いのか?
    weights = np.mean(grads_val, axis = (0, 1))
    #cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)    # 別の作者の自作モデルではこちらを使用

    # ----- 5. 最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせて、ReLUを通す -----

    # 最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせ
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (hw_height, hw_width))   # 入力画像のサイズにリサイズ(14, 14) → (224, 224)
    cam = np.maximum(cam, 0)        # 負の値を0に置換。処理としてはReLUと同じ。
    # 値を0~1に正規化。
    # ※疑問2 : (cam - np.min(cam))/(np.max(cam) - np.min(cam))でなくて良いのか?
    #heatmap = cam / np.max(cam)
    print(cam)
    print(np.min(cam))
    print(np.max(cam))
    heatmap = (cam - np.min(cam))/(np.max(cam) - np.min(cam))    # 別の作者の自作モデルではこちらを使用

    # ----- 6. 入力画像とheatmapをかける -----

    # 入力画像imageの値を0~255に正規化. image.shape=(1, 224, 224, 3) → (224, 224, 3)
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    # ※疑問3 : np.uint8(image / np.max(image))でなくても良いのか?
    image = np.minimum(image, 255)

    # heatmapの値を0~255にしてカラーマップ化(3チャンネル化)
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)   # 入力画像とheatmapの足し合わせ
    cam = 255 * cam / np.max(cam)   # 値を0~255に正規化
    return np.uint8(cam), heatmap

# ① 入力画像の読み込み
# 入力画像を変換する場合はこちらを変更
#preprocessed_input = load_image(sys.argv[1])
#preprocessed_input = load_image("img_0691_21.jpg")
preprocessed_input = load_image(path_img)

# ② モデルの読み込み
# 自作モデルを使用する場合はこちらを変更
#model = VGG16(weights='imagenet')
modelname_text = open("model.json").read()
json_strings = modelname_text.split('##########')
textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
model = model_from_json(json_strings[0])
model.load_weights("last.hdf5")  # best.hdf5 で損失最小のパラメータを使用

# ③ 入力画像の予測確率(predictions)と予測ｸﾗｽ(predicted_class)の計算
# VGG16以外のモデルを使用する際はtop_1=~から3行はコメントアウト
predictions = model.predict(preprocessed_input)
#top_1 = decode_predictions(predictions)[0][0]
#print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
print("判定結果" + str(predicted_class))
# ④ Grad-Camの計算
# 自作モデルの場合、引数の"block5_conv3"を自作モデルの最終conv層のレイヤー名に変更.
#cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "conv2d_3")
print(cam)
#cv2.imshow("cam", cam)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# ⑤ 画像の保存
# now = time.ctime()
now = datetime.datetime.now()   # 現在時刻の取得
# result_name = "gradcam_"+now+".jpg"
path_name, ext = os.path.splitext(os.path.basename(path_img))   # ファイル名の取得
result_name = path_name + "_gradcam_{0:%Y%m%d-%H%M%S}.jpg".format(now)
# cv2.imwrite("gradcam.jpg", cam)
cv2.imwrite(result_name, cam)

### Guided Grad-CAM #############
# ① GuidedBackPropagation用勾配の実装
register_gradient()
# ② ReLUの勾配計算をGuidedBackPropagationの勾配計算に変更
guided_model = modify_backprop(model, 'GuidedBackProp')
# ③ GaidedBackPropagation計算用の関数の定義
# 自作クラスを使う場合は、こちらの引数に最後のconv層のレイヤー名を追加で指定
saliency_fn = compile_saliency_function(guided_model)
# ④ GaidedBackPropagationの計算
saliency = saliency_fn([preprocessed_input, 0])
# ⑤ Guided Grad-CAMの計算
gradcam = saliency[0] * heatmap[..., np.newaxis]
# ⑥ 画像の保存
# now = time.ctime()
now = datetime.datetime.now()   # 現在時刻の取得
# result_name = "guided_gradcam_"+now+".jpg"
result_name = path_name + "_guided_gradcam_{0:%Y%m%d-%H%M%S}.jpg".format(now)
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
cv2.imwrite(result_name, deprocess_image(gradcam))

