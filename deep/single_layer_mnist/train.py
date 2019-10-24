import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import datetime
import numpy as np


# ## データの読み込みとビジュアライズ
# データセットのインポート
fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# 学習用教師ラベルには、「Tシャツ」「靴」などのカテゴリ番号が入る
# 番号は、以下のクラス名リストのインデックスに対応
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ## データの前処理
# 0-1の範囲に入力データをスケーリングする
train_x = train_x / 255.0
test_x = test_x / 255.0

# ## 単層ネットワークの構築と学習
#  学習モデルの定義
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 入力層. 28x28の二次元配列を784の一次元配列にフォーマットする (何も学習していない)
    keras.layers.Dense(128, activation='relu'), # 中間全結合層. 活性化関数には勾配喪失問題に対応したReLUを使用
    keras.layers.Dense(10, activation='softmax'), # 出力層. 多分類のための出力設計としてsoftmaxを活性化関数に使用
])

# 学習手法の定義
model.compile(
    optimizer='adam', # 勾配を使ってどのように重みを更新するか
    loss='sparse_categorical_crossentropy', # 出力層の勾配をどのように計算するか (中間層の勾配は誤差逆伝搬法を使った自動微分で算出される)
    metrics=['accuracy'], # モデルの学習精度をどのような指標で計測するか
)

# 学習ログをtensorboardで表示するためのログを残す
log_dir = "logs/single_layer_mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練 - バッチ学習(ミニではない)
model.fit(train_x, train_y, epochs=5, callbacks=[tensorboard_callback])

# ## 単層ネットワークの評価
# テストデータを使った、汎化性能の精度評価
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

# ↑model.fitで出力されたログに記載のaccuracyと比較すると、テストデータの方が精度が悪い。つまり、汎化性のない過学習を起こしている
# ## 学習済みネットワークを使った予測

# softmaxの定義通り、10クラスに属する各確率を配列で返す
preds = model.predict(test_x)