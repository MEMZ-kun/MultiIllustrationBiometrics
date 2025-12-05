import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import random
import os

# 既存のDataLoaderクラスをインポート
from dataLoader import DataLoader

# --- ハイパーパラメータ設定 (論文準拠) ---
# バッチサイズ: 10 [cite: 117]
BATCH_SIZE = 10
# エポック数: 100 [cite: 117]
EPOCHS = 100
# 検証データの割合: 10% [cite: 117]
VALIDATION_SPLIT = 0.1
# 学習率: 0.01 [cite: 117]
LEARNING_RATE = 0.01
# 再現性のためのシード値 
SEED = 42

def set_seed(seed):
    """
    モデルを再現できるように乱数シードを固定値で初期化 
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_model(input_shape):
    """
    3.2節 機械学習 の仕様に基づくモデル構築 [cite: 114-117]
    """
    model = Sequential()

    # (1) マスキング層: mask_value=0.0 [cite: 115]
    # mask_zero=trueオプションを使用してマスキング処理 [cite: 117]
    model.add(Masking(mask_value=0.0, input_shape=input_shape))

    # (2) LSTM層: units=128, return_sequences=False [cite: 116]
    model.add(LSTM(128, return_sequences=False))

    # (3) 全結合層: units=1 [cite: 117]
    # 2クラス分類のためSigmoid活性化関数を使用 (論文の文脈的補完)
    model.add(Dense(1, activation='sigmoid'))

    # トレーニング率は0.01 [cite: 117]
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    # 追加実験で2クラス分類を実施 [cite: 212-213]
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    return model

def evaluate_part(part_name, target_subject_id):
    """
    パーツごとに異なるモデルを用いて分類精度を算出する 
    """
    print(f"\n========== Evaluating Part: {part_name} ==========")
    
    # 各パーツの実験前にシードをリセットして条件を統一 
    set_seed(SEED)
    
    loader = DataLoader()
    
    # データの読み込み
    # トレーニングデータ: ドラえもん [cite: 124]
    X_train, y_train = loader.load_part_data(
        part_name=part_name, 
        is_train=True, 
        target_subject_id=target_subject_id
    )
    
    # テストデータ: アンパンマン [cite: 124]
    X_test, y_test = loader.load_part_data(
        part_name=part_name, 
        is_train=False, 
        target_subject_id=target_subject_id
    )
    
    # モデル構築
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # 学習実行 [cite: 117]
    # verbose=0 にしてログ出力を抑制（進捗バーを出したい場合は1に変更）
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=0
    )
    
    # 評価 [cite: 119]
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  -> Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # 論文の追加実験で高精度が出た「被験者9」を対象とする 
    TARGET_SUBJECT = 9
    
    # 評価対象の全パーツ [cite: 36, 121]
    # ファイル名の定義順序に従いリスト化
    parts_list = ["right_eye", "left_eye", "nose", "mouth"]
    
    results = {}
    
    print(f"Starting 2-Class Verification for Subject {TARGET_SUBJECT}...")
    print("Hyperparameters:")
    print(f"  Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    
    # パーツごとのループ実行
    for part in parts_list:
        acc = evaluate_part(part, TARGET_SUBJECT)
        results[part] = acc
        
    # 最終結果の表示 (論文の表1や図1に対応する形式)
    print(f"\n\n################ FINAL RESULTS (Subject {TARGET_SUBJECT}) ################")
    print(f"{'Part':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for part, acc in results.items():
        print(f"{part:<15} | {acc * 100:.2f}%")
    
    # 参考: 論文記載の被験者9(左目)の正解率は 92.2% 
    print("-" * 30)
    print("Reference (Paper): Subject 9 Left Eye = 92.2%")
    print("############################################################")

if __name__ == "__main__":
    main()