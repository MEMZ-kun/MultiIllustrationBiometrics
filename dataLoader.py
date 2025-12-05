import os
import csv
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataLoader:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        
        # 論文で定義された厳密な最大シーケンス長 
        self.SEQ_LEN_MAP = {
            "right_eye": 338,
            "left_eye": 272,
            "nose": 511,
            "mouth": 355
        }

        # ステップ1と同じファイル名定義（論文の表記・誤記を再現） [cite: 105-151]
        self.FILES_TRAIN = {
            "right_eye": {"x": "dra light_eye_x.csv", "y": "dra light_eye_y.csv", "x_acc": "dra light_eye_x.acc.csv", "y_acc": "dra light_eye_y_acc.csv"},
            "left_eye": {"x": "dra left_eye_x.csv", "y": "dra left_eye_y.csv", "x_acc": "dra left_eye_x_acc.csv", "y_acc": "dra_left_eye_y_acc.csv"},
            "nose": {"x": "dra nose_x.csv", "y": "dra_nose_y.csv", "x_acc": "dra_nose_x_acc.csv", "y_acc": "dra_nose_y_acc.csv"},
            "mouth": {"x": "dra mouse_x.csv", "y": "dra mouse_y.csv", "x_acc": "dra mouse_x_acc.csv", "y_acc": "dra_mouse_y_acc.csv"}
        }

        self.FILES_TEST = {
            "right_eye": {"x": "an light_eye_x.csv", "y": "an light_eye_y.csv", "x_acc": "an light_eye_x_acc.csv", "y_acc": "an light_eye_y_acc.csv"},
            "left_eye": {"x": "an left_eye x.csv", "y": "an left eye_y.csv", "x_acc": "an left_eye_x_acc.csv", "y_acc": "an left_eye_y_acc.csv"},
            "nose": {"x": "an_nose x.csv", "y": "an_nose_y.csv", "x_acc": "an nose_x_acc.csv", "y_acc": "an_nose_y_acc.csv"},
            "mouth": {"x": "an mouse x.csv", "y": "an_mouse_y.csv", "x_acc": "an mouse_x_acc.csv", "y_acc": "an mouse_y_acc.csv"}
        }

    def _read_csv(self, filename):
        """
        1つのCSVファイルを読み込み、ラベルとデータリストを返す。
        1列目: ラベル
        2列目以降: 時系列データ
        """
        filepath = os.path.join(self.data_dir, filename)
        labels = []
        data_list = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # 1列目はラベル [cite: 83]
                labels.append(int(row[0]))
                # 2列目以降は数値データ
                data_values = [float(x) for x in row[1:] if x != '']
                data_list.append(data_values)
        
        return np.array(labels), data_list

    def load_part_data(self, part_name, is_train=True, target_subject_id=1):
        """
        指定されたパーツのデータを読み込み、前処理を行う。
        
        Args:
            part_name (str): 'right_eye', 'left_eye', 'nose', 'mouth'
            is_train (bool): Trueならトレーニングデータ(ドラえもん), Falseならテストデータ(アンパンマン)
            target_subject_id (int): 本人確認対象の被験者ID (1~10)。これ以外はラベル0になる。
        
        Returns:
            X (np.array): (サンプル数, シーケンス長, 4) の形状。4は特徴量(x, y, x_acc, y_acc)
            y (np.array): (サンプル数, 1) の形状。2クラス分類用ラベル(0 or 1)
        """
        
        # ファイル定義の選択
        file_map = self.FILES_TRAIN[part_name] if is_train else self.FILES_TEST[part_name]
        max_len = self.SEQ_LEN_MAP[part_name]

        # 4つの特徴量ファイルをそれぞれ読み込む
        print(f"Loading {part_name} ({'Train' if is_train else 'Test'})...")
        
        # x座標
        labels_x, data_x = self._read_csv(file_map["x"])
        # y座標
        _, data_y = self._read_csv(file_map["y"])
        # x加速度
        _, data_x_acc = self._read_csv(file_map["x_acc"])
        # y加速度
        _, data_y_acc = self._read_csv(file_map["y_acc"])

        # サンプル数が一致しているか確認
        assert len(data_x) == len(data_y) == len(data_x_acc) == len(data_y_acc), "Sample count mismatch across files"

        # データの結合とパディング処理
        num_samples = len(data_x)
        
        # 最終的な格納用配列: (サンプル数, max_len, 4)
        # 初期値は0.0 (パディング値)
        X = np.zeros((num_samples, max_len, 4), dtype=np.float32)

        for i in range(num_samples):
            # 各特徴量のシーケンス長を取得（すべて同じ長さであるべきだが、安全のため最小長を取るか、index合わせる）
            # ここではダミー生成時に揃っている前提だが、念のため長さを揃える処理
            current_len = min(len(data_x[i]), len(data_y[i]), len(data_x_acc[i]), len(data_y_acc[i]), max_len)
            
            # データを埋める (不足分は初期値0.0のまま=ゼロパディング)
            # 論文: "満たないデータはゼロパディングした" 
            # パディング位置についての明記はないが、時系列なら通常「後ろ」を埋める (post-padding)
            X[i, :current_len, 0] = data_x[i][:current_len]
            X[i, :current_len, 1] = data_y[i][:current_len]
            X[i, :current_len, 2] = data_x_acc[i][:current_len]
            X[i, :current_len, 3] = data_y_acc[i][:current_len]

        # ラベルの2クラス分類化 (本人確認タスク) 
        # target_subject_id (本人) -> 1, それ以外 -> 0
        y = np.where(labels_x == target_subject_id, 1, 0)
        y = y.reshape(-1, 1) # (サンプル数, 1)

        print(f"  Shape: X={X.shape}, y={y.shape}")
        return X, y

# 動作確認用
if __name__ == "__main__":
    loader = DataLoader()
    # テストとして右目のデータを読み込んでみる
    X_train, y_train = loader.load_part_data("right_eye", is_train=True, target_subject_id=9)