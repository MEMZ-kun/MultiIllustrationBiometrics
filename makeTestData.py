import os
import csv
import numpy as np

# 保存先ディレクトリ
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# 論文の仕様に基づく設定値
# [cite_start]被験者数: 10名, 各100サンプル [cite: 6, 98]
NUM_SUBJECTS = 10
SAMPLES_PER_SUBJECT = 100
TOTAL_SAMPLES = NUM_SUBJECTS * SAMPLES_PER_SUBJECT

# [cite_start]パーツごとの最大シーケンス長（これに合わせてパディングを行うための基準長） [cite: 154]
SEQ_LEN_RIGHT_EYE = 338
SEQ_LEN_LEFT_EYE = 272
SEQ_LEN_NOSE = 511
SEQ_LEN_MOUTH = 355

# [cite_start]ファイル名の定義（論文の表記・誤記を厳密に再現） [cite: 105-151]
# キーは (パーツ名, データ種別)
# データ種別: x, y, x_acc, y_acc

FILES_TRAIN = {
    # トレーニングデータ (ドラえもん)
    # 右目 (light_eye)
    "right_eye": {
        "x": "dra light_eye_x.csv",
        "y": "dra light_eye_y.csv",
        "x_acc": "dra light_eye_x.acc.csv", # ここだけ .acc
        "y_acc": "dra light_eye_y_acc.csv"
    },
    # 左目 (left_eye)
    "left_eye": {
        "x": "dra left_eye_x.csv",
        "y": "dra left_eye_y.csv",
        "x_acc": "dra left_eye_x_acc.csv",
        "y_acc": "dra_left_eye_y_acc.csv"   # 先頭に _ あり
    },
    # 鼻 (nose)
    "nose": {
        "x": "dra nose_x.csv",
        "y": "dra_nose_y.csv",            # 先頭に _ あり
        "x_acc": "dra_nose_x_acc.csv",    # 先頭に _ あり
        "y_acc": "dra_nose_y_acc.csv"     # 先頭に _ あり
    },
    # 口 (mouse)
    "mouth": {
        "x": "dra mouse_x.csv",
        "y": "dra mouse_y.csv",
        "x_acc": "dra mouse_x_acc.csv",
        "y_acc": "dra_mouse_y_acc.csv"    # 先頭に _ あり
    }
}

FILES_TEST = {
    # テストデータ (アンパンマン)
    # 右目 (light_eye)
    "right_eye": {
        "x": "an light_eye_x.csv",
        "y": "an light_eye_y.csv",
        "x_acc": "an light_eye_x_acc.csv",
        "y_acc": "an light_eye_y_acc.csv"
    },
    # 左目 (left_eye)
    "left_eye": {
        "x": "an left_eye x.csv",         # スペース区切り
        "y": "an left eye_y.csv",         # スペース区切り
        "x_acc": "an left_eye_x_acc.csv",
        "y_acc": "an left_eye_y_acc.csv"
    },
    # 鼻 (nose)
    "nose": {
        "x": "an_nose x.csv",             # _ と スペース混在
        "y": "an_nose_y.csv",
        "x_acc": "an nose_x_acc.csv",     # スペース区切り
        "y_acc": "an_nose_y_acc.csv"
    },
    # 口 (mouse)
    "mouth": {
        "x": "an mouse x.csv",            # スペース区切り
        "y": "an_mouse_y.csv",
        "x_acc": "an mouse_x_acc.csv",    # スペース区切り
        "y_acc": "an mouse_y_acc.csv"     # スペース区切り
    }
}

def generate_dummy_data(filename, max_len):
    """
    論文のCSV形式仕様に従いダミーデータを生成する
    1列目: ラベル (1~10)
    2列目以降: データ (可変長だが、ここでは簡単のためmax_len分の乱数を入れる)
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    print(f"Generating {filename} (Max Len: {max_len})...")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 被験者10人分、各100サンプルを生成
        for subject_id in range(1, NUM_SUBJECTS + 1):
            for _ in range(SAMPLES_PER_SUBJECT):
                # [cite_start]1列目はラベル [cite: 83-84]
                row = [subject_id]
                
                # 2列目以降は時系列データ
                # 実際には可変長でパディングされるが、ダミーなので
                # ランダムな長さ(max_lenの50%~100%)のデータを作成し、残りを空にするか
                # あるいは単純にmax_len分のデータを作成する。
                # [cite_start]論文では「満たないデータはゼロパディングした」とあるため [cite: 154]
                # CSV自体には生の長さが入っていると想定し、少し短めに生成する
                actual_len = np.random.randint(int(max_len * 0.5), max_len + 1)
                data_values = np.random.rand(actual_len).tolist()
                
                # 行に書き込み (可変長のまま書き込むのがCSVの一般的な挙動)
                writer.writerow(row + data_values)

def main():
    # トレーニングデータの生成
    print("--- Generating Training Data (Doraemon) ---")
    for part, files in FILES_TRAIN.items():
        # パーツごとのシーケンス長を決定
        if part == "right_eye": seq_len = SEQ_LEN_RIGHT_EYE
        elif part == "left_eye": seq_len = SEQ_LEN_LEFT_EYE
        elif part == "nose": seq_len = SEQ_LEN_NOSE
        elif part == "mouth": seq_len = SEQ_LEN_MOUTH
        
        for file_type, filename in files.items():
            generate_dummy_data(filename, seq_len)

    # テストデータの生成
    print("\n--- Generating Test Data (Anpanman) ---")
    for part, files in FILES_TEST.items():
        if part == "right_eye": seq_len = SEQ_LEN_RIGHT_EYE
        elif part == "left_eye": seq_len = SEQ_LEN_LEFT_EYE
        elif part == "nose": seq_len = SEQ_LEN_NOSE
        elif part == "mouth": seq_len = SEQ_LEN_MOUTH
        
        for file_type, filename in files.items():
            generate_dummy_data(filename, seq_len)
            
    print("\nAll dummy files created in 'dataset/' directory.")

if __name__ == "__main__":
    main()