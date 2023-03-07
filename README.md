```
├── .gitignore         <- datasetフォルダやmodelsフォルダを追跡したくない場合記述．
├── LICENSE
├── Makefile           <- `make data` や `make train`のようなコマンドをここに記述．
├── README.md          <- このプロジェクトの説明
├── dataset
│   ├── external       <- 第三者が作成したデータ
│   ├── interim        <- 何かしらの変換をした中間データ
│   ├── processed      <- モデルに入力する最終的なデータ
│   └── raw            <- オリジナルのデータ
│
├── docs               <- デフォルトのSphinxプロジェクト．詳細:sphinx-doc.org
│
├── models             <- 訓練済みのモデルを格納．
│
├── notebooks          <- 分析用のジュピターノートブックを格納．
│                       
│
├── references         <- 参考にした資料を格納．
│
├── reports            <- 論文などを格納．
│   └── figures        <- 論文に使用した画像を格納．
│
├── requirements.txt   <- 環境を再現するためのファイル．
│
├── src                <- 本プロジェクトで用いるソースコード
│   ├── __init__.py    <- srcをPythonモジュールとするためのファイル．
│   │
│   ├── data           <- データのダウンロードや生成のためのファイルを格納するフォルダ．
│   │   └── make_dataset.py
│   │
│   ├── features       <- 生データを特徴量に変換するファイルを格納するフォルダ．
│   │   └── build_features.py
│   │
│   ├── models         <- モデルを学習させるファイルを格納するフォルダ．            
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- 実験結果の可視化用コード
│       └── visualize.py
│
├── tests              <- テストコードを格納するフォルダ．
│
└── tox.ini            <- 設定を記述するファイル．詳細:tox.testrun.or
```