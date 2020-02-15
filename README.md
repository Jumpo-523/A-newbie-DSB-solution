

# dsb-2019


- 12/05
    - shapで何がどう効いているのか確認する。
    - 難しい。→自作関数に対して処理を実行できない。
    - simpleなlgbで実行して検証してみる。
- 12/06
    - ランダム性がseed averagingしても存在してしまう。
    - lightgbmに使うカラムが実行段階で変わってしまう？何をどんだけ使っていルカ。
    特徴量重要度を必ず取ってきて→標準偏差・平均を取りたい。
    - transformer過学習起こしうるからやめたい。importancetop層いない。（出現回数少ないからかも。）


- 1/1
    - level毎になんのゲームを経験したか（＝game_session_count）を特徴量に加える。
    - 望ましいルートか否かの指標の追加
    [TBD]
    0 or Otherの識別はうまく行っている印象がある。
    単純なイージーミスが多いか否かの識別をどうするべきか？
        特徴量の選定をすべき
        countではなく、割合もしくはtime
    - 4●系のactivityをちゃんとやっているか？（event_dataの詳細を追う）
        - 4000系であればなんでもいい
        - session_count数ではなく、割合に変える必要がある。
- 1/3
    - stackingするタイミングをcross validation内部にセット
    - n_seeds = 10でセットしてやっているが、同じことを10回ループしているだけになっている。
    - Lightgbmのrandom_seed引数について調べる。
    - <font color="red">confusion_matrixを出力する</font>
        できていない
    - lightgbmの挙動がkernel環境で異なる。
        - versionの違いだろう：欠損したカラムがあるorカラムネームに空白文字があるケースを許容しなくなったと思われる
    - 説明をちゃんと聞く子か否かは取れるか？

- 1/5 division by total_countをやるべきでは？ (discussionから)



- 1/22
    - test dataがランダムに取られていたのか。
    - https://www.kaggle.com/c/data-science-bowl-2019/discussion/126395#721312



- 反省点
    - distributionをチェックして、標準化すべきかいなかなどの考察を怠った（カーネルで判断してしまった。）
    - インターネット契約していない家なので、平日は何にもできないという不利さはあった。

What I learnt from this competition.

- groupby Obj 's transformer method
- HTTPErrorの処理
    - urllib.requests.HTTPERRORだとキャッチできない。
    - request.exceptions.HTTPERROR
    - job's APIが利用しているモジュールに合わせないとキャッチできない。




```python

conf_string = '''
dataset:
  dir: "../input/data-science-bowl-2019/"
  feature_dir: "features"
  params:

features:
  - PastSummary3
  - NakamaV8

av:
  split_params:
    n_splits: 5
    random_state: 42

  model_params:
    objective: "binary"
    metric: "auc"
    boosting: "gbdt"
    max_depth: 7
    num_leaves: 75
    learning_rate: 0.01
    colsample_bytree: 0.7
    subsample: 0.1
    subsample_freq: 1
    seed: 111
    feature_fraction_seed: 111
    drop_seed: 111
    verbose: -1
    n_jobs: -1
    first_metric_only: True

  train_params:
    num_boost_round: 50000
    early_stopping_rounds: 200
    verbose_eval: 200

model:
  name: "mlp"
  mode: "ovr"
  save_path: "pth/"
  policy: "best_score"

  model_params:
    emb_drop: 0.3
    drop: 0.5

  train_params:
    batch_size: 256
    n_epochs: 50
    lr: 0.001
    scheduler:
      name: "cosine"
      T_max: 10
      eta_min: 0.00001

post_process:
  params:
    reverse: False
    n_overall: 20
    n_classwise: 20

val:
  name: "group_kfold"
  n_delete: 0.9
  percentile: 60
  params:
    n_splits: 5
    random_state: 111

output_dir: "output"
'''
```

- ```remove_correlated_features```
    is it enough to see train without test data in order to feature selection?
# useful tips for data competitions

- this repository will not show my solution until the competition is completed. 


What I learnt.
---
(Technical ones, which might show how stupid I am.)
1. groupby can be iterated.
2. Things can be looped like as follows 
  ```python
  for i, (j, group) in enumerate(df.groupby(xxx)):
   ```
   If you want to loop the groupby object, then you might be faced with tuple composed of (index,pd.DataFrame)
