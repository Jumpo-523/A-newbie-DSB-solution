

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


---

大学院（博士過程）を辞めてSIの仕事を1年半、また研究の道に戻りたいと思っている人の話

戻りたいと思っているにもかかわらず、また同じ様な失敗（下記参照）を繰り返しそうだったので、思い立って振り返りを書きました。

今までの経緯

1. 2015年経済学部卒：特にやりたいと思っていることもなく、とりあえず進学させてもらった。大学院合格後学内の経済学ワークショップに行く。
    - ワークショップで聞いた先生の研究が面白く、研究職に就きたいと思い始める
2. ひたすら経済学の基礎をたたき込み、専門分野の論文を読んだ。
3. 自分が修論を書く段階になって本当にやりたいことってなんだっけ？となった。
4. 博士過程に進学させてもらったが、どんどん気が滅入ってきてしまい、一旦就職することを決意した。
5. 大手SIerの会津支部に行き、（炎上案件を経験し）ある程度のITのスキルを身に付けた。エンジニアリングがとても好きになった。
6. データ分析を完全に諦めることが出来ず、また家庭の事情もあり、会津から実家のある東京に戻ってきた。

### どう失敗したか？
- 理解しきってないのに、理解したつもりになる。

        こういうことっぽいなーと自分の中で附に落ちてOKとして、発表の時に詰まってしまう。
        自分はとても緊張しいで発表が苦手なだけだと思っていた。問題はそこではなく、理解の段階にあるのかもしれないと今思う。
<!-- 【解決策】 文字起こしして、明日の自分に説明する  -->
- 本当に自分がその分野に興味があるのかわからなくなった。  
    「やりたい事がわからない学生なんて、海外留学した学生にもたくさんいるよ」と先生に仰って頂いた事があった。  
    ただ、自分の中でこの分野で何か新しいことを追求する気概がなくなってしまった。

知識の生産者としての自分の成果が出ず、苦しい時に耐えられるのか？


### 心に残っていること

- 「何かを変えたい、挑戦する時は、「実験」なんだと思えばいい。この決断（退学・就職）も実験だと思ってリスクテイクしていくのがいい。」
    - TAしていた先生がおっしゃってくれたお言葉 
- 「知識の消費者から生産者にならないといけない」
    - 修士と博士++の違いを意識させてくれたお言葉
- 4年間遊んだんだから１人前になるまで10年くらいの辛抱の時期はあるだろう。
    - 自分に完全に自信を失った時に恩師に言われたこと。

## これからどうしていくのか？

- 理解した気になっちゃう自分に対して、
    - 文字起こしして、明日の自分に説明する。発表時は特に声に出して徹底的に練習する。
- 興味ある分野の探し方
    - 実験的に興味のある事に携わっていくしかないのかなと思っています。
        今のところ強化学習と因果推論を勉強していきたい。
- アカデミアとの繋がりを模索する  
    コンサル・受託会社でデータサイエンスを多少齧ったが、アカデミアでの面白みには敵わなかった。（面白い研究を見つけた時の高揚感）
    - 興味のある分野で研究開発職に行き、学問に携わり続けるキャリアにしていきたい。
- 脳のスペックがあまりよくない自分を認める。
    - 自分ができることを行い、出来ないことを諦める。
        （ex：言語化せずに不安に押しつぶされそうになる時、言語化して問題を克服できる様になる次の手を考える。）  
        <!-- （ex：太ったことでいびきがひどくなったらしい。最近寝ても疲れが取れないと悩んでいたが、睡眠の質を高める為に痩せる必要があるかもしれない。）
        →次の手として、健康的な食事を心がける -->
    - 自分にできる事なんて限られているんだから、人にちゃんと話す。







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
