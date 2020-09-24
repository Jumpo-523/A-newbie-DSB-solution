

# dsb-2019


[kaggle のDSBコンペ](https://www.kaggle.com/c/data-science-bowl-2019/)にて利用したコード群が格納されるrepository。

- 銀圏５９位に入賞しました。


# 当コンペに関する分析日記(完全に個人的なメモです。。)


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
    - 各特徴量でPDCAを回すことを怠った。
    - ちゃんとPipelineを作成して分析を効率化すべきだった。
    
<!--What I learnt from this competition.
- groupby Obj 's transformer method
- HTTPErrorの処理
    - urllib.requests.HTTPERRORだとキャッチできない。
    - request.exceptions.HTTPERROR
    - job's APIが利用しているモジュールに合わせないとキャッチできない。
-->



