# Creative_factory_bit_2022




##心音の主成分と心雑音のHz
https://www.jstage.jst.go.jp/article/jjsem/17/1/17_39/_pdf



# 概要

このプロダクトは，保育園向けの連絡帳機能を実装したWEBアアプリケーションです．
フロントエンドにはFlutter，バックエンドにはDjango（Django Rest Framework）を使用しています．

# 環境構築

## ローカルでの環境構築

1. Cloneする
2. pipenv install
2. postgresqlを入れる
   - Macの人は「brew install postgresql」
   - Windowsの人は調査中
3. おわり

## デプロイ or リモートで開発する人

herokuとGithubアカウントをリンク
おわり
コードを書き換えたらDeply

## データベースの切り替え

ローカルDB(db.sqlite3ファイル)とリモートDB(heroku Postgres)の切り替えが可能．
ローカルDBのメリットは設定がいらないこと，リモートDBの利点はDeploy時にmigrate関連でエラーが出なくなること

### 方法

リモートDBのURIをローカルの環境変数DATABASE_URLに設定する．

リモートURIの取得方法は以下の2通り

- ターミナルから`heroku config:get -a {HEROKU_APP_NAME} DATABASE_URL`を実行する
- [herokuのDBのページ](https://data.heroku.com)からDBを選び，Settings→View  Credentialsで表示

環境変数の設定はWindowsとmacOSで異なる

macOSの場合

```bash
export DATABASE_URL={さっきコピーしたURI}
```

リモートDBのURIは突然変わる可能性があるため気をつけること，環境変数もログアウトや再起動などで消える場合があるため確認すること

### リモートDBの構成図

![zu](https://user-images.githubusercontent.com/40960166/186350100-39da9775-e408-4c52-ab66-d114e6df16bc.jpg)

# API

## 概要

保育園向け連絡帳アプリの開発者向け API （さむいなまこ API ）の仕様を説明します．
全てのデータはJSON形式で送受信が行われます．

## パスで要求されるパラメーター

| パラメーター | 概要 | 生成方法 | 
| :---------: | :---: | :------------------ |
| `child_id` | 各園児に振られる固有のID | adminから追加 |
| `staff_id` | 各保育士に振られる固有のID | adminから追加 |

## さむいなまこ API を使って出来ること

| No. | 画面 | 出来ること | メソッド | URI | 
| :--: | :---: |:--- | :---------------: | :------------------ |
| 1 | 保護者用| [指定した child_id に対応する連絡帳の情報を返す](#1-指定した-child_id-に対応する連絡帳の情報を返す)     | GET | /child/{child_id}/ |
| 2 | 保護者用| [任意の連絡帳に記入されたメッセージを保存する](#2-任意の連絡帳に記入されたメッセージを保存する) | POST | /child/{child_id}/write/  |
| 3 | 保育士用| [園児一覧を返す](#3-園児一覧を返す)   | GET | /staff/{staff_id}/ |
| 4 | 保育士用| [指定した child_id に対応する連絡帳の情報を返す](#4-指定した-child_id-に対応する連絡帳の情報を返す) | GET | /staff/{staff_id}/{child_id}/ |
| 5 | 保育士用| [任意の連絡帳のメッセージに対する返信を保存する](#5-任意の連絡帳のメッセージに対する返信を保存する) | POST | /staff/{staff_id}/{child_id}/reply/ |

### 保護者用

#### 1. 指定した child_id に対応する連絡帳の情報を返す

今日の連絡帳があったらそれを開く，なかったら新規作成される．
child_idがなかったら404エラーが返される．

##### エンドポイント

```URL
GET /child/{child_id}/
```

##### パラメーター

`{child_id}`

各園児の連絡帳が開かれる

##### 返却データ（JSON形式）

| JSON Key | 型 | サイズ | デフォルト値 | 値の説明 |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|
| `note_id` | 文字列 | - | 自動生成 | 連絡帳のID |
| `child_id` | 文字列 | - | - | 園児のID |
| `child_name` | 文字列 | 50 | - | 園児の名前|
| `date` | 文字列 | 500 | - | 保護者が連絡帳を記入した日時 |
| `message_parent` | 文字列 | 500 | - | 保護者からのメッセージ |
| `message_staff` | 文字列 | 500 | - | 保育士からの返信 |
| `staff_id` | 文字列 | - | - | 返信した保育士のID |
| `staff_name` | 文字列 | 50 | - | 返信した保育士の名前|
| `body_templeture` | 文字列 | - | 前回の入力値 | 園児の体温 |
| `pickup_time` | 文字列 | - | 前回の入力値 | お迎えの時間 |
| `pickup_person` | 文字列 | 50 | 前回の入力値 | お迎えの方 |
| `write_flag` | 真偽値 | - | False | 提出がされているか否か |
| `reply_flag` | 真偽値 | - | False | 返信がされているか否か |

##### 応答の例

```JSON
{
  "note_id": "10",
  "child_id": "1",
  "child_name": "豊洲太郎",
  "date": "2022-08-25",
  "message_parent": "朝ご飯を食べるのを嫌がりました，あまり多く食べていないのでお昼頃に機嫌が悪くなりそうです",
  "message_staff": "",
  "staff_id": "None",
  "staff_name": "None",
  "body_temperature": "36.5",
  "pickup_time": "2022-01-01T18:30:00",
  "pickup_person": "父",
  "write_flag": True,
  "replay_flag": False
}
```

#### 2. 任意の連絡帳に記入されたメッセージを保存する

保護者が連絡帳に記入した文字列をデータベースに保存する．

##### エンドポイント

```URL
POST /child/{child_id}/write/
```

##### パラメーター

`{child_id}`

`{child_id}`のデータが更新される

##### POSTデータ（JSON形式）

変更がない部分は，[事前にGETしたデータ](#1-指定した-child_id-に対応する連絡帳の情報を返す)を使用する．
すべてのキーは必須である．

| JSON Key | 型 | サイズ | デフォルト値 | 値の説明 |
|:-----------:|:-----------:|:----------:|:-----------:|:-----------|
| `note_id` | 文字列 | - | 自動生成 | 連絡帳のID |
| `message_parent` | 文字列 | 500 | - | 保護者からのメッセージ |
| `body_templeture` | 文字列 | - | 前回の入力値 | 園児の体温 |
| `pickup_time` | 文字列 | - | 前回の入力値 | お迎えの時間 |
| `pickup_person` | 文字列 | 50 | - | お迎えの方 |

##### POSTの例

```JSON
{
  "note_id": "10",
  "message_parent": "急に熱が出たのでお休みします",
  "body_temperature": "38.2",
  "pickup_time": "2022-01-01T18:30:00",
  "pickup_person": "父"
}
```

### 保育士用

#### 3. 園児一覧を返す

園児の一覧を配列形式で返す．

##### エンドポイント

```URL
GET /staff/{staff_id}/
```

##### パラメーター

`{staff_id}`

登録されているstaff_idであれば，その値によらず全ての園児の情報を返す．

##### 返却データ（JSON形式）

| JSON Key | 型 | サイズ | デフォルト値 | 値の説明 |
|:----------:|:-----------:|:-----------:|:-----------:|:-----------|
| `child_id` | 文字列 | - | - | 園児のID |
| `child_name` | 文字列 | 50 | - | 園児の名前 |
| `parent_name` | 文字列 | 50 | - | 親の名前 |
| `reply_flag` | 真偽値 | - | False | 返信がされているか否か|

##### 応答の例

```JSON
[
  {
    "child_id": "1",
    "name": "豊洲太郎",
    "parent_name": "豊洲貴子",
    "reply_flag" : True
  },
  {
    "child_id": "2",
    "name": "有楽町ミカ",
    "parent_name": "有楽町英二",
    "reply_flag" : True
  },
  {
    "child_id": "3",
    "name": "月島日向",
    "parent_name": "月島昭代",
    "reply_flag" : False
  }
]

```


#### 4. 指定した child_id に対応する連絡帳の情報を返す

##### エンドポイント

```URL
GET  /staff/{staff_id}/{child_id}/
```

##### パラメーター

`{staff_id}`, `{child_id}`

##### 返却データ（JSON形式）

| JSON Key | 型 | サイズ | デフォルト値 | 値の説明 |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|
| `note_id` | 文字列 | - | 自動生成 | 連絡帳のID |
| `child_id` | 文字列 | - | - | 園児のID |
| `child_name` | 文字列 | 50 | - | 園児の名前|
| `date` | 文字列 | - | - | 保護者が連絡帳を記入した日時 |
| `message_parent` | 文字列 | 500 | - | 保護者からのメッセージ |
| `message_staff` | 文字列 | 500 | - | 保育士からの返信 |
| `staff_id` | 文字列 | - | - | 返信した保育士のID |
| `staff_name` | 文字列 | 50 | - | 返信した保育士の名前|
| `body_templeture` | 文字列 | - | 前回の入力値 | 園児の体温 |
| `pickup_time` | 文字列 | - | 前回の入力値 | お迎えの時間 |
| `pickup_person` | 文字列 | 50 | - | お迎えの方 |

##### 応答の例

```JSON
{
  "note_id": "10",
  "child_id": "1",
  "child_name": "豊洲太郎",
  "date": "2022-08-25",
  "message_parent": "朝ご飯を食べるのを嫌がりました，あまり多く食べていないのでお昼頃に機嫌が悪くなりそうです",
  "message_staff": "",
  "staff_id": "None",
  "staff_name": "None",
  "body_temperature": "36.5",
  "pickup_time": "2022-01-01T18:30:00",
  "pickup_person": "父"
}
```

#### 5. 任意の連絡帳のメッセージに対する返信を保存する

保育士が連絡帳に記入した文字列をデータベースに保存する．

##### エンドポイント

```URL
 POST /staff/{staff_id}/{child_id}/reply/
```

##### パラメーター

`{staff_id}`, `{child_id}`

##### POSTデータ（JSON形式）

| JSON Key | 型 | サイズ | デフォルト値 | 値の説明 |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|
| `note_id` | 文字列 | - | 自動生成 | 連絡帳のID |
| `message_staff` | 文字列 | 500 | - | 保育士のからの返信 |


##### POSTの例

```JSON
{
  "note_id": "10",
  "message_staff": "かしこまりました，お昼前に気をつけます"
}
```
