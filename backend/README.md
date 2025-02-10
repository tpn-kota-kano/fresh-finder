# FreshFinder backend

このプロジェクトは、Google の Vertex AI と Vision API を使用して画像内のオブジェクトを検出、分類、ランク付けする AI 画像解析 API を提供します。特に野菜や肉などの製品タイプに焦点を当てています。

アプリケーションは FastAPI を使用して構築されており、Google Cloud の AI サービスを活用して高度な画像解析タスクを実行します。アップロードされた画像を処理し、画像内のオブジェクトを検出し、バウンディングボックスとランキングを含む処理済み画像を返すように設計されています。

## リポジトリ構造

```
.
├── Dockerfile
├── poetry.lock
├── pyproject.toml
└── src
    ├── entrypoint.py
    ├── modules
    │   ├── image_analyser_gemini.py
    │   └── image_analyser_visionai.py
    └── prompts
        ├── gemini
        │   ├── meat.txt
        │   └── vegetables.txt
        └── visionapi
            ├── meat.txt
            └── vegetables.txt
```

- `Dockerfile`: アプリケーションのコンテナイメージを定義します。
- `pyproject.toml`: プロジェクトの依存関係と設定を指定します。
- `src/entrypoint.py`: メインの FastAPI アプリケーションのエントリーポイントです。
- `src/modules/`: コア画像解析モジュールが含まれています。
- `src/prompts/`: 異なる製品タイプのプロンプトテンプレートを保存します。

## 使用方法

### インストール

前提条件:

- Python 3.12
- Docker（オプション、コンテナ化されたデプロイメント用）

プロジェクトをローカルにセットアップするには:

1. リポジトリをクローンします:

   ```
   git clone <repository-url>
   cd <repository-directory>/backend
   ```

2. Poetryとpoethepoetをインストールします：

   ```
   # https://python-poetry.org/docs/#installing-with-the-official-installer
   curl -sSL https://install.python-poetry.org | python3 -

   # https://poethepoet.natn.io/installation.html
   poetry self add 'poethepoet[poetry_plugin]'
   ```

3. Poetry を使用して依存関係をインストールします:

   ```
   poetry install
   ```

4. Google Cloud の認証情報を設定します:
   - Vertex AI と Vision API が有効になっている Google Cloud プロジェクトを持っていることを確認します。
   - `gcloud auth application-default login`でローカルの認証情報を設定します。

5. Google Cloud プロジェクトIDを環境変数に設定します：
   - `export PROJECT_ID={project_id}`でGoogle Cloud プロジェクトIDを設定します

### アプリケーションの実行

開発サーバーを実行するには：

```
poetry poe dev
```

### API の使用

画像解析のメインエンドポイントは次のとおりです:

```
POST /genai-image-analysis
```

パラメータ:

- `image`: 解析する画像ファイル (multipart/form-data)
- `productType`: 画像内の製品タイプ (例: "vegetables", "meat")
- `desiredInfo`: 解析に関する追加情報

cURL リクエストの例:

```bash
curl -X POST "http://localhost:8000/genai-image-analysis" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@/path/to/image.jpg" \
     -F "productType=vegetables" \
     -F "desiredInfo=freshness"
```

API はバウンディングボックスとランキングを含む base64 エンコードされた処理済み画像を JSON レスポンスとして返します。

### トラブルシューティング

1. 問題: Google Cloud に認証できない

   - ローカルの認証情報が設定されていることを確認します。
   - アカウントに必要な権限があることを確認します。

2. 問題: 画像処理が失敗する

   - 入力画像がサポートされている形式（JPEG、PNG）であることを確認します。
   - 画像ファイルが破損していないことを確認します。

3. 問題: API が 500 Internal Server Error を返す
   - 詳細なエラーメッセージについてアプリケーションログを確認します。
   - 必要なすべての Google Cloud API が有効になっていることを確認します。

デバッグのために:

- `IS_PROD`環境変数を`0`に設定してデバッグログを有効にします。
- 各処理ステップの詳細情報についてログを確認します。

## データフロー

画像解析プロセスは次のステップに従います:

1. クライアントが`/genai-image-analysis`エンドポイントに画像をアップロードします。
2. FastAPI アプリケーションが画像とパラメータを受け取ります。
3. `image_analyser_gemini.py`モジュールが画像を処理します:
   a. Vertex AI の生成モデルを使用してオブジェクトを検出します。
   b. 最も一般的なカテゴリに焦点を当てるためにオブジェクトをフィルタリングします。
   c. 検出されたオブジェクトのために画像にバウンディングボックスを描画します。
   d. 指定された基準に基づいてオブジェクトをランク付けします。
4. 処理済み画像が base64 エンコードされた文字列としてクライアントに返されます。

```
[クライアント] -> [FastAPI] -> [Vertex AI] -> [画像処理] -> [FastAPI] -> [クライアント]
   |            |            |                 |                  |
   |            |            |                 |                  |
   +--- 画像 ---+--- 画像 ---+---- オブジェクト ----+---- 処理済み ---+
                                                     画像
```
