import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from sudachipy import Dictionary, SplitMode
from umap import UMAP


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HDBSCAN_MIN_CLUSTER_SIZE = 5
NUM_TOPICS = 5
LABEL_MAX_LENGTH = 15
UMAP_RANDOM_STATE = 42
UMAP_N_NEIGHBORS = 5
DOCUMENTS = [
    # --- 技術・プログラミング ---
    "Pythonは機械学習やデータ分析で広く使われるプログラミング言語です",
    "scikit-learnは機械学習のための代表的なPythonライブラリです",
    "深層学習ではPyTorchやTensorFlowが人気のフレームワークです",
    "自然言語処理ではBERTやGPTなどの大規模言語モデルが注目されています",
    "Rustはメモリ安全性を重視したシステムプログラミング言語です",
    "Dockerはコンテナ技術を利用してアプリケーションを隔離実行できます",
    "Kubernetesはコンテナオーケストレーションの標準的なプラットフォームです",
    "GitHubはソースコード管理とチーム開発のためのプラットフォームです",
    "TypeScriptはJavaScriptに静的型付けを追加した言語です",
    "データベースにはリレーショナル型とNoSQL型の2種類があります",
    # --- 観光・地理 ---
    "東京タワーは東京都港区にある高さ333メートルの電波塔です",
    "京都の金閣寺は世界遺産に登録されている有名な観光地です",
    "富士山は日本一高い山で標高は3776メートルです",
    "沖縄の美ら海水族館は巨大な水槽でジンベエザメを展示しています",
    "北海道の函館山からは美しい夜景を一望できます",
    "奈良公園では野生の鹿と触れ合うことができる人気スポットです",
    "広島の厳島神社は海に浮かぶ鳥居で有名な世界遺産です",
    "日光東照宮は徳川家康を祀る歴史的な神社です",
    "屋久島の縄文杉は樹齢数千年といわれる巨大な屋久杉です",
    "姫路城は白鷺城とも呼ばれる日本を代表する城郭建築です",
    # --- スポーツ ---
    "サッカーワールドカップは4年に一度開催される国際大会です",
    "野球は日本で最も人気のあるスポーツの一つです",
    "オリンピックは世界中のアスリートが集まるスポーツの祭典です",
    "テニスの四大大会は全豪・全仏・ウィンブルドン・全米の4つです",
    "マラソンは42.195キロメートルを走る長距離走競技です",
    "バスケットボールのNBAは世界最高峰のプロリーグです",
    "ラグビーワールドカップは2019年に日本で初開催されました",
    "水泳の自由形は最も速いタイムが出る泳法です",
    "ゴルフのマスターズはオーガスタで毎年開催される名門大会です",
    "柔道は日本発祥の武道でオリンピック正式種目です",
    # --- 料理・食文化 ---
    "寿司は酢飯と新鮮な魚介を組み合わせた日本の伝統料理です",
    "ラーメンは地域ごとに味噌・醤油・豚骨など多様なスタイルがあります",
    "和菓子は季節の素材を活かした繊細な日本の菓子文化です",
    "天ぷらは食材に衣をつけて油で揚げる江戸時代からの料理法です",
    "抹茶は茶道で用いられる粉末状の緑茶で海外でも人気です",
    "うどんは小麦粉から作る太い麺で讃岐うどんが特に有名です",
    "焼肉は韓国料理の影響を受けて日本で独自に発展しました",
    "おにぎりは米を握って海苔で巻いた日本の携帯食です",
    "味噌汁は大豆から作る味噌を使った日本の代表的な汁物です",
    "日本酒は米と水と麹から造られる日本固有の醸造酒です",
    # --- 科学・宇宙 ---
    "はやぶさ2は小惑星リュウグウからサンプルを持ち帰りました",
    "国際宇宙ステーションは地上約400キロメートルを周回しています",
    "ブラックホールは光さえも脱出できない強い重力を持つ天体です",
    "iPS細胞は山中伸弥教授が開発した万能細胞技術です",
    "量子コンピュータは量子力学の原理を利用した次世代の計算機です",
    "火星探査ではNASAのローバーが地表の調査を続けています",
    "ゲノム編集技術CRISPRは遺伝子治療の可能性を広げています",
    "重力波の検出はアインシュタインの予言を実証した成果です",
    "超伝導は特定の温度以下で電気抵抗がゼロになる現象です",
    "ニュートリノは質量がほぼゼロの素粒子で検出が困難です",
]


def sudachi_tokenize(text: str) -> list[str]:
    # SudachiPyで形態素解析し正規化形を返す（1文字は除外）
    sudachi_tokenizer = Dictionary().create()
    target_pos = {"名詞"}
    morphemes = sudachi_tokenizer.tokenize(text, SplitMode.C)
    target_normalized_tokens = [
        m.normalized_form()
        for m in morphemes
        if m.part_of_speech()[0] in target_pos and len(m.normalized_form()) > 1
    ]
    return target_normalized_tokens


def run_topic_modeling(documents: list[str]) -> None:
    # BERTopicモデルを作成しトピック抽出を実行する
    embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    vectorizer = CountVectorizer(tokenizer=sudachi_tokenize)
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        prediction_data=True,
    )
    model = BERTopic(
        embedding_model=embedding_model_name,
        vectorizer_model=vectorizer,
        hdbscan_model=hdbscan_model,
        nr_topics=NUM_TOPICS,
    )

    # トピック抽出
    topics, _probs = model.fit_transform(documents)
    topic_info = model.get_topic_info()

    logger.info("=== トピック一覧 ===")
    logger.info("検出されたトピック数: %d", len(topic_info) - 1)
    for _, row in topic_info.iterrows():
        logger.info(
            "Topic %d (文書数: %d): %s", row["Topic"], row["Count"], row["Name"]
        )

    logger.info("=== 各文書へトピックを割り当て ===")
    for doc, topic_id in zip(documents, topics, strict=True):
        logger.info("Topic %d: %s", topic_id, doc)

    # 二次元プロット表示
    plot_topics_2d(documents, topics, topic_info, embedding_model_name)


def plot_topics_2d(
    documents: list[str],
    topics: list[int],
    topic_info: pd.DataFrame,
    embedding_model_name: str,
) -> None:
    # 文書埋め込みを2次元に削減しトピックごとに色分けした散布図を表示する
    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(documents)

    reducer = UMAP(
        n_components=2,
        random_state=UMAP_RANDOM_STATE,
        n_neighbors=UMAP_N_NEIGHBORS,
    )
    coords = reducer.fit_transform(embeddings)

    topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"], strict=True))
    unique_topics = sorted(set(topics))

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    fig, ax = plt.subplots(figsize=(12, 8))
    topics_array = np.array(topics)

    for topic_id in unique_topics:
        mask = topics_array == topic_id
        label = topic_name_map.get(topic_id, f"Topic {topic_id}")
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.7, s=60)

    for i, doc in enumerate(documents):
        short_label = (
            doc[:LABEL_MAX_LENGTH] + "..." if len(doc) > LABEL_MAX_LENGTH else doc
        )
        ax.annotate(short_label, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.6)

    ax.set_title("BERTopic 2D")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_topic_modeling(DOCUMENTS)
