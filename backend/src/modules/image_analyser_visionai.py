import base64
import io
import json
import logging
import math
import os
import re
import tempfile
from typing import Any, Tuple, cast

import numpy as np
import vertexai
import vertexai.generative_models as genmod
from google.cloud import vision
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

if os.getenv("IS_PROD"):
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PERSPECTIVE: int = getattr(Image, "PERSPECTIVE", 5)
BICUBIC: int = getattr(Image, "BICUBIC", 3)


def localize_objects(path: str) -> list[dict[str, Any]]:
    """ローカル画像からオブジェクトを検出し、各オブジェクトの名前とバウンディングポリゴンの４頂点の正規化座標を取得する.

    Args:
        path (str): ローカル画像のパス.

    Returns:
        list[dict[str, Any]]: オブジェクトの名前と頂点情報を含む辞書のリスト.
    """
    logger.debug(f"Reading image from {path}")
    client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content: bytes = image_file.read()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    logger.debug(f"Detected {len(objects)} objects")

    result: list[dict[str, Any]] = []
    for obj in objects:
        name: str = obj.name
        vertices: list[Tuple[float, float]] = [
            (vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices
        ]
        logger.debug(f"Object '{name}' vertices: {vertices}")
        result.append({"name": name, "vertices": vertices})
    return result


def find_perspective_coeffs(
    src_pts: list[Tuple[float, float]], dst_pts: list[Tuple[float, float]]
) -> NDArray[np.float64]:
    """透視変換の係数を計算するヘルパー関数.

    Args:
        src_pts (list[Tuple[float, float]]): ソースの座標点.
        dst_pts (list[Tuple[float, float]]): 変換先の座標点.

    Returns:
        NDArray[np.float64]: 計算された透視変換の係数.
    """
    logger.debug("Calculating perspective coefficients")
    matrix = []
    for src, dst in zip(src_pts, dst_pts):
        matrix.append([src[0], src[1], 1, 0, 0, 0, -dst[0] * src[0], -dst[0] * src[1]])
        matrix.append([0, 0, 0, src[0], src[1], 1, -dst[1] * src[0], -dst[1] * src[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(dst_pts).reshape(8)
    coeffs = np.linalg.solve(A, B)
    logger.debug(f"Perspective coefficients: {coeffs}")
    return coeffs


def crop_object(image_path: str, norm_vertices: list[Tuple[float, float]]) -> Image.Image | None:
    """正規化座標から絶対座標に変換し、透視変換により画像から対象領域を切り出す.

    Args:
        image_path (str): 画像ファイルのパス.
        norm_vertices (list[Tuple[float, float]]): 正規化された座標点のリスト.

    Returns:
        Image.Image | None: 切り出された画像オブジェクト、または切り出しに失敗した場合は None.
    """
    if not norm_vertices or len(norm_vertices) != 4:
        logger.warning(f"Insufficient normalized vertices. Skipping crop for image {image_path}")
        return None

    img: Image.Image = Image.open(image_path)
    img_width, img_height = img.size
    logger.debug(f"Image size: {img_width}x{img_height}")

    src_pts: list[Tuple[float, float]] = [(v[0] * img_width, v[1] * img_height) for v in norm_vertices]
    logger.debug(f"Absolute source points: {src_pts}")

    def distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    width_top: float = distance(src_pts[0], src_pts[1])
    width_bottom: float = distance(src_pts[3], src_pts[2])
    w: int = int(max(width_top, width_bottom))

    height_left: float = distance(src_pts[0], src_pts[3])
    height_right: float = distance(src_pts[1], src_pts[2])
    h: int = int(max(height_left, height_right))
    logger.debug(f"Crop dimensions: width={w}, height={h}")

    dst_pts: list[Tuple[float, float]] = [(0, 0), (w, 0), (w, h), (0, h)]
    logger.debug(f"Destination points: {dst_pts}")

    coeffs: NDArray[np.float64] = find_perspective_coeffs(dst_pts, src_pts)

    transform_method: Any = PERSPECTIVE
    resample_mode: Any = BICUBIC
    cropped: Image.Image = img.transform((w, h), transform_method, coeffs.ravel().tolist(), resample_mode)
    logger.debug(f"Cropped image created with size: {cropped.size[0]}x{cropped.size[1]}")
    return cropped


def pil_image_to_bytes(pil_img: Image.Image, format: str = "PNG") -> bytes:
    """PIL Image オブジェクトをバイト列に変換する.

    Args:
        pil_img (Image.Image): 変換する PIL Image.
        format (str, optional): 出力フォーマット. Defaults to "PNG".

    Returns:
        bytes: 画像を表すバイト列.
    """
    logger.debug(f"Converting PIL image to bytes with format: {format}")
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()


def clean_response_text(text: str) -> str:
    """レスポンステキストから markdown のコードフェンス（```json ... ```）を取り除く.

    Args:
        text (str): 元のレスポンステキスト.

    Returns:
        str: コードフェンスを除去したクリーンなテキスト.
    """
    logger.debug("Cleaning response text")
    text = re.sub(r"^```(?:json)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    cleaned = text.strip()
    logger.debug(f"Cleaned text: {cleaned}")
    return cleaned


def draw_bounding_boxes_with_rank(
    image: Image.Image, objects: list[dict[str, Any]], ranking: list[dict[str, Any]]
) -> None:
    """元画像上に、各オブジェクトのバウンディングボックスと rank を左上に描画する.

    Args:
        image (Image.Image): 元画像.
        objects (list[dict[str, Any]]): 検出されたオブジェクト情報（正規化頂点を含む）.
        ranking (list[dict[str, Any]]): Generative Model の出力。各辞書は {"image_number": ..., "rank": ...} を含む.
    """
    draw = ImageDraw.Draw(image)

    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=28
        )
        logger.debug("Using bold custom font for drawing")
    except IOError:
        font = ImageFont.load_default()
        logger.debug("Bold custom font not found. Using default font.")

    width, height = image.size
    for item in ranking:
        image_number = item.get("image_number")
        rank = item.get("rank")
        if image_number is None or rank is None:
            logger.warning(f"Missing image_number or rank in item: {item}")
            continue
        index = int(image_number) - 1
        if index < 0 or index >= len(objects):
            logger.warning(f"Invalid index {index} for objects list")
            continue

        norm_vertices = objects[index]["vertices"]
        abs_vertices = [(v[0] * width, v[1] * height) for v in norm_vertices]
        logger.debug(f"Drawing bounding box for object {index} with vertices: {abs_vertices}")

        draw.line(abs_vertices + [abs_vertices[0]], fill=(255, 255, 0), width=3)

        text = str(rank)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = 2
        x0, y0 = abs_vertices[0]
        draw.rectangle(
            [
                x0 - margin,
                y0 - margin,
                x0 + text_width + margin,
                y0 + text_height + margin,
            ],
            fill=(255, 255, 255),
        )
        offset = 10
        draw.text((x0, y0 - offset), text, fill=(0, 0, 0), font=font)
        logger.debug(f"Drawn rank '{text}' at position ({int(x0)}, {int(y0)})")


def filter_objects_by_common_category(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """objs を name でカテゴリ分けし、一番数の多いカテゴリのオブジェクトのみを返す.

    同数の場合、最初に登場したカテゴリを優先する.

    Args:
        objs (list[dict[str, Any]]): オブジェクトのリスト.

    Returns:
        list[dict[str, Any]]: 最も多いカテゴリに属するオブジェクトのみを含むリスト.
    """
    category_counts: dict[str, int] = {}
    category_order: list[str] = []
    for obj in objs:
        name = obj["name"]
        logger.info(f"Detected object: {name}")
        if name not in category_counts:
            category_counts[name] = 0
            category_order.append(name)
        category_counts[name] += 1

    if category_counts:
        max_count = max(category_counts.values())
        target_category: str | None = None
        for cat in category_order:
            if category_counts[cat] == max_count:
                target_category = cat
                break
        objs = [obj for obj in objs if obj["name"] == target_category]
    return objs


def crop_objects_from_image(image_path: str, objs: list[dict[str, Any]]) -> list[Image.Image]:
    """各オブジェクトごとに透視変換により対象領域を切り出し、切り出した画像のリストを返す.

    Args:
        image_path (str): 画像ファイルのパス.
        objs (list[dict[str, Any]]): オブジェクト情報のリスト.

    Returns:
        list[Image.Image]: 切り出された画像オブジェクトのリスト.
    """
    images: list[Image.Image] = []
    for obj in objs:
        cropped_img = crop_object(image_path, obj["vertices"])
        if cropped_img is not None:
            images.append(cropped_img)
    return images


def images_to_bytes(images: list[Image.Image], format: str = "PNG") -> list[bytes]:
    """PIL Image のリストを、バイト列のリストに変換して返す.

    Args:
        images (list[Image.Image]): PIL Image オブジェクトのリスト.
        format (str, optional): 出力フォーマット. Defaults to "PNG".

    Returns:
        list[bytes]: 変換された画像のバイト列のリスト.
    """
    return [pil_image_to_bytes(img, format) for img in images]


def initialize_vertexai() -> None:
    """Vertex AI の初期化を実施する.

    環境変数 PROJECT_ID を利用し、us-central1 ロケーションで初期化する.

    Raises:
        EnvironmentError: PROJECT_ID 環境変数が設定されていない場合.
    """
    project_id: str | None = os.getenv("PROJECT_ID")
    if project_id is None:
        logger.error("PROJECT_ID environment variable is not set")
        raise EnvironmentError("PROJECT_ID environment variable is not set")
    vertexai.init(project=project_id, location="us-central1")
    logger.debug("Initialized Vertex AI")


def create_parts_from_images_bytes(images_bytes: list[bytes]) -> list[genmod.Part]:
    """バイト列のリストから、Generative Model に渡すための parts のリストを作成して返す.

    Args:
        images_bytes (list[bytes]): 画像のバイト列のリスト.

    Returns:
        list[genmod.Part]: 作成された parts のリスト.
    """
    parts: list[genmod.Part] = [genmod.Part.from_image(genmod.Image.from_bytes(b)) for b in images_bytes]
    return parts


def load_prompt_file(prompt_path: str) -> str:
    """指定したパスから prompt ファイルを読み込み、その内容を返す.

    Args:
        prompt_path (str): 読み込む prompt ファイルのパス.

    Returns:
        str: ファイルの内容.
    """
    with open(prompt_path, "r", encoding="utf8") as f:
        prompt: str = f.read()
    logger.debug(f"Loaded prompt from {prompt_path}")
    return prompt


def generate_ranking_from_model(
    prompt: str, parts: list[genmod.Part], model_name: str = "gemini-1.5-pro-002"
) -> list[dict[str, Any]]:
    """prompt と parts を用いて Generative Model からランキング情報を生成する.

    期待するレスポンスのスキーマは以下の通り:
        [
            {"image_number": int, "rank": int},
            ...
        ]

    Args:
        prompt (str): プロンプト文字列.
        parts (list[genmod.Part]): Generative Model に渡すパートのリスト.
        model_name (str, optional): 使用するモデル名. Defaults to "gemini-1.5-pro-002".

    Returns:
        list[dict[str, Any]]: ランキング情報のリスト.
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"image_number": {"type": "integer"}, "rank": {"type": "integer"}},
            "required": ["image_number", "rank"],
        },
    }
    config = genmod.GenerationConfig(response_mime_type="application/json", response_schema=schema)
    model = genmod.GenerativeModel(model_name)
    response = model.generate_content([prompt, *parts], generation_config=config)
    logger.info("Received response from Generative Model")
    cleaned_text = clean_response_text(response.text)
    logger.info(f"Cleaned response text: {cleaned_text}")
    try:
        ranking = cast(list[dict[str, Any]], json.loads(cleaned_text))
        logger.debug(f"Parsed ranking JSON: {ranking}")
    except json.JSONDecodeError as e:
        logger.error(f"JSONデコードエラー: {e}")
        ranking = []
    return ranking


def draw_bounding_boxes_on_image(
    image_path: str,
    objs: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    output_path: str,
) -> None:
    """元画像を読み込み、各オブジェクトのバウンディングボックスとランキング情報を描画した後、指定された出力パスに画像を保存する.

    Args:
        image_path (str): 元画像のファイルパス.
        objs (list[dict[str, Any]]): オブジェクト情報のリスト.
        ranking (list[dict[str, Any]]): ランキング情報のリスト.
        output_path (str): 出力画像の保存パス.
    """
    original_img = Image.open(image_path)
    draw_bounding_boxes_with_rank(original_img, objs, ranking)
    original_img.save(output_path)
    logger.info(f"Bounding Box と rank を描画した画像を '{output_path}' として保存しました。")


def process_base64_image(input_b64: str, product_type: str, desired_info: str) -> str:
    """base64 エンコードされた画像文字列を入力として受け取り、バウンディングボックスとランキング情報を描画した画像を
    base64 エンコード文字列として返す.

    Args:
        input_b64 (str): 入力画像の base64 エンコード文字列.
        product_type (str): プロダクトタイプに応じたプロンプトの選択.
        desired_info (str): 追加の情報指示文字列.

    Returns:
        str: 処理後の画像の base64 エンコード文字列.
    """
    image_bytes = base64.b64decode(input_b64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as input_file:
        input_file.write(image_bytes)
        input_path = input_file.name
    logger.debug(f"Temporary input image path: {input_path}")

    output_path = tempfile.mktemp(suffix=".png")
    logger.debug(f"Temporary output image path: {output_path}")

    try:
        logger.info(f"Processing image from temporary file: {input_path}")

        objs: list[dict[str, Any]] = localize_objects(input_path)
        logger.info(f"Number of detected objects: {len(objs)}")

        objs = filter_objects_by_common_category(objs)
        target_category = objs[0]["name"] if objs else "None"
        logger.info(f"Target category: {target_category}")
        logger.info(f"Number of objects in target category: {len(objs)}")

        cropped_images: list[Image.Image] = crop_objects_from_image(input_path, objs)
        logger.info(f"Number of cropped images: {len(cropped_images)}")

        images_from_bytes: list[bytes] = images_to_bytes(cropped_images)
        logger.debug("Converted images to bytes")

        initialize_vertexai()
        parts = create_parts_from_images_bytes(images_from_bytes)

        prompt_path: str = f"./prompts/visionapi/{product_type}.txt"
        prompt: str = load_prompt_file(prompt_path)
        if desired_info:
            prompt += f"\n{desired_info}"
        else:
            prompt += "\n鮮度が良いものを選んでください。"
        logger.info(f"Prompt: {prompt}")
        ranking: list[dict[str, Any]] = generate_ranking_from_model(prompt, parts)

        draw_bounding_boxes_on_image(input_path, objs, ranking, output_path)

        with open(output_path, "rb") as f:
            output_bytes = f.read()
        output_b64 = base64.b64encode(output_bytes).decode("utf-8")

    finally:
        try:
            os.remove(input_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary input file: {e}")
        try:
            os.remove(output_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary output file: {e}")

    return output_b64
