import base64
import io
import json
import logging
import os
import re
import tempfile
from typing import Any, cast

import vertexai
import vertexai.generative_models as genmod
from PIL import Image, ImageDraw, ImageFont

if os.getenv("IS_PROD"):
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PERSPECTIVE: int = getattr(Image, "PERSPECTIVE", 5)
BICUBIC: int = getattr(Image, "BICUBIC", 3)

BOUNDING_BOX_SYSTEM_INSTRUCTIONS = (
    "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. "
    "Limit to 25 objects. If an object is present multiple times, name them according to their unique characteristic"
    "(colors, size, position, unique characteristics, etc..)."
)
DETECTION_PROMPT = "Show me the positions of the vegetables in the image."


def clean_json_text(text: str) -> str:
    """テキストからmarkdownのコードフェンスを削除します。

    この関数は、テキストから開始コードフェンス（例："```json" や "```"）と
    対応する終了フェンスを削除し、周囲の空白を取り除きます。

    Args:
        text (str): markdownのコードフェンスを含む可能性のあるテキスト。

    Returns:
        str: コードフェンスを除去したクリーンなテキスト。
    """
    text = re.sub(r"^```(?:json)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def localize_objects(image_path: str) -> list[dict[str, Any]]:
    """生成モデルを使用して画像内のオブジェクトを検出します。

    指定されたパスの画像内のオブジェクトを生成モデルを使用して検出します。
    検出された各オブジェクトは、'box_2d'キー（0-1000のスケールの値）と'label'キーを
    含む辞書として表現されます。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        list[dict[str, Any]]: 検出されたオブジェクトを表す辞書のリスト。
    """
    logger.info(f"Detecting objects in image: {image_path}")

    safety_settings = [
        genmod.SafetySetting(
            category=genmod.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=genmod.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]
    config = genmod.GenerationConfig(temperature=0.5)
    model = genmod.GenerativeModel(
        "gemini-2.0-flash-exp",
        safety_settings=safety_settings,
        system_instruction=BOUNDING_BOX_SYSTEM_INSTRUCTIONS,
    )
    text_part = genmod.Part.from_text(DETECTION_PROMPT)
    image_part = genmod.Part.from_image(genmod.Image.load_from_file(image_path))
    response = model.generate_content([text_part, image_part], generation_config=config)
    logger.info("Received response from object detection generative model")
    cleaned_response = response.text.strip()
    logger.debug(f"Raw detection response: {cleaned_response}")
    json_output = clean_json_text(cleaned_response)
    logger.debug(f"Cleaned JSON for object detection: {json_output}")
    objects: list[dict[str, Any]] = []
    try:
        objects = json.loads(json_output)
        logger.info(f"Detected {len(objects)} objects.")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in object detection: {e}")
    return objects


def filter_objects_by_common_category(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """最も頻繁に検出されたカテゴリーのオブジェクトのみをフィルタリングします。

    Args:
        objs (list[dict[str, Any]]): 検出されたオブジェクトを表す辞書のリスト。

    Returns:
        list[dict[str, Any]]: 最も一般的なカテゴリーのオブジェクトのみを含むフィルタリングされたリスト。
    """
    category_counts: dict[str, int] = {}
    category_order: list[str] = []
    for obj in objs:
        label = obj.get("label", "Unknown")
        logger.info(f"Detected object: {label}")
        if label not in category_counts:
            category_counts[label] = 0
            category_order.append(label)
        category_counts[label] += 1

    if category_counts:
        max_count = max(category_counts.values())
        target_category: str | None = None
        for cat in category_order:
            if category_counts[cat] == max_count:
                target_category = cat
                break
        objs = [obj for obj in objs if obj.get("label") == target_category]
        logger.info(f"Filtered objects to target category '{target_category}' with {len(objs)} objects.")
    return objs


def crop_objects_from_image(image_path: str, objs: list[dict[str, Any]]) -> list[Image.Image]:
    """正規化されたバウンディングボックスに基づいて画像領域を切り抜きます。

    各オブジェクトの'box_2d'値（0-1000のスケール）を画像の寸法を使用して絶対座標に変換し、
    対応する画像領域を切り抜いて返します。

    Args:
        image_path (str): 画像ファイルのパス。
        objs (list[dict[str, Any]]): オブジェクト検出結果を含む辞書のリスト。

    Returns:
        list[Image.Image]: 切り抜かれた画像領域のリスト。
    """
    logger.info(f"Cropping objects from image: {image_path}")
    try:
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        logger.error(f"Failed to open image for cropping: {e}")
        return []

    cropped_images: list[Image.Image] = []
    for obj in objs:
        box = obj.get("box_2d")
        if not box or len(box) != 4:
            logger.warning(f"Invalid box_2d for object: {obj}")
            continue
        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        cropped = img.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_images.append(cropped)
        logger.debug(f"Cropped object with converted box: {(abs_x1, abs_y1, abs_x2, abs_y2)}")
    logger.info(f"Cropped {len(cropped_images)} objects from image.")
    return cropped_images


def draw_bounding_boxes_with_rank(
    image: Image.Image, objects: list[dict[str, Any]], ranking: list[dict[str, Any]]
) -> None:
    """上位5位までのランク付けされたオブジェクトに対して、拡張されたバウンディングボックスとランクラベルを描画します。

    正規化されたバウンディングボックス座標を絶対座標に変換し、上位5位のランキングオブジェクトに対して、
    各サイド10ピクセル拡張したバウンディングボックスとランクラベルを描画します。

    Args:
        image (Image.Image): ソース画像。
        objects (list[dict[str, Any]]): 検出されたオブジェクトのリスト。
        ranking (list[dict[str, Any]]): ランキング結果のリスト。上位5位（ランク順に昇順でソート）のみが使用されます。
    """
    logger.info("Drawing bounding boxes with rank on image")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=28
        )
        logger.debug("Using bold custom font for drawing")
    except IOError:
        font = ImageFont.load_default()
        logger.debug("Bold custom font not found. Using default font.")
    ranking_top5 = sorted(ranking, key=lambda x: x["rank"])[:5]
    for item in ranking_top5:
        image_number = item.get("image_number")
        rank = item.get("rank")
        if image_number is None or rank is None:
            logger.warning(f"Missing image_number or rank in ranking item: {item}")
            continue
        index = int(image_number) - 1
        if index < 0 or index >= len(objects):
            logger.warning(f"Invalid index {index} for objects list")
            continue
        box = objects[index].get("box_2d")
        if not box or len(box) != 4:
            logger.warning(f"Invalid box_2d for object at index {index}")
            continue
        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        new_x1 = max(0, abs_x1 - 10)
        new_y1 = max(0, abs_y1 - 10)
        new_x2 = abs_x2 + 10
        new_y2 = abs_y2 + 10
        draw.rectangle([(new_x1, new_y1), (new_x2, new_y2)], outline=(255, 255, 0), width=3)
        text = str(rank)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        margin = 2
        draw.rectangle(
            [(new_x1 - margin, new_y1 - margin), (new_x1 + text_width + margin, new_y1 + text_height + margin)],
            fill=(255, 255, 255),
        )
        draw.text((new_x1, new_y1), text, fill=(0, 0, 0), font=font)
        logger.debug(f"Drawn rank '{text}' at position ({new_x1}, {new_y1})")


def draw_bounding_boxes_only(image: Image.Image, objects: list[dict[str, Any]]) -> None:
    """すべてのオブジェクトに対して、右側に連番を表示した赤色のバウンディングボックスを描画します。

    正規化されたバウンディングボックス座標（0-1000のスケール）を絶対座標に変換し、
    各オブジェクトに赤い矩形を描画し、矩形の右側に連番（1から始まる）を白い背景で表示します。

    Args:
        image (Image.Image): ソース画像。
        objects (list[dict[str, Any]]): 検出されたオブジェクトのリスト。
    """
    logger.info("Drawing bounding boxes only on image with sequential numbering (right side, red)")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    red_color = (255, 0, 0)
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=28
        )
        logger.debug("Using bold custom font for drawing")
    except IOError:
        font = ImageFont.load_default()
        logger.debug("Bold custom font not found. Using default font.")
    for i, obj in enumerate(objects):
        box = obj.get("box_2d")
        if not box or len(box) != 4:
            logger.warning(f"Invalid box_2d for object: {obj}")
            continue
        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=red_color, width=4)
        number_str = str(i + 1)
        text_position = (abs_x2 + 5, abs_y1)
        bbox_text = draw.textbbox((0, 0), number_str, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        margin = 2
        draw.rectangle(
            [
                (text_position[0] - margin, text_position[1] - margin),
                (text_position[0] + text_width + margin, text_position[1] + text_height + margin),
            ],
            fill=(255, 255, 255),
        )
        draw.text(text_position, number_str, fill=(0, 0, 0), font=font)
        logger.debug(f"Drawn sequential number '{number_str}' at position {text_position}")


def draw_expanded_ranking_boxes(
    input_path: str, objs: list[dict[str, Any]], ranking: list[dict[str, Any]], output_path: str
) -> None:
    """画像にランキング用の拡張バウンディングボックスを描画します。

    `input_path`から画像を開き、すべてのオブジェクトの通常のバウンディングボックスを描画し
    （draw_bounding_boxes_onlyを使用）、その上に上位5位のランキングオブジェクトの
    拡張バウンディングボックス（各サイドを10ピクセル拡張）を重ねます。
    結果の画像は`output_path`に保存されます。

    Args:
        input_path (str): 入力画像のパス。
        objs (list[dict[str, Any]]): 検出されたオブジェクトのリスト。
        ranking (list[dict[str, Any]]): ランキング結果のリスト。
        output_path (str): 注釈付き画像を保存するパス。
    """
    ranking_top5 = sorted(ranking, key=lambda x: x["rank"])[:5]
    img_response = Image.open(input_path)
    draw_bounding_boxes_only(img_response, objs)
    width, height = img_response.size
    draw = ImageDraw.Draw(img_response)
    try:
        rank_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=28
        )
    except IOError:
        rank_font = ImageFont.load_default()
    for item in ranking_top5:
        image_number = item.get("image_number")
        rank = item.get("rank")
        if image_number is None or rank is None:
            continue
        index = int(image_number) - 1
        if index < 0 or index >= len(objs):
            continue
        box = objs[index].get("box_2d")
        if not box or len(box) != 4:
            continue
        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        new_x1 = max(0, abs_x1 - 10)
        new_y1 = max(0, abs_y1 - 10)
        new_x2 = abs_x2 + 10
        new_y2 = abs_y2 + 10
        draw.rectangle([(new_x1, new_y1), (new_x2, new_y2)], outline=(255, 255, 0), width=3)
        text = str(rank)
        bbox_text = draw.textbbox((0, 0), text, font=rank_font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        margin = 2
        draw.rectangle(
            [(new_x1 - margin, new_y1 - margin), (new_x1 + text_width + margin, new_y1 + text_height + margin)],
            fill=(255, 255, 255),
        )
        draw.text((new_x1, new_y1), text, fill=(0, 0, 0), font=rank_font)
    img_response.save(output_path)
    logger.info(f"Saved response image with combined bounding boxes to '{output_path}'.")


def draw_bounding_boxes_on_image(
    image_path: str, objects: list[dict[str, Any]], ranking: list[dict[str, Any]], output_path: str
) -> None:
    """画像を読み込み、通常のバウンディングボックスと拡張ランキングボックスの両方を描画し、出力画像を保存します。

    この関数は最初にすべてのオブジェクトのバウンディングボックス（右側に赤色の連番付き）を描画し、
    次に上位5位のランク付けされたオブジェクトの拡張バウンディングボックス
    （各サイドを10ピクセル拡張）を重ねます。

    Args:
        image_path (str): 入力画像ファイルのパス。
        objects (list[dict[str, Any]]): 検出されたオブジェクトのリスト。
        ranking (list[dict[str, Any]]): ランキング結果のリスト。
        output_path (str): 注釈付き画像を保存するパス。
    """
    try:
        original_img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to open image for drawing bounding boxes: {e}")
        return
    draw_bounding_boxes_only(original_img, objects)
    draw_expanded_ranking_boxes(image_path, objects, ranking, output_path)
    logger.info(f"Saved image with bounding boxes and ranks to '{output_path}'.")


def pil_image_to_bytes(pil_img: Image.Image, format: str = "PNG") -> bytes:
    """PIL画像をバイト配列に変換します。

    Args:
        pil_img (Image.Image): PIL画像オブジェクト。
        format (str, optional): 出力画像フォーマット。デフォルトは"PNG"。

    Returns:
        bytes: バイト配列としての画像データ。
    """
    logger.debug(f"Converting PIL image to bytes with format: {format}")
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()


def clean_response_text(text: str) -> str:
    """レスポンステキストからmarkdownのコードフェンスを削除します。

    Args:
        text (str): markdownのコードフェンスを含むレスポンステキスト。

    Returns:
        str: クリーンなテキスト。
    """
    logger.debug("Cleaning response text")
    text = re.sub(r"^```(?:json)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    cleaned = text.strip()
    logger.debug(f"Cleaned text: {cleaned}")
    return cleaned


def initialize_vertexai() -> None:
    """プロジェクトIDとロケーションでVertex AIを初期化します。

    Raises:
        EnvironmentError: PROJECT_ID環境変数が設定されていない場合。
    """
    project_id: str | None = os.getenv("PROJECT_ID")
    if project_id is None:
        logger.error("PROJECT_ID environment variable is not set")
        raise EnvironmentError("PROJECT_ID environment variable is not set")
    vertexai.init(project=project_id, location="us-central1")
    logger.debug("Initialized Vertex AI")


def create_parts_from_images_bytes(images_bytes: list[bytes]) -> list[genmod.Part]:
    """画像バイト配列のリストから生成モデルのパーツを作成します。

    Args:
        images_bytes (list[bytes]): バイトデータとしての画像のリスト。

    Returns:
        list[genmod.Part]: 生成モデル用のパーツのリスト。
    """
    parts: list[genmod.Part] = [genmod.Part.from_image(genmod.Image.from_bytes(b)) for b in images_bytes]
    return parts


def load_prompt_file(prompt_path: str) -> str:
    """ファイルからプロンプトテキストを読み込みます。

    Args:
        prompt_path (str): プロンプトのファイルパス。

    Returns:
        str: 読み込まれたプロンプトテキスト。
    """
    with open(prompt_path, "r", encoding="utf8") as f:
        prompt: str = f.read()
    logger.debug(f"Loaded prompt from {prompt_path}")
    return prompt


def generate_ranking_from_model(
    prompt: str, parts: list[genmod.Part], model_name: str = "gemini-1.5-pro-002"
) -> list[dict[str, Any]]:
    """提供されたプロンプトとパーツを使用して生成モデルからランキング情報を生成します。

    Args:
        prompt (str): プロンプトテキスト。
        parts (list[genmod.Part]): モデル用のパーツ（画像データ）のリスト。
        model_name (str, optional): モデル名。デフォルトは"gemini-1.5-pro-002"。

    Returns:
        list[dict[str, Any]]: 'image_number'と'rank'を含むランキング結果のリスト。
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"image_number": {"type": "integer"}, "rank": {"type": "integer"}},
            "required": ["image_number", "rank"],
        },
    }
    config = genmod.GenerationConfig(temperature=0.0, response_mime_type="application/json", response_schema=schema)
    model = genmod.GenerativeModel(model_name)
    response = model.generate_content([prompt, *parts], generation_config=config)
    logger.info("Received response from Generative Model for ranking")
    cleaned_text = clean_response_text(response.text)
    logger.info(f"Cleaned response text: {cleaned_text}")
    try:
        ranking = cast(list[dict[str, Any]], json.loads(cleaned_text))
        logger.debug(f"Parsed ranking JSON: {ranking}")
    except json.JSONDecodeError as e:
        logger.error(f"JSONデコードエラー: {e}")
        ranking = []
    return ranking


def process_base64_image(input_b64: str, product_type: str, desired_info: str) -> str:
    """オブジェクト検出、切り抜き、ランキング生成、画像への合成バウンディングボックスの描画を実行して、
    base64エンコードされた画像を処理します。

    結果の画像には以下が含まれます：
      - すべてのオブジェクトの赤色のバウンディングボックス（右側に連番表示）
      - 上位5位のランキングオブジェクトに対する拡張バウンディングボックス（各サイド10px拡張）とランクラベル

    Args:
        input_b64 (str): base64エンコードされた入力画像。
        product_type (str): 商品タイプ（例：'vegetables'）。
        desired_info (str): ランキングの基準。

    Returns:
        str: base64エンコードされた結果画像。
    """
    image_bytes = base64.b64decode(input_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as input_file:
        input_file.write(image_bytes)
        input_path = input_file.name
    logger.debug(f"Temporary input image path: {input_path}")

    output_path = tempfile.mktemp(suffix=".png")
    logger.debug(f"Temporary output image path: {output_path}")

    try:
        logger.info("Initializing Vertex AI")
        initialize_vertexai()

        logger.info(f"Processing image from temporary file: {input_path}")

        objs: list[dict[str, Any]] = localize_objects(input_path)
        logger.info(f"Number of detected objects: {len(objs)}")

        objs = filter_objects_by_common_category(objs)
        target_category = objs[0].get("label") if objs else "None"
        logger.info(f"Target category: {target_category}")
        logger.info(f"Number of objects in target category: {len(objs)}")
        logger.info(f"Objects in target category: {objs}")

        cropped_images: list[Image.Image] = crop_objects_from_image(input_path, objs)
        logger.info(f"Number of cropped images: {len(cropped_images)}")

        images_from_bytes: list[bytes] = [pil_image_to_bytes(img) for img in cropped_images]
        logger.debug("Converted images to bytes")

        parts = create_parts_from_images_bytes(images_from_bytes)

        prompt_path: str = f"./prompts/gemini/{product_type}.txt"
        prompt_template: str = load_prompt_file(prompt_path)
        criteria = desired_info if desired_info else "鮮度が良いものを選んでください。"
        num_images = len(cropped_images)
        prompt = prompt_template.format(criteria=criteria, num_images=num_images)
        logger.info(f"Prompt for ranking: {prompt}")
        ranking: list[dict[str, Any]] = generate_ranking_from_model(prompt, parts)
        logger.info(f"Generated ranking: {ranking}")

        draw_expanded_ranking_boxes(input_path, objs, ranking, output_path)

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
