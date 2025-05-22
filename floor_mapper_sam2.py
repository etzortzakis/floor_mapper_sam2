import os
import json
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
# ---------- SAM2 SEGMENTATION FUNCTIONS ----------

def run_sam2_segmentation(image_path, replicate_api_key):
    import replicate
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    with open(image_path, "rb") as image_file:
        output = replicate.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",  # Replace with your actual model
            input={
                "image": image_file,
                "points_per_side": 50,
                "pred_iou_thresh": 0.60,
                "stability_score_thresh": 0.40
            }
        )
    return output  # Should contain "individual_masks"

def download_all_masks(sam2_output):
    from io import BytesIO
    masks = []
    for mask_obj in sam2_output["individual_masks"]:
        url = mask_obj.url if hasattr(mask_obj, "url") else mask_obj["url"]
        response = requests.get(url)
        mask_img = Image.open(BytesIO(response.content)).convert('L')
        mask_array = np.array(mask_img)
        masks.append(mask_array)
    return masks

def analyze_and_select_floor_mask(masks):
    mask_infos = []
    h, w = masks[0].shape

    for idx, mask in enumerate(masks):
        mask_bin = (mask > 0)
        area = np.count_nonzero(mask_bin)
        touches_border = (
            mask_bin[0, :].any() or mask_bin[-1, :].any() or
            mask_bin[:, 0].any() or mask_bin[:, -1].any()
        )
        mask_infos.append({
            "index": idx,
            "mask": mask_bin,
            "area": area,
            "touches_border": touches_border
        })

    candidates = [info for info in mask_infos if not info["touches_border"]]

    if not candidates:
        print("No non-border-touching mask found. Returning largest mask as fallback.")
        largest = max(mask_infos, key=lambda x: x["area"])
        return largest["mask"]

    largest = max(candidates, key=lambda x: x["area"])
    return largest["mask"]

def overlay_mask_on_image(image, mask, save_path="sam2_floor_mask_overlay.png"):
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green
    alpha = 0.5
    blended = cv2.addWeighted(image, 1.0, overlay, alpha, 0)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Floor Mask Overlay')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved overlay visualization to {save_path}")

# ---------- GRID GENERATION & VISUALIZATION FUNCTIONS ----------

def generate_grid_data(floor_mask, grid_size=5, coverage_threshold=0.4):
    h, w = floor_mask.shape
    tile_h = h // grid_size
    tile_w = w // grid_size

    tiles = []

    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            tile_mask = floor_mask[y0:y1, x0:x1]
            coverage = tile_mask.sum() / (tile_h * tile_w)

            tiles.append({
                "row": row,
                "col": col,
                "x": x0,
                "y": y0,
                "walkable": bool(coverage >= coverage_threshold)
            })

    return {
        "grid_size": grid_size,
        "tile_size": [tile_w, tile_h],
        "tiles": tiles
    }

def annotate_grid_on_image(image, floor_mask, grid_size=5, coverage_threshold=0.4, save_path="sam2_annotated_grid.png"):
    import matplotlib.patches as patches
    h, w = floor_mask.shape
    tile_h = h // grid_size
    tile_w = w // grid_size

    fig, ax = plt.subplots()
    ax.imshow(image)

    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            tile = floor_mask[y0:y1, x0:x1]
            coverage = tile.sum() / (tile_h * tile_w)

            walkable = coverage >= coverage_threshold

            if walkable:
                rect = patches.Rectangle(
                    (x0, y0), tile_w, tile_h,
                    linewidth=0,
                    facecolor='limegreen',
                    alpha=0.3
                )
                ax.add_patch(rect)

            border = patches.Rectangle(
                (x0, y0), tile_w, tile_h,
                linewidth=1.5,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(border)

    ax.set_title("Walkable Tile Grid Overlay")
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Annotated grid saved to {save_path}")

# ---------- MAIN PIPELINE ----------

def main(state: dict) -> dict:
    """
    Floor mapping pipeline using SAM2 segmentation and grid mapping, matching the updated SSA structure.
    Input:
        - state["prompt"]: str, required
        - state["replicate_api_key"]: str, required
        - state["scene_size_m"]: int, required 
    Output:
        - dict with:
            - "image": base64-encoded annotated grid image (str)
            - "grid_size": number of tiles per side (int)
            - "map": 2D list of 1s (walkable) and 0s (non-walkable)
    """
    prompt = state.get("prompt")
    api_key = state.get("replicate_api_key")
    scene_size_m = state.get("scene_size_m")

    if not prompt or not api_key or not scene_size_m:
        return {"error": "Missing 'prompt', 'scene_size_m', or 'replicate_api_key'"}

    try:
        os.environ["REPLICATE_API_TOKEN"] = api_key

        # === Step 1: Generate image ===
        print("→ Generating image with Flux Schnell")
        import replicate
        result = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "megapixels": "0.25",
                "output_format": "jpg"
            }
        )
        image_url = result[0].url
        image_path = "generated_map.jpg"
        with open(image_path, "wb") as f:
            f.write(requests.get(image_url).content)
        image = np.array(Image.open(image_path))

        # === Step 2: Segment using SAM2 ===
        print("→ Segmenting image using SAM2")
        sam2_output = run_sam2_segmentation(image_path, api_key)
        masks = download_all_masks(sam2_output)

        # === Step 3: Convert masks to DataFrame ===
        # Ensure all masks have the same size as the image
        mask_size = list(image.shape[:2])
        df = []
        for mask in masks:
            area = int(np.count_nonzero(mask))
            size = list(mask.shape)
            df.append({
                "area": area,
                "mask": mask,
                "segmentation": {"size": size}
            })
        df = pd.DataFrame(df)

        # === Step 4: Floor + Grid Processing ===
        # Calculate grid size as before
        grid_size = math.ceil(scene_size_m / 1.5)
        output_prefix = "output_map"
        floor_mask, grid_data, files = process_main_pipeline(
            df, image, output_prefix=output_prefix,
            grid_size=grid_size, min_pixels=20000, center_tolerance=0.25
        )

        if floor_mask is None:
            return {"error": "Failed to detect floor mask"}

        # === Step 5: Output Processing ===
        with open(files["annotated_image"], "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        size = grid_data["grid_size"]
        tile_map = [[0] * size for _ in range(size)]
        for tile in grid_data["tiles"]:
            row, col = tile["row"], tile["col"]
            tile_map[row][col] = 1 if tile["walkable"] else 0

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

    return {
        "image": image_b64,
        "grid_size": grid_size,
        "map": tile_map
    }
