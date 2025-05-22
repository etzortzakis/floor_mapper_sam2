import os
import json
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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
    Floor mapping pipeline using SAM2 segmentation.
    """
    prompt = state.get("prompt")
    api_key = state.get("replicate_api_key")
    if not prompt or not api_key:
        state["error"] = "Missing 'prompt' or 'replicate_api_key'"
        return state

    try:
        os.environ["REPLICATE_API_TOKEN"] = api_key

        # --- Step 1: Generate or load image ---
        image_path = state.get("image_path", "generated_map.jpg")
        image = np.array(Image.open(image_path).convert("RGB"))
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Step 2: Run SAM2 Segmentation ---
        print("→ Segmenting image using SAM2")
        sam2_output = run_sam2_segmentation(image_path, api_key)

        # --- Step 3: Download all masks ---
        masks = download_all_masks(sam2_output)

        # --- Step 4: Analyze and select the floor mask ---
        floor_mask = analyze_and_select_floor_mask(masks)
        if floor_mask is None:
            state["error"] = "Failed to detect floor mask"
            return state

        # --- Step 5: Visualize the mask overlay ---
        overlay_mask_on_image(image, floor_mask, save_path="sam2_floor_mask_overlay.png")

        # --- Step 6: Grid Generation & Annotation ---
        grid_data = generate_grid_data(floor_mask, grid_size=5)
        with open("sam2_grid.json", "w") as f:
            json.dump(grid_data, f, indent=2)

        annotate_grid_on_image(image, floor_mask, grid_size=5, save_path="sam2_annotated_grid.png")

        # --- Step 7: Update state ---
        state.update({
            "floor_mask_image": "sam2_floor_mask_overlay.png",
            "grid_json": "sam2_grid.json",
            "annotated_image": "sam2_annotated_grid.png",
            "grid_data": grid_data
        })

    except Exception as e:
        state["error"] = f"Unexpected error: {e}"

    return state
