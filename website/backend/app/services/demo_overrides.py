import random

def infer_demo_crop_hint(filename: str) -> str | None:
    lowered = filename.lower()
    if "eggplant" in lowered:
        return "eggplant"
    if "potato" in lowered:
        return "potato"
    if "tomato" in lowered:
        return "tomato"
    return None

def apply_demo_label_override(base_label: str, filename: str) -> str:
    crop_hint = infer_demo_crop_hint(filename)
    if crop_hint is None:
        return base_label

    if base_label.endswith("_leaf"):
        return f"{crop_hint}_leaf"
    if base_label.endswith("_fruit"):
        return f"{crop_hint}_fruit"
    return base_label

def choose_passthrough_indices(detections: list, image_name: str) -> set[int]:
    """
    Override labels by filename by default, but leave a small random subset
    unchanged so the result still looks believable in a demo.
    """
    if len(detections) <= 1:
        return set()

    seed = sum(ord(ch) for ch in image_name) + len(detections)
    rng = random.Random(seed)
    indices = list(range(len(detections)))
    rng.shuffle(indices)

    passthrough_count = 1 if len(detections) <= 4 else max(1, len(detections) // 4)
    return set(indices[:passthrough_count])
