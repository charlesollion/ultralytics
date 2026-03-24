"""Standalone validation for multi-hot model with group-argmax decode."""

from ultralytics import YOLO

model = YOLO("runs/segment/train50/weights/best.pt")

# Inject decode config (model was trained before these were added to YAML)
model.model.yaml["decode_groups"] = {
    "material": [1, 2, 3, 4, 5, 6, 13],  # plastique, papier, carton, bois, metal, verre, textile
    "object": [0, 7, 8, 9, 10, 11, 12],  # alimentaire, bouteille, canette, encombrant, megot, paquet_cigarette, sac
}
model.model.yaml["decode_rules"] = {
    "object_priority": [8, 11],  # canette, paquet_cigarette
    "discard_on_invalid": [12],  # sac
    "material_priority": {5: [8, 9]},  # metal: only allow canette, encombrant
}

# Also ensure the 14-label names are available for per-label metrics
model.model.yaml["names"] = {
    0: "alimentaire",
    1: "plastique",
    2: "papier",
    3: "carton",
    4: "bois",
    5: "metal",
    6: "verre",
    7: "bouteille",
    8: "canette",
    9: "encombrant",
    10: "megot",
    11: "paquet_cigarette",
    12: "sac",
    13: "textile",
}

results = model.val(plots=True)
