"""
TimberGrade - Grading logic
Gives a plank Grade A / B / C based on the type and size of its defects.
"""

# How serious each defect is (penalty points)
DEFECT_INFO = {
    "crack":           {"name": "Crack",            "penalty": 25, "serious": True},
    "knot_with_crack": {"name": "Knot with crack",        "penalty": 22, "serious": True},
    "dead_knot":       {"name": "Dead knot", "penalty": 12, "serious": False},
    "knot_missing":    {"name": "Missing knot (hole)",           "penalty": 14, "serious": False},
    "live_knot":       {"name": "Live knot",  "penalty": 5,  "serious": False},
    "resin":           {"name": "Resin",              "penalty": 6,  "serious": False},
    "marrow":          {"name": "Marrow",   "penalty": 6,  "serious": False},
    "quartzity":       {"name": "Quartzity",            "penalty": 4,  "serious": False},
    "blue_stain":      {"name": "Blue stain",   "penalty": 8,  "serious": False},
    "overgrown":       {"name": "Overgrown",                 "penalty": 8,  "serious": False},
}
DEFAULT = {"name": "Other defect", "penalty": 8, "serious": False}

GRADE_USE = {
    "A": "Furniture, interiors and export quality",
    "B": "Construction, doors and general use",
    "C": "Only packaging, pallets or crates",
}


def info(name):
    return DEFECT_INFO.get(name.lower().replace(" ", "_"), DEFAULT)


def grade_plank(detections, img_w, img_h):
    """
    detections: list of (class_name, confidence, (x1, y1, x2, y2))
    returns: dict(grade, score, use, defects=[...], counts={...})
    """
    area = max(1, img_w * img_h)
    score, has_serious, defects, counts = 100.0, False, [], {}
    for name, conf, (x1, y1, x2, y2) in detections:
        d = info(name)
        frac = max(0.0, (x2 - x1) * (y2 - y1)) / area
        size_factor = 1 + min(frac * 20, 1.0)          # bada defect = zyada penalty (max 2x)
        penalty = d["penalty"] * size_factor
        score -= penalty
        has_serious |= d["serious"]
        defects.append({"Defect": d["name"], "Class": name, "Confidence %": round(conf * 100),
                        "Area % of plank": round(frac * 100, 2), "Penalty": round(penalty, 1)})
        counts[d["name"]] = counts.get(d["name"], 0) + 1
    score = max(0, round(score))

    if score >= 85 and not has_serious:
        grade = "A"
    elif score >= 60:
        grade = "B"
    else:
        grade = "C"
    return {"grade": grade, "score": score, "use": GRADE_USE[grade], "defects": defects, "counts": counts}


if __name__ == "__main__":
    print(grade_plank([], 2800, 1024))
    print(grade_plank([("Live_Knot", .9, (100, 100, 180, 160))], 2800, 1024))
    print(grade_plank([("Dead_Knot", .94, (0, 0, 200, 150)), ("Crack", .91, (300, 400, 900, 450)),
                       ("Live_Knot", .89, (1000, 300, 1080, 360))], 2800, 1024))
    