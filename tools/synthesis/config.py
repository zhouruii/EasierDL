import random

DV = {
    0: 15,
    1: random.uniform(8, 10),
    2: random.uniform(6, 8),
}

RAIN = {
    1: 0.2083,
    2: 0.729167,
    3: 1.2,
}

RAIN_STREAK = {
    1: dict(num_drops=random.randint(3000, 3500), streak_length=random.randint(20, 25),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0, 0.05)),  # 小雨
    2: dict(num_drops=random.randint(3500, 4000), streak_length=random.randint(30, 35),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.05, 0.1)),  # 中雨
    3: dict(num_drops=random.randint(4500, 5000), streak_length=random.randint(40, 45),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.1, 0.15)),  # 大雨
}

RAIN_STREAK_BATCH = {
    'small': {"height": 1024, "width": 1024, "depth": 312, "num_drops": random.randint(1500, 1800),
              "streak_length": random.randint(30, 35), "wind_angle": random.randint(-180, 180),
              "wind_strength": random.uniform(0, 0.05), "f": 512},
    'medium': {"height": 1024, "width": 1024, "depth": 512, "num_drops": random.randint(1800, 1900),
               "streak_length": random.randint(40, 45), "wind_angle": random.randint(-180, 180),
               "wind_strength": random.uniform(0.05, 0.1), "f": 512},
    'heavy': {"height": 1024, "width": 1024, "depth": 512, "num_drops": random.randint(2800, 3200),
              "streak_length": random.randint(47, 53), "wind_angle": random.randint(-180, 180),
              "wind_strength": random.uniform(0.1, 0.2), "f": 512},
}

PARAMS = [
    {"height": 1024, "width": 1024, "depth": 312, "num_drops": 3000, "streak_length": 25, "wind_angle": 10,
     "wind_strength": 0.05, "f": 500, "type": "small"},
    {"height": 1024, "width": 1024, "depth": 512, "num_drops": 2000, "streak_length": 40, "wind_angle": 10,
     "wind_strength": 0.1, "f": 512, "type": "medium"},
    {"height": 1024, "width": 1024, "depth": 512, "num_drops": 3000, "streak_length": 50, "wind_angle": 10,
     "wind_strength": 0.2, "f": 512, "type": "heavy"}
]
