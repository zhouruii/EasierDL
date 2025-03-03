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
