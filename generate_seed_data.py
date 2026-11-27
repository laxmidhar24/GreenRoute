"""Creates training_data.csv (no header) so the models can be trained before any real usage."""
import csv
import random
from datetime import datetime

from utils import estimate_emission

BASE = {"petrol": 0.192, "diesel": 0.171, "ev": 0.07, "bike": 0.103, "bus": 0.27}
random.seed(42)

with open("training_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    for _ in range(2000):
        fuel = random.choice(list(BASE))
        dist = random.uniform(1, 80)
        elev = random.uniform(0, 400)
        cong = random.uniform(0, 1.0)
        em = estimate_emission(dist, elev, BASE[fuel], cong) * random.uniform(0.95, 1.05)
        w.writerow([datetime.now().isoformat(), round(dist, 3), round(elev, 1), round(cong, 3),
                    fuel, BASE[fuel], round(em, 4), random.randint(20, 160)])
print("training_data.csv created with 2000 rows")
