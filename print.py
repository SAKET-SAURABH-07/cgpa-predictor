import pandas as pd
import numpy as np

#–– 1. Load your cleaned data
path = "cleaned_student_data.csv"
df = pd.read_csv(path)

#–– 2. Decide on the new prev_sem_cgpa curve
n = len(df)

#  a) a perfect descending line from 10 → 3
base = np.linspace(10.0, 3.0, n)

#  b) add a bit of small noise so it isn't too sterile
noise = np.random.normal(loc=0, scale=0.05, size=n)

#  c) combine and clip into [0, 10]
new_prev = (base + noise).clip(0, 10)

#–– 3. Assign back in original row-order
df['prev_sem_cgpa'] = new_prev

#–– 4. Save in-place
df.to_csv(path, index=False)

#–– 5. Print confirmation
print(f"[OK] Updated prev_sem_cgpa in {path}")
