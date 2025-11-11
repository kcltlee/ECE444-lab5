import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

'''
Flask app exposed endpoint

POST /predict
Content-Type: application/json
Body: {"message": "<text to classify>"}

'''

# === 1. Your deployed API URL ===
API_URL = "http://ece444-env.eba-djnmkrrq.us-east-2.elasticbeanstalk.com/predict"  # <-- replace with your endpoint

# === 2. Define 4 test cases ===
TEST_CASES = {
    "Fake_1": "Breaking news: President resigns amid scandal and corruption allegations.",
    "Fake_2": "Aliens have landed in Ottawa and are negotiating trade deals with Canada.",
    "Real_1": "The Bank of Canada announced it will hold its key interest rate at 5% this month.",
    "Real_2": "Toronto's transit authority plans to expand its subway network by 2030."
}

# === 3. Number of repetitions per case ===
N_RUNS = 100

# === 4. Storage for results ===
results = []

for label, text in TEST_CASES.items():
    print(f"\nTesting case: {label}")
    for i in range(N_RUNS):
        start = time.time()
        res = requests.post(API_URL, json={"message": text})
        latency = time.time() - start

        results.append({
            "test_case": label,
            "run": i + 1,
            "latency_sec": latency,
            "status_code": res.status_code,
            "response": res.json() if res.headers.get("Content-Type") == "application/json" else res.text
        })

# === 3. Save results ===
df = pd.DataFrame(results)
df.to_csv("api_latency_results.csv", index=False)
print("\nResults saved to api_latency_results.csv")

# === 4. Create boxplot ===
plt.figure(figsize=(10, 6))
df.boxplot(column="latency_sec", by="test_case", grid=False)
plt.title("API Latency Distribution per Test Case")
plt.suptitle("")  # remove the default Pandas title
plt.xlabel("Test Case")
plt.ylabel("Latency (seconds)")
plt.savefig("api_latency_boxplot.png")
plt.show()

# === 5. Print average latency ===
print("\nAverage Latency per Test Case (seconds):")
print(df.groupby("test_case")["latency_sec"].mean())