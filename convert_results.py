import json
from tqdm import tqdm

sample_json_data = json.load(open("public_test_sample_submission.json"))
json_data = json.load(open("public_test_predictions_.json"))

for sample_item in tqdm(sample_json_data):
    for item in json_data:
        if sample_item["id"] == item["filename"]:
            sample_item["captions"] = item["gen"][0]

json.dump(sample_json_data, open("public_test_results.json", "w+"), ensure_ascii=False)
