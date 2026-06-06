import json
import os

def main():
    try:
        with open("results/raw_scores.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    audit = {
        "total_records": len(data),
        "unique_prompt_ids": len(set(r.get("prompt_id") for r in data)),
        "unique_models": len(set(r.get("model_name") for r in data)),
        "average_latency_ms": 0,
        "average_token_estimate": 0,
        "total_valid_records": 0,
        "total_invalid_records": 0,
        "empty_responses": 0,
        "duplicate_entries": 0,
        "missing_fields": [],
        "longest_response": {"length": 0, "id": None},
        "shortest_response": {"length": float('inf'), "id": None}
    }

    if data:
        latencies = [r.get("latency_ms", 0) for r in data if "latency_ms" in r]
        tokens = [r.get("token_estimate", 0) for r in data if "token_estimate" in r]
        
        if latencies:
            audit["average_latency_ms"] = sum(latencies) / len(latencies)
        if tokens:
            audit["average_token_estimate"] = sum(tokens) / len(tokens)

    seen = set()
    required_fields = ["prompt_id", "model_name", "response", "latency_ms", "timestamp"]

    for i, r in enumerate(data):
        is_valid = True
        
        # Check required fields
        missing = [f for f in required_fields if f not in r]
        if missing:
            audit["missing_fields"].append({"index": i, "missing": missing})
            is_valid = False

        # Check empty responses
        resp = r.get("response", "")
        if not resp or str(resp).strip() == "":
            audit["empty_responses"] += 1
            # Empty is technically valid for JSON schema, but logically might be an issue. 
            # We'll just count it.

        # Check duplicates
        pid = r.get("prompt_id")
        mname = r.get("model_name")
        key = f"{mname}_{pid}"
        if key in seen:
            audit["duplicate_entries"] += 1
            is_valid = False
        seen.add(key)

        if is_valid:
            audit["total_valid_records"] += 1
        else:
            audit["total_invalid_records"] += 1

        # Response length stats
        resp_len = len(str(resp))
        if resp_len > audit["longest_response"]["length"]:
            audit["longest_response"] = {"length": resp_len, "id": pid}
        
        if resp_len < audit["shortest_response"]["length"]:
            audit["shortest_response"] = {"length": resp_len, "id": pid}

    os.makedirs("results", exist_ok=True)
    with open("results/raw_scores_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
