import os
import json
from openai import OpenAI
from environment import MedCheckEnvironment
from models import Action

def get_client():
    return OpenAI(
        api_key=os.environ.get("HF_TOKEN", "dummy-key"),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def run_task(task_id: str):
    env = MedCheckEnvironment()
    observation = env.reset(task_id=task_id)

    prompt = f"""You are a medical prescription safety checker.

Patient: {observation.patient_name}, Age: {observation.patient_age}
Allergies: {', '.join(observation.allergies) if observation.allergies else 'None'}
Existing Conditions: {', '.join(observation.conditions)}
Current Medications: {', '.join(observation.current_medications) if observation.current_medications else 'None'}
New Prescription: {observation.new_prescription}
Dosage: {observation.dosage}

Analyze this prescription for errors. Respond ONLY with valid JSON:
{{
    "detected_errors": ["error1", "error2"],
    "severity": "critical",
    "recommendation": "your recommendation here"
}}"""

    print(json.dumps({"type": "[START]", "task_id": task_id}))

    action = None
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30
        )

        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        action = Action(**parsed)

    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "error": str(e)}))
        action = Action(
            detected_errors=["inference error - could not complete analysis"],
            severity="none",
            recommendation="Inference failed gracefully"
        )

    try:
        _, reward, done, info = env.step(action)
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "error": str(e)}))
        reward_score = 0.0
        print(json.dumps({"type": "[STEP]", "task_id": task_id, "score": 0.0, "feedback": "step failed"}))
        print(json.dumps({"type": "[END]", "task_id": task_id, "final_score": 0.0}))
        return 0.0

    print(json.dumps({
        "type": "[STEP]",
        "task_id": task_id,
        "detected_errors": action.detected_errors,
        "severity": action.severity,
        "score": reward.score,
        "feedback": reward.feedback
    }))

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "final_score": reward.score
    }))

    return reward.score


if __name__ == "__main__":
    print("\n=== MedCheck Baseline Inference ===\n")

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n--- Running task: {task_id} ---")
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(json.dumps({"type": "[ERROR]", "task_id": task_id, "error": str(e)}))
            scores[task_id] = 0.0

    print("\n=== FINAL SCORES ===")
    for task_id, score in scores.items():
        print(f"{task_id}: {score}")
    print(f"Average: {round(sum(scores.values()) / len(scores), 2)}")