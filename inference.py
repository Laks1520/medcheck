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

    print(f"[START] task={task_id}", flush=True)

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
        print(f"[STEP] task={task_id} step=1 reward=0.01 error={str(e)}", flush=True)
        print(f"[END] task={task_id} score=0.01 steps=1", flush=True)
        return 0.01

    try:
        _, reward, done, info = env.step(action)
    except Exception as e:
        print(f"[STEP] task={task_id} step=1 reward=0.01 error={str(e)}", flush=True)
        print(f"[END] task={task_id} score=0.01 steps=1", flush=True)
        return 0.01

    # Clamp score strictly between 0 and 1
    final_score = reward.score
    if final_score <= 0.0:
        final_score = 0.01
    if final_score >= 1.0:
        final_score = 0.99

    print(f"[STEP] task={task_id} step=1 reward={final_score}", flush=True)
    print(f"[END] task={task_id} score={final_score} steps=1", flush=True)

    return final_score


if __name__ == "__main__":
    print("=== MedCheck Baseline Inference ===", flush=True)

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[STEP] task={task_id} step=1 reward=0.01 error={str(e)}", flush=True)
            print(f"[END] task={task_id} score=0.01 steps=1", flush=True)
            scores[task_id] = 0.01

    print("=== FINAL SCORES ===", flush=True)
    for task_id, score in scores.items():
        print(f"{task_id}: {score}", flush=True)