import os
import json
from openai import OpenAI
from environment import MedCheckEnvironment
from models import Action

# The judges run this script to verify your environment works
# It must print logs in EXACT format: [START], [STEP], [END]

client = OpenAI(
    api_key=os.environ.get("HF_TOKEN"),
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def run_task(task_id: str):
    """Run the AI agent on one task and print structured logs"""
    
    env = MedCheckEnvironment()
    observation = env.reset(task_id=task_id)

    # Convert observation to text for the LLM
    prompt = f"""You are a medical prescription safety checker.

Patient: {observation.patient_name}, Age: {observation.patient_age}
Allergies: {', '.join(observation.allergies) if observation.allergies else 'None'}
Existing Conditions: {', '.join(observation.conditions)}
Current Medications: {', '.join(observation.current_medications) if observation.current_medications else 'None'}
New Prescription: {observation.new_prescription}
Dosage: {observation.dosage}

Analyze this prescription for errors. Respond ONLY with valid JSON in this exact format:
{{
    "detected_errors": ["error1", "error2"],
    "severity": "critical" or "moderate" or "none",
    "recommendation": "your recommendation here"
}}"""

    print(json.dumps({"type": "[START]", "task_id": task_id}))

    # Call the LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    
    # Clean up response in case model adds markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    action = Action(**parsed)

    _, reward, done, info = env.step(action)

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
        score = run_task(task_id)
        scores[task_id] = score

    print("\n=== FINAL SCORES ===")
    for task_id, score in scores.items():
        print(f"{task_id}: {score}")
    print(f"Average: {round(sum(scores.values()) / len(scores), 2)}")