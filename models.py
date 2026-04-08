from pydantic import BaseModel
from typing import Optional, List

# This is the "patient + prescription" that gets sent TO the environment
class Action(BaseModel):
    detected_errors: List[str]  # list of errors the agent found
    severity: str               # "critical", "moderate", or "none"
    recommendation: str         # what the agent recommends doing

# This is what the environment sends BACK to the agent
class Observation(BaseModel):
    patient_name: str
    patient_age: int
    allergies: List[str]        # e.g. ["Penicillin"]
    conditions: List[str]       # e.g. ["kidney disease", "diabetes"]
    current_medications: List[str]
    new_prescription: str       # the drug being prescribed
    dosage: str                 # e.g. "3000mg/day"
    task_id: str                # "easy", "medium", or "hard"

# This is the reward/score the environment gives after each step
class Reward(BaseModel):
    score: float                # 0.0 to 1.0
    feedback: str               # explanation of the score