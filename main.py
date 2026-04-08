from fastapi import FastAPI, HTTPException
from models import Observation, Action, Reward
from environment import MedCheckEnvironment

app = FastAPI(
    title="MedCheck — Prescription Error Detector",
    description="An OpenEnv environment where an AI agent detects dangerous prescription errors.",
    version="1.0.0"
)

# One environment instance
env = MedCheckEnvironment()

@app.get("/")
def home():
    """Health check — judges ping this to confirm deployment works"""
    return {
        "name": "MedCheck",
        "description": "Prescription Error Detector Environment",
        "status": "running",
        "tasks": ["easy", "medium", "hard"]
    }

@app.post("/reset")
def reset(task_id: str = "easy"):
    """
    Start a new episode.
    Returns the patient scenario for the agent to analyze.
    """
    try:
        observation = env.reset(task_id=task_id)
        return observation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: Action):
    """
    Agent submits their findings.
    Returns score + feedback.
    """
    try:
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    """Returns current environment state."""
    return env.state()

@app.get("/tasks")
def list_tasks():
    """Lists all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Allergy violation — patient allergic to Penicillin, prescribed Amoxicillin",
                "difficulty": "easy"
            },
            {
                "id": "medium", 
                "description": "Drug interaction — Warfarin + Aspirin dangerous bleeding risk",
                "difficulty": "medium"
            },
            {
                "id": "hard",
                "description": "Multiple errors — Metformin overdose + Ibuprofen contraindicated with kidney disease",
                "difficulty": "hard"
            }
        ]
    }