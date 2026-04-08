from models import Observation, Action, Reward
from tasks import TASKS, grade_response

class MedCheckEnvironment:
    """
    This is the actual OpenEnv environment.
    Think of it like a game:
    - reset() = start a new game
    - step()  = agent makes a move, environment scores it
    - state() = what does the environment look like right now
    """

    def __init__(self):
        self.current_task_id = None
        self.current_observation = None
        self.done = False
        self.last_reward = None

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Start fresh with a new task.
        Returns the patient scenario to the agent.
        """
        if task_id not in TASKS:
            task_id = "easy"  # default to easy if invalid

        self.current_task_id = task_id
        self.current_observation = TASKS[task_id]["observation"]
        self.done = False
        self.last_reward = None

        return self.current_observation

    def step(self, action: Action) -> tuple:
        """
        Agent submits their answer (detected errors + severity).
        Environment scores it and returns result.
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        # Grade the agent's response
        score = grade_response(
            task_id=self.current_task_id,
            detected_errors=action.detected_errors,
            severity=action.severity
        )

        # Build feedback message
        if score >= 0.8:
            feedback = "Excellent! Agent correctly identified the prescription errors."
        elif score >= 0.5:
            feedback = "Partial credit. Agent found some but not all errors."
        elif score > 0:
            feedback = "Poor performance. Agent missed most critical errors."
        else:
            feedback = "Failed. Agent did not identify any errors correctly."

        reward = Reward(score=score, feedback=feedback)
        self.last_reward = reward
        self.done = True  # one step per episode

        return self.current_observation, reward, self.done, {
            "task_id": self.current_task_id,
            "score": score
        }

    def state(self) -> dict:
        """
        Returns current state of the environment.
        """
        return {
            "current_task_id": self.current_task_id,
            "done": self.done,
            "last_reward": self.last_reward.dict() if self.last_reward else None,
            "observation": self.current_observation.dict() if self.current_observation else None
        }