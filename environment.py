from models import Observation, Action, Reward
from tasks import TASKS, grade_response

class MedCheckEnvironment:
    def __init__(self):
        self.current_task_id = None
        self.current_observation = None
        self.done = False
        self.last_reward = None

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            task_id = "easy"
        self.current_task_id = task_id
        self.current_observation = TASKS[task_id]["observation"]
        self.done = False
        self.last_reward = None
        return self.current_observation

    def step(self, action: Action) -> tuple:
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        score = grade_response(
            task_id=self.current_task_id,
            detected_errors=action.detected_errors,
            severity=action.severity
        )

        intermediate_feedback = []

        if len(action.detected_errors) == 0:
            intermediate_feedback.append("No errors detected — check patient allergies and current medications.")
        else:
            intermediate_feedback.append(f"Agent identified {len(action.detected_errors)} potential error(s).")

        if action.severity not in ["critical", "moderate", "none"]:
            intermediate_feedback.append("Invalid severity level — must be critical, moderate, or none.")
        else:
            intermediate_feedback.append(f"Severity classified as: {action.severity}.")

        if score >= 0.8:
            final_feedback = "Excellent! All critical errors identified correctly."
        elif score >= 0.5:
            final_feedback = "Partial credit. Some errors missed."
        elif score > 0:
            final_feedback = "Poor performance. Most errors missed."
        else:
            final_feedback = "Failed. No errors correctly identified."

        full_feedback = " | ".join(intermediate_feedback) + " | " + final_feedback

        reward = Reward(score=score, feedback=full_feedback)
        self.last_reward = reward
        self.done = True

        return self.current_observation, reward, self.done, {
            "task_id": self.current_task_id,
            "score": score,
            "intermediate_feedback": intermediate_feedback,
            "final_feedback": final_feedback
        }

    def state(self) -> dict:
        return {
            "current_task_id": self.current_task_id,
            "done": self.done,
            "last_reward": self.last_reward.dict() if self.last_reward else None,
            "observation": self.current_observation.dict() if self.current_observation else None
        }