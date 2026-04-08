from models import Observation

# These are our 3 tasks - easy, medium, hard
# Each task has a patient scenario and a CORRECT answer the agent must find

TASKS = {
    "easy": {
        "observation": Observation(
            patient_name="Raj Kumar",
            patient_age=35,
            allergies=["Penicillin"],
            conditions=["throat infection"],
            current_medications=[],
            new_prescription="Amoxicillin",  # Amoxicillin IS a penicillin-type drug!
            dosage="500mg/day",
            task_id="easy"
        ),
        # What the correct answer looks like
        "expected_errors": ["penicillin allergy"],
        "expected_severity": "critical"
    },

    "medium": {
        "observation": Observation(
            patient_name="Sunita Sharma",
            patient_age=58,
            allergies=[],
            conditions=["blood clot history"],
            current_medications=["Warfarin"],  # blood thinner
            new_prescription="Aspirin",        # ALSO a blood thinner = dangerous!
            dosage="100mg/day",
            task_id="medium"
        ),
        "expected_errors": ["drug interaction", "bleeding risk"],
        "expected_severity": "critical"
    },

    "hard": {
        "observation": Observation(
            patient_name="Mohammed Ismail",
            patient_age=67,
            allergies=[],
            conditions=["kidney disease", "diabetes"],
            current_medications=["Insulin"],
            new_prescription="Metformin + Ibuprofen",  # both dangerous here!
            dosage="3000mg/day",                        # Metformin max is 2000mg!
            task_id="hard"
        ),
        "expected_errors": ["dosage overdose", "contraindication with kidney disease"],
        "expected_severity": "critical"
    }
}


def grade_response(task_id: str, detected_errors: list, severity: str) -> float:
    """
    This is the grader - it scores the agent's answer from 0.0 to 1.0
    Think of it like a marking scheme
    """
    task = TASKS[task_id]
    expected_errors = task["expected_errors"]
    expected_severity = task["expected_severity"]

    score = 0.0

    # Check severity (worth 30% of score)
    if severity.lower() == expected_severity.lower():
        score += 0.3

    # Check if agent found the errors (worth 70% of score)
    errors_found = 0
    for expected in expected_errors:
        for detected in detected_errors:
            # check if the key word appears in what agent said
            if expected.lower() in detected.lower():
                errors_found += 1
                break

    # partial credit - found some errors = some score
    if len(expected_errors) > 0:
        score += 0.7 * (errors_found / len(expected_errors))

    return round(score, 2)