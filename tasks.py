from models import Observation

DRUG_INTERACTIONS = {
    "amoxicillin": ["penicillin"],
    "aspirin": ["warfarin", "ibuprofen"],
    "ibuprofen": ["kidney disease", "warfarin", "aspirin"],
    "metformin": ["kidney disease"],
    "lisinopril": ["kidney disease", "potassium supplements"],
    "simvastatin": ["clarithromycin", "erythromycin"],
}

TASKS = {
    "easy": {
        "observation": Observation(
            patient_name="Raj Kumar",
            patient_age=35,
            allergies=["Penicillin"],
            conditions=["throat infection"],
            current_medications=[],
            new_prescription="Amoxicillin",
            dosage="500mg/day",
            task_id="easy"
        ),
        "expected_errors": ["penicillin allergy"],
        "expected_severity": "critical"
    },

    "medium": {
        "observation": Observation(
            patient_name="Sunita Sharma",
            patient_age=58,
            allergies=[],
            conditions=["blood clot history"],
            current_medications=["Warfarin"],
            new_prescription="Aspirin",
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
            new_prescription="Metformin + Ibuprofen",
            dosage="3000mg/day",
            task_id="hard"
        ),
        "expected_errors": ["dosage overdose", "contraindication with kidney disease"],
        "expected_severity": "critical"
    },

    "easy_2": {
        "observation": Observation(
            patient_name="Priya Nair",
            patient_age=28,
            allergies=["Sulfa"],
            conditions=["urinary tract infection"],
            current_medications=[],
            new_prescription="Sulfamethoxazole",
            dosage="800mg/day",
            task_id="easy_2"
        ),
        "expected_errors": ["sulfa allergy"],
        "expected_severity": "critical"
    },

    "medium_2": {
        "observation": Observation(
            patient_name="Arjun Reddy",
            patient_age=52,
            allergies=[],
            conditions=["high cholesterol"],
            current_medications=["Simvastatin"],
            new_prescription="Clarithromycin",
            dosage="500mg/day",
            task_id="medium_2"
        ),
        "expected_errors": ["drug interaction", "muscle damage risk"],
        "expected_severity": "critical"
    },

    "hard_2": {
        "observation": Observation(
            patient_name="Lakshmi Devi",
            patient_age=72,
            allergies=["Aspirin"],
            conditions=["heart failure", "kidney disease"],
            current_medications=["Lisinopril"],
            new_prescription="Ibuprofen + Aspirin",
            dosage="400mg/day",
            task_id="hard_2"
        ),
        "expected_errors": ["aspirin allergy", "contraindication with kidney disease", "drug interaction with lisinopril"],
        "expected_severity": "critical"
    }
}


def grade_response(task_id: str, detected_errors: list, severity: str) -> float:
    task = TASKS[task_id]
    expected_errors = task["expected_errors"]
    expected_severity = task["expected_severity"]

    score = 0.0

    if severity.lower() == expected_severity.lower():
        score += 0.25

    errors_found = 0
    for expected in expected_errors:
        for detected in detected_errors:
            if expected.lower() in detected.lower():
                errors_found += 1
                break

    if len(expected_errors) > 0:
        score += 0.65 * (errors_found / len(expected_errors))

    score = round(score, 2)

    # Strictly enforce (0, 1) exclusive
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99

    return score