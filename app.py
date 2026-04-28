from fastapi import FastAPI, Body
import prompt_temp
import connect_db
import os
import json

app = FastAPI()

latest_output = None

@app.post("/doctor/notes")
async def save_doctor_notes(
    doctor_id: str = Body(..., example="D101"),
    doctor_notes: str = Body(..., example="Patient is stable")
):
    global latest_output

    result = connect_db.set_values_and_process(doctor_id, doctor_notes)
    latest_output = prompt_temp.generate_answer()

    return {
        "status": "success",
        "data": result,
        "generated_output": json.loads(latest_output)
    }
