from transformers import pipeline

MODEL_ID = "google/gemma-4-E2B-it"

pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)