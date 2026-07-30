from src.utils import save_object, load_object

obj = {
    "course": "Machine Learning",
    "episode": 5
}

save_object("artifacts/sample.pkl", obj)

loaded = load_object("artifacts/sample.pkl")

print(loaded)