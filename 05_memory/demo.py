"""A memory can correct an old association without relearning a network."""
import numpy as np
import cadence as cd

names = ("red", "blue")
key = np.array([[1.0, 0.0]])
red, blue = np.eye(2)

for rule in ("hebb", "delta"):
    memory = cd.FastSeams(np.arange(2), np.arange(2, 4), rule=rule)
    for _ in range(100):
        memory.observe(key, red[None])
    memory.observe(key, blue[None])
    answer = memory.recall(key)[0]
    print(f"{rule:5s}: {names[answer.argmax()]}  (readback: {answer})")

print("A dictionary also returns blue. Try the benchmark for correlated keys and old memories.")
