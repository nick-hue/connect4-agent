
with open("counters.txt", "r") as f:
    data = f.readline().split(",")

p1_data, p2_data = data

print(f"P1 wins : {p1_data.split(':')[1]}")
print(f"P2 wins : {p2_data.split(':')[1]}")

