import os

print(os.path.dirname(os.path.abspath(__file__)))
print(type(os.path.dirname(os.path.abspath(__file__))))

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"current_dir = {current_dir}")

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
print(f"parent_dir = {parent_dir}")