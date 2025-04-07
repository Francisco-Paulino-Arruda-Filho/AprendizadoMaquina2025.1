import re

my_string = "temperatura: 75.5F"

# Use regex to extract the number and unit from the string
temp = re.findall(r"(\d+\.?\d*)([CF])", my_string)
print(float(temp.group[0]))  # Output: 75.5