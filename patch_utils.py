p = "utils_users.py"
text = open(p, encoding="utf-8").read()
lines = text.split("\n")
out = []

for line in lines:
    # rregullo hapësirat e rreshtit me len(parts) not in (6, 8)
    if "if len(parts) not in (6, 8)" in line:
        line = " " * 20 + line.strip()
    out.append(line)

    # shto dy rreshtat e rinj, vetëm nëse nuk ekzistojnë
    if line.strip() == "self.dv_threshold = float(header[5])" and "self.dv_exception" not in text:
        out.append(" " * 8 + "self.dv_exception = float(header[6]) if len(header) == 8 else self.dv_threshold")
        out.append(" " * 8 + "self.n_exceptions = int(float(header[7])) if len(header) == 8 else 0")

open(p, "w", encoding="utf-8").write("\n".join(out))
print("U rregullua")