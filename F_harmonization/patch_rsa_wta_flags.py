#!/usr/bin/env python3
"""Idempotent patch for 05_stats_harmony.py: add --rsa AND --wta flags + use them.
Declares globals as the FIRST statement in main(), reassigns from args after
parse_args. Safe to run repeatedly (strips prior attempts first)."""
import re, sys, subprocess
P = "/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/05_stats_harmony.py"
src = open(P).read()

# 1. strip all prior attempts (rsa or wta, dupes, typos, misplaced globals)
def keep(l):
    s = l.strip()
    if "ap.add_argument('--rsa'" in l: return False
    if "ap.add_argument('--wta'" in l: return False
    if "global RSA_CSV" in l or "gloyal RSA_CSV" in l: return False
    if "global WTA_CSV" in l: return False
    if s in ("RSA_CSV = args.rsa", "WTA_CSV = args.wta"): return False
    return True
src = "\n".join(l for l in src.splitlines() if keep(l))

# 2. globals as the first line inside main()
src = re.sub(r"(\ndef main\(\):\n)",
             r"\1    global RSA_CSV, WTA_CSV\n", src, count=1)

# 3. add both flags after --univar
flags = ("    ap.add_argument('--rsa', default=str(RSA_CSV))   # harmonized RSA input\n"
         "    ap.add_argument('--wta', default=str(WTA_CSV))   # harmonized WTA input")
src = re.sub(r"(\n[ \t]*ap\.add_argument\('--univar'[^\n]*\n)",
             r"\1" + flags + "\n", src, count=1)

# 4. reassign both right after parse_args
src = re.sub(r"(\n[ \t]*args = ap\.parse_args\(\)\n)",
             r"\1    RSA_CSV = args.rsa\n    WTA_CSV = args.wta\n", src, count=1)

open(P, "w").write(src)

print("global decls:", src.count("global RSA_CSV, WTA_CSV"), "(want 1)")
print("--rsa flags :", src.count("ap.add_argument('--rsa'"), "(want 1)")
print("--wta flags :", src.count("ap.add_argument('--wta'"), "(want 1)")
print("rsa reassign:", src.count("RSA_CSV = args.rsa"), "(want 1)")
print("wta reassign:", src.count("WTA_CSV = args.wta"), "(want 1)")
r = subprocess.run([sys.executable, "-m", "py_compile", P], capture_output=True, text=True)
print("syntax:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr)