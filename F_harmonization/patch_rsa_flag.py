#!/usr/bin/env python3
"""Idempotent patch for 05_stats_harmony.py: add --rsa flag + use it.
Declares `global RSA_CSV` as the FIRST statement in main() to avoid
use-before-global, then reassigns from args.rsa after parse_args.
Safe to run repeatedly."""
import re, sys, subprocess
P = "/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/05_stats_harmony.py"
src = open(P).read()

# 1. strip all prior attempts (dupes, typos, misplaced globals)
src = "\n".join(l for l in src.splitlines()
                if "ap.add_argument('--rsa'" not in l
                and "global RSA_CSV" not in l
                and "gloyal RSA_CSV" not in l
                and l.strip() != "RSA_CSV = args.rsa")

# 2. make `global RSA_CSV` the first line inside main()
src = re.sub(r"(\ndef main\(\):\n)",
             r"\1    global RSA_CSV\n", src, count=1)

# 3. add the --rsa flag after --univar
flag = "    ap.add_argument('--rsa', default=str(RSA_CSV))   # harmonized RSA input"
src = re.sub(r"(\n[ \t]*ap\.add_argument\('--univar'[^\n]*\n)",
             r"\1" + flag + "\n", src, count=1)

# 4. reassign RSA_CSV from args right after parse_args
src = re.sub(r"(\n[ \t]*args = ap\.parse_args\(\)\n)",
             r"\1    RSA_CSV = args.rsa\n", src, count=1)

open(P, "w").write(src)

print("global decls:", src.count("global RSA_CSV"), "(want 1)")
print("--rsa flags :", src.count("ap.add_argument('--rsa'"), "(want 1)")
print("reassigns   :", src.count("RSA_CSV = args.rsa"), "(want 1)")
r = subprocess.run([sys.executable, "-m", "py_compile", P], capture_output=True, text=True)
print("syntax:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr)