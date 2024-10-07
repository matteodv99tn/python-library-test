import sys
import os
import subprocess


def figlet(text: str) -> str:

    def fallback(text: str) -> str:
        n_chars = 80
        borders = "+" + "-" * (n_chars - 2) + "+"
        text = f"|{text.center(n_chars-2)}|"
        out = borders + "\n" + text + "\n" + borders
        return out

    try:
        ret = subprocess.check_output(["figlet", text])
        return ret.decode("utf-8").rstrip()
    except Exception as e:
        return fallback(text)


def heading(text: str, *args):
    head_text = figlet(text)
    head_split = head_text.split("\n")
    max_chars = max(head_split, key=len)
    n_lines = len(head_split)
    n_delta = n_lines - len(args)
    for i in range(n_lines):
        line = head_split[i].rstrip().ljust(len(max_chars))
        if i >= n_delta:
            line += " " + args[i-n_delta]
        print(line)
