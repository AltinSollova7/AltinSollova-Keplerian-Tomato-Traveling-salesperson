import sys
from kttsp_block import run

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "easy.kttsp"
    run(path, "small2", mode="earliest", B=1, time_limit=900)