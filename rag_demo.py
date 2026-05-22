"""兼容入口 → 单条评估（与教程 rag_demo 同名）。"""

from rag_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["single"]))
