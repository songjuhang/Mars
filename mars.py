import sys
from engine.engine import MarsEngine


if __name__ == "__main__":
    mode = "pipe"
    nobuf = False

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"

    MarsEngine(
        mode=mode,
        # cfgname="c1.nano.full",
        # cfgname="c1.nano.full.cuda@0",
        # cfgname="c1.nano.teacher",
        cfgname="c1.nano.distillation",
        root="/workspace/mars12", # 注意项目运行root不要放在代码路径下
        nobuf=nobuf,
    ).run()
