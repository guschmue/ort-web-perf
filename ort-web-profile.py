import argparse
import logging
import re
import sys

# pylint: disable=missing-docstring
# [profiling] kernel "132726896|Gather" execution time: 8192 ns

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('strings', metavar='N', type=str, nargs='+', help='strings')
    parser.add_argument('--name')
    args = parser.parse_args()
    return args


def process_file(fname, fp, pfp):
    exp1 = r"^.*LOG: '(\{.*)'}\",.*$"
    exp2 = r"^.*(\{\"cat.*\},).*$"
    # exp3 = r"^.*kernel\s\"\d+\|\[(\w+)\]\s([/\.\w]+)\"\sexecution\stime:\s(\d+).*$"
    exp3 = r"^.*kernel\s\"\d+\|\[(\w+)\]\s([/\.\w]+)\"\s.*,\sexecution\stime:\s(\d+).*$"
    # [profiling] kernel "92210584|[Mul] /Mul_16" input[0]: [] | float32, input[1]: [1,4] | float32, output[0]: [1,4] | float32, execution time: 18560 ns
    fp.write("[\n")
    pfp.write("[\n")
    with open(fname, "r") as f:
        for line in f:
            # look for onnxruntime trace
            m = re.match(exp1, line)
            if m:
                fp.write(m.group(1) + "\n")
            else:
                m = re.match(exp2, line)
                if m:
                    fp.write(m.group(1) + "\n")
            # look for webgpu timestamps
            m = re.match(exp3, line)
            if m:
                pfp.write(f"{{\"op\": \"{m.group(1)}\", \"name\": \"{m.group(2)}\", \"time\": {float(m.group(3)) / 1000.}}},\n")
    fp.write("{}]\n")
    pfp.write("{\"op\": \"0\", \"name\": \"0\", \"time\": 0}\n]\n")


def main():
    args = get_args()
    for fname in args.strings:
        outname = fname.replace(".log", "")
        with open(outname + ".json", "w") as fp:
            with open(outname + "_gpu.json", "w") as pfp:
                process_file(fname, fp, pfp)


if __name__ == "__main__":
    sys.exit(main())
