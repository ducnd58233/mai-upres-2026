#!/usr/bin/env python3
import argparse, collections
import tflite_runtime.interpreter as tflite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    itp = tflite.Interpreter(model_path=args.model, num_threads=args.threads)
    itp.allocate_tensors()

    print("INPUT:", [(i["name"], i["shape"], i["dtype"], i.get("quantization")) for i in itp.get_input_details()])
    print("OUTPUT:", [(o["name"], o["shape"], o["dtype"], o.get("quantization")) for o in itp.get_output_details()])

    hist = collections.Counter(str(t["dtype"]) for t in itp.get_tensor_details())
    print("dtype histogram:", dict(hist))

if __name__ == "__main__":
    main()
