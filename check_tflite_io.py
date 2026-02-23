#!/usr/bin/env python3
import argparse, collections
import tflite_runtime.interpreter as tflite

def _qstr(d):
    q = d.get("quantization", None)
    qp = d.get("quantization_parameters", None)
    return {"quantization": q, "quantization_parameters": qp}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    itp = tflite.Interpreter(model_path=args.model, num_threads=args.threads)
    itp.allocate_tensors()

    ins = itp.get_input_details()
    outs = itp.get_output_details()

    print("INPUT:")
    for i in ins:
        print(" ", i["name"], i["shape"], i["dtype"], _qstr(i))

    print("OUTPUT:")
    for o in outs:
        print(" ", o["name"], o["shape"], o["dtype"], _qstr(o))

    hist = collections.Counter(str(t["dtype"]) for t in itp.get_tensor_details())
    print("dtype histogram:", dict(hist))

if __name__ == "__main__":
    main()