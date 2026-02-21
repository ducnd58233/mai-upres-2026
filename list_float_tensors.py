#!/usr/bin/env python3
import argparse
import tflite_runtime.interpreter as tflite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    itp = tflite.Interpreter(model_path=args.model)
    itp.allocate_tensors()

    floats = []
    for t in itp.get_tensor_details():
        if str(t["dtype"]) == "<class 'numpy.float32'>":
            floats.append((t["index"], t["name"], t.get("shape", None)))

    print("float32 tensors:", len(floats))
    for idx, name, shape in floats:
        print(f"  idx={idx:4d} shape={shape} name={name}")

if __name__ == "__main__":
    main()
