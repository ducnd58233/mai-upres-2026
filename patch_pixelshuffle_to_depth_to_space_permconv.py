# # #!/usr/bin/env python3
# # """
# # Patch a quantized TFLite model to replace a PixelShuffle-like tail
# # (Transpose -> Reshape -> Transpose -> Reshape) with:
# #   DEPTH_TO_SPACE (NHWC) + TRANSPOSE (NHWC->NCHW)
# # AND permute the producer CONV_2D's output channels so the new tail is
# # bit-exact equivalent to the old PixelShuffle tail.

# # Why the permutation is needed:
# # - PyTorch PixelShuffle convention (common) arranges channels as: [c, by, bx]
# #   old_idx = c*r*r + by*r + bx
# # - TFLite DEPTH_TO_SPACE expects channels grouped by spatial offset first:
# #   new_idx = (by*r + bx)*C + c
# # So we permute the last conv's output channels from old_idx -> new_idx.

# # Usage:
# #   python patch_pixelshuffle_to_depth_to_space_permconv.py \
# #     --in_model antsr_720x1280_int8_nhwc.tflite \
# #     --out_model antsr_720x1280_int8_nhwc_d2s_equiv.tflite \
# #     --block 3
# # """
# # import argparse
# # import copy
# # import sys
# # from typing import List, Optional, Tuple

# # import numpy as np

# # # These come from TensorFlow package (present in your litert_export env)
# # # These come from TensorFlow package (present in your litert_export env)
# # from tensorflow.lite.python import schema_py_generated as schema_fb

# # try:
# #     # Most TF builds expose flatbuffer_utils here
# #     from tensorflow.lite.tools import flatbuffer_utils as fbu
# # except Exception:
# #     # Some older TF builds expose it here
# #     from tensorflow.lite.python import flatbuffer_utils as fbu


# # def _convert_buf_to_model(buf: bytes) -> "schema_fb.ModelT":
# #     # TF versions differ: convert_bytearray_to_object(buf) vs (buf, ModelT)
# #     try:
# #         return fbu.convert_bytearray_to_object(buf, schema_fb.ModelT)
# #     except TypeError:
# #         return fbu.convert_bytearray_to_object(buf)


# # def _model_to_buf(model: "schema_fb.ModelT") -> bytes:
# #     return fbu.convert_object_to_bytearray(model)


# # def _as_list_int(x) -> List[int]:
# #     if x is None:
# #         return []
# #     if isinstance(x, list):
# #         return [int(v) for v in x]
# #     if isinstance(x, np.ndarray):
# #         return [int(v) for v in x.tolist()]
# #     try:
# #         return [int(v) for v in list(x)]
# #     except Exception:
# #         return []


# # def _get_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int) -> Optional[bytes]:
# #     buf = model.buffers[buffer_idx]
# #     if buf is None or buf.data is None:
# #         return None
# #     data = buf.data
# #     if isinstance(data, (bytes, bytearray)):
# #         return bytes(data)
# #     # sometimes it's a list[int]
# #     return bytes(bytearray(data))


# # def _set_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int, data_bytes: bytes) -> None:
# #     model.buffers[buffer_idx].data = bytearray(data_bytes)


# # def _dtype_for_tensor(t: "schema_fb.TensorT"):
# #     # Map TFLite tensor types to numpy dtype
# #     tt = t.type
# #     if tt == schema_fb.TensorType.INT8:
# #         return np.int8
# #     if tt == schema_fb.TensorType.UINT8:
# #         return np.uint8
# #     if tt == schema_fb.TensorType.INT32:
# #         return np.int32
# #     if tt == schema_fb.TensorType.FLOAT32:
# #         return np.float32
# #     raise ValueError(f"Unsupported tensor type for buffer patching: {tt}")


# # def _permute_quant_params(q: "schema_fb.QuantizationParametersT", perm: List[int]) -> None:
# #     if q is None:
# #         return
# #     if q.scale is not None and len(q.scale) == len(perm):
# #         q.scale = [float(q.scale[i]) for i in perm]
# #     if q.zeroPoint is not None and len(q.zeroPoint) == len(perm):
# #         q.zeroPoint = [int(q.zeroPoint[i]) for i in perm]
# #     if q.min is not None and len(q.min) == len(perm):
# #         q.min = [float(q.min[i]) for i in perm]
# #     if q.max is not None and len(q.max) == len(perm):
# #         q.max = [float(q.max[i]) for i in perm]


# # def _find_tail_pixelshuffle(subg: "schema_fb.SubGraphT") -> Optional[Tuple[int, int, int, int]]:
# #     """
# #     Find last 4 ops pattern at graph tail:
# #       TRANSPOSE -> RESHAPE -> TRANSPOSE -> RESHAPE
# #     Return their operator indices (op0, op1, op2, op3) if found.
# #     """
# #     ops = subg.operators
# #     if len(ops) < 4:
# #         return None

# #     def op_builtin(op_idx: int) -> int:
# #         op = ops[op_idx]
# #         code = subg._model.operatorCodes[op.opcodeIndex]  # patched below
# #         # Some TF versions use builtinCode, some use deprecatedBuiltinCode
# #         bc = int(getattr(code, "builtinCode", 0))
# #         if bc == 0:
# #             bc = int(getattr(code, "deprecatedBuiltinCode", 0))
# #         return bc

# #     # We'll scan from the end to be robust
# #     for k in range(len(ops) - 4, -1, -1):
# #         bc0, bc1, bc2, bc3 = (op_builtin(k + i) for i in range(4))
# #         if (bc0 == schema_fb.BuiltinOperator.TRANSPOSE and
# #             bc1 == schema_fb.BuiltinOperator.RESHAPE and
# #             bc2 == schema_fb.BuiltinOperator.TRANSPOSE and
# #             bc3 == schema_fb.BuiltinOperator.RESHAPE):
# #             return (k, k + 1, k + 2, k + 3)
# #     return None


# # def _find_producer_op_idx(subg: "schema_fb.SubGraphT", tensor_id: int) -> Optional[int]:
# #     for i, op in enumerate(subg.operators):
# #         outs = _as_list_int(op.outputs)
# #         if tensor_id in outs:
# #             return i
# #     return None


# # def _build_perm(out_c: int, block: int) -> List[int]:
# #     """perm[new_idx] = old_idx mapping."""
# #     r = block
# #     perm = [0] * (out_c * r * r)
# #     for by in range(r):
# #         for bx in range(r):
# #             for c in range(out_c):
# #                 new_idx = (by * r + bx) * out_c + c
# #                 old_idx = c * (r * r) + by * r + bx
# #                 perm[new_idx] = old_idx
# #     return perm


# # def _permute_tensor_buffer(model: "schema_fb.ModelT", tensor: "schema_fb.TensorT", perm: List[int], axis: int = 0) -> None:
# #     """
# #     Permute a tensor's buffer along `axis` using `perm` (gather indices).
# #     Also permute per-channel quantization params if present.
# #     """
# #     shape = _as_list_int(getattr(tensor, "shape", None))
# #     if not shape:
# #         raise ValueError("Tensor has no shape; cannot permute.")
# #     if axis < 0:
# #         axis += len(shape)
# #     if axis != 0:
# #         # For our case we only need axis 0 (out_channels)
# #         raise ValueError("This patcher only supports axis=0 permutation.")

# #     if shape[0] != len(perm):
# #         raise ValueError(f"Perm length {len(perm)} != tensor dim0 {shape[0]} for tensor {tensor.name}")

# #     buf_bytes = _get_buf_bytes(model, tensor.buffer)
# #     if buf_bytes is None:
# #         raise ValueError(f"Tensor {tensor.name} has empty buffer; cannot permute.")
# #     np_dtype = _dtype_for_tensor(tensor)
# #     arr = np.frombuffer(buf_bytes, dtype=np_dtype).copy()
# #     arr = arr.reshape(shape)
# #     arr_new = arr[perm, ...].copy()

# #     _set_buf_bytes(model, tensor.buffer, arr_new.tobytes())

# #     # per-channel quantization: reorder vectors if they match dim0
# #     if tensor.quantization is not None:
# #         _permute_quant_params(tensor.quantization, perm)


# # def _make_quant_from(ref: "schema_fb.QuantizationParametersT") -> "schema_fb.QuantizationParametersT":
# #     if ref is None:
# #         return schema_fb.QuantizationParametersT()
# #     q = schema_fb.QuantizationParametersT()
# #     if ref.scale is not None:
# #         q.scale = [float(x) for x in list(ref.scale)]
# #     if ref.zeroPoint is not None:
# #         q.zeroPoint = [int(x) for x in list(ref.zeroPoint)]
# #     if ref.min is not None:
# #         q.min = [float(x) for x in list(ref.min)]
# #     if ref.max is not None:
# #         q.max = [float(x) for x in list(ref.max)]
# #     q.quantizedDimension = int(getattr(ref, "quantizedDimension", 0))
# #     return q


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--in_model", required=True)
# #     ap.add_argument("--out_model", required=True)
# #     ap.add_argument("--block", type=int, default=3)
# #     ap.add_argument("--no_permconv", action="store_true",
# #                     help="Only replace tail ops; DO NOT permute last conv weights (will change output if ordering differs).")
# #     args = ap.parse_args()

# #     with open(args.in_model, "rb") as f:
# #         buf = f.read()
# #     model = _convert_buf_to_model(buf)

# #     # Keep backref used by helper in _find_tail_pixelshuffle
# #     for sg in model.subgraphs:
# #         setattr(sg, "_model", model)

# #     if not model.subgraphs:
# #         raise SystemExit("Model has no subgraphs?")
# #     subg = model.subgraphs[0]
# #     tensors = subg.tensors
# #     ops = subg.operators

# #     tail = _find_tail_pixelshuffle(subg)
# #     if tail is None:
# #         raise SystemExit("Could not find tail pattern: TRANSPOSE->RESHAPE->TRANSPOSE->RESHAPE")
# #     op0_idx, op1_idx, op2_idx, op3_idx = tail
# #     op0, op1, op2, op3 = (ops[i] for i in tail)

# #     # We assume tail in/out tensors based on pattern:
# #     # op0 consumes tail_in (feature map before shuffle) and produces t_a
# #     # op3 produces the final output tensor
# #     tail_in_id = int(_as_list_int(op0.inputs)[0])
# #     tail_out_id = int(_as_list_int(op3.outputs)[0])

# #     in_t = tensors[tail_in_id]
# #     out_t = tensors[tail_out_id]

# #     in_shape = _as_list_int(getattr(in_t, "shape", None))
# #     out_shape = _as_list_int(getattr(out_t, "shape", None))

# #     print("Matched tail PixelShuffle-like pattern")
# #     print(f"  tail input : {tail_in_id} shape: {in_shape}")
# #     print(f"  tail output: {tail_out_id} shape: {out_shape}")

# #     if len(in_shape) != 4 or len(out_shape) != 4:
# #         raise SystemExit("Unexpected rank for tail tensors (expected 4D).")

# #     block = int(args.block)
# #     if in_shape[-1] % (block * block) != 0:
# #         raise SystemExit(f"Input channels {in_shape[-1]} not divisible by block^2={block*block}")
# #     out_c = in_shape[-1] // (block * block)

# #     # Preserve original output quantization metadata (some TF versions can disturb it during roundtrip)
# #     out_quant_orig = _make_quant_from(out_t.quantization)
# #     out_type_orig = int(out_t.type)

# #     # 1) (Optional) Permute the producer conv's output channels to match DEPTH_TO_SPACE ordering
# #     if not args.no_permconv:
# #         prod_idx = _find_producer_op_idx(subg, tail_in_id)
# #         if prod_idx is None:
# #             raise SystemExit(f"Could not find producer op for tensor id {tail_in_id}")
# #         prod_op = ops[prod_idx]
# #         code = model.operatorCodes[prod_op.opcodeIndex]
# #         bc = int(getattr(code, "builtinCode", 0)) or int(getattr(code, "deprecatedBuiltinCode", 0))
# #         if bc != schema_fb.BuiltinOperator.CONV_2D:
# #             print(f"WARNING: producer op for tail_in is not CONV_2D (builtinCode={bc}).")
# #             print("         The conv-weight permutation step is skipped. You can rerun with --no_permconv to silence.")
# #         else:
# #             prod_inputs = _as_list_int(prod_op.inputs)
# #             if len(prod_inputs) < 2:
# #                 raise SystemExit("CONV_2D has too few inputs?")
# #             w_tid = int(prod_inputs[1])
# #             b_tid = int(prod_inputs[2]) if len(prod_inputs) > 2 and prod_inputs[2] >= 0 else None

# #             w_t = tensors[w_tid]
# #             w_shape = _as_list_int(getattr(w_t, "shape", None))
# #             # TFLite CONV_2D weights are OHWI -> [out_ch, kh, kw, in_ch]
# #             if not w_shape or w_shape[0] != out_c * block * block:
# #                 print(f"WARNING: unexpected weight shape {w_shape}; expected first dim {out_c*block*block}.")
# #                 print("         Skipping conv-weight permutation.")
# #             else:
# #                 perm = _build_perm(out_c, block)
# #                 print(f"Permuting last CONV_2D weights for DEPTH_TO_SPACE (out_ch={len(perm)})")
# #                 _permute_tensor_buffer(model, w_t, perm, axis=0)
# #                 if b_tid is not None:
# #                     b_t = tensors[b_tid]
# #                     b_shape = _as_list_int(getattr(b_t, "shape", None))
# #                     if b_shape and b_shape[0] == len(perm):
# #                         _permute_tensor_buffer(model, b_t, perm, axis=0)
# #                     else:
# #                         print(f"WARNING: bias tensor shape {b_shape} doesn't match; skipping bias permutation.")

# #     # 2) Replace tail 4 ops with DEPTH_TO_SPACE + TRANSPOSE (keep output NCHW)
# #     # We'll reuse tail_in tensor id as input to DEPTH_TO_SPACE, create a new intermediate tensor (NHWC),
# #     # then write into the existing tail_out tensor via TRANSPOSE.
# #     # New NHWC shape:
# #     nhwc = [in_shape[0], in_shape[1] * block, in_shape[2] * block, out_c]
# #     print(f"  expected NHWC after D2S: {nhwc} (then transpose to output)")
# #     # Create new tensor id
# #     new_tid = len(tensors)
# #     new_t = schema_fb.TensorT()
# #     new_t.name = b"d2s_out"
# #     new_t.shape = list(nhwc)
# #     new_t.type = int(in_t.type)  # int8
# #     new_t.buffer = 0  # empty buffer (will be produced by op)
# #     new_t.isVariable = False
# #     # Quantization metadata: for reshape-like ops, keep same quant as input tensor
# #     new_t.quantization = _make_quant_from(in_t.quantization)
# #     tensors.append(new_t)

# #     # Create DEPTH_TO_SPACE op
# #     d2s_op = schema_fb.OperatorT()
# #     d2s_code = schema_fb.OperatorCodeT()
# #     d2s_code.builtinCode = schema_fb.BuiltinOperator.DEPTH_TO_SPACE
# #     model.operatorCodes.append(d2s_code)
# #     d2s_op.opcodeIndex = len(model.operatorCodes) - 1
# #     d2s_op.inputs = [tail_in_id]
# #     d2s_op.outputs = [new_tid]
# #     d2s_opt = schema_fb.DepthToSpaceOptionsT()
# #     d2s_opt.blockSize = block
# #     d2s_op.builtinOptionsType = schema_fb.BuiltinOptions.DepthToSpaceOptions
# #     d2s_op.builtinOptions = d2s_opt

# #     # Create TRANSPOSE op (NHWC->NCHW)
# #     tr_op = schema_fb.OperatorT()
# #     tr_code = schema_fb.OperatorCodeT()
# #     tr_code.builtinCode = schema_fb.BuiltinOperator.TRANSPOSE
# #     model.operatorCodes.append(tr_code)
# #     tr_op.opcodeIndex = len(model.operatorCodes) - 1

# #     # Create perm const tensor [0,3,1,2]
# #     perm = np.array([0, 3, 1, 2], dtype=np.int32)
# #     perm_buf = schema_fb.BufferT()
# #     perm_buf.data = bytearray(perm.tobytes())
# #     model.buffers.append(perm_buf)
# #     perm_buf_idx = len(model.buffers) - 1

# #     perm_tid = len(tensors)
# #     perm_t = schema_fb.TensorT()
# #     perm_t.name = b"perm_nhwc_to_nchw"
# #     perm_t.shape = [4]
# #     perm_t.type = schema_fb.TensorType.INT32
# #     perm_t.buffer = perm_buf_idx
# #     perm_t.isVariable = False
# #     tensors.append(perm_t)

# #     tr_op.inputs = [new_tid, perm_tid]
# #     tr_op.outputs = [tail_out_id]
# #     tr_op.builtinOptionsType = schema_fb.BuiltinOptions.TransposeOptions
# #     tr_op.builtinOptions = schema_fb.TransposeOptionsT()

# #     # Replace ops: keep prefix, drop old 4 ops, append new 2 ops
# #     print("Patch mode: 4 ops -> 2 ops (DEPTH_TO_SPACE + TRANSPOSE), output stays NCHW")
# #     ops_new = ops[:op0_idx] + [d2s_op, tr_op]
# #     subg.operators = ops_new

# #     # Restore original output tensor metadata explicitly
# #     out_t = tensors[tail_out_id]
# #     out_t.type = out_type_orig
# #     out_t.quantization = out_quant_orig

# #     out_buf = _model_to_buf(model)
# #     with open(args.out_model, "wb") as f:
# #         f.write(out_buf)
# #     print(f"Saved: {args.out_model}")
# #     print("Next: verify bit-exactness vs the original model with a random int8 input.")


# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except KeyboardInterrupt:
# #         sys.exit(130)




# #!/usr/bin/env python3
# """
# Patch a TFLite model tail that implements PixelShuffle via
# (Transpose -> Reshape -> Transpose -> Reshape) into a faster form using:
#   DEPTH_TO_SPACE (+ optional TRANSPOSE depending on original output layout)

# Also permute the producer CONV_2D's out-channels so the new tail is
# bit-exact equivalent to PixelShuffle convention.

# This version is hardened:
# - Detects whether the tensor before tail is NHWC or needs the first transpose.
# - Keeps suffix ops after the tail.
# - Keeps original output layout (NHWC or NCHW) to avoid extra transpose when not needed.

# Usage:
#   python patch_pixelshuffle_to_depth_to_space_permconv.py \
#     --in_model in.tflite --out_model out.tflite --block 3
# """
# import argparse
# import sys
# from typing import List, Optional, Tuple

# import numpy as np
# from tensorflow.lite.python import schema_py_generated as schema_fb

# try:
#     from tensorflow.lite.tools import flatbuffer_utils as fbu
# except Exception:
#     from tensorflow.lite.python import flatbuffer_utils as fbu


# def _convert_buf_to_model(buf: bytes) -> "schema_fb.ModelT":
#     try:
#         return fbu.convert_bytearray_to_object(buf, schema_fb.ModelT)
#     except TypeError:
#         return fbu.convert_bytearray_to_object(buf)


# def _model_to_buf(model: "schema_fb.ModelT") -> bytes:
#     return fbu.convert_object_to_bytearray(model)


# def _as_list_int(x) -> List[int]:
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return [int(v) for v in x]
#     if isinstance(x, np.ndarray):
#         return [int(v) for v in x.tolist()]
#     try:
#         return [int(v) for v in list(x)]
#     except Exception:
#         return []


# def _get_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int) -> Optional[bytes]:
#     buf = model.buffers[buffer_idx]
#     if buf is None or buf.data is None:
#         return None
#     data = buf.data
#     if isinstance(data, (bytes, bytearray)):
#         return bytes(data)
#     return bytes(bytearray(data))


# def _set_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int, data_bytes: bytes) -> None:
#     model.buffers[buffer_idx].data = bytearray(data_bytes)


# def _dtype_for_tensor(t: "schema_fb.TensorT"):
#     tt = t.type
#     if tt == schema_fb.TensorType.INT8:
#         return np.int8
#     if tt == schema_fb.TensorType.UINT8:
#         return np.uint8
#     if tt == schema_fb.TensorType.INT32:
#         return np.int32
#     if tt == schema_fb.TensorType.FLOAT32:
#         return np.float32
#     raise ValueError(f"Unsupported tensor type: {tt}")


# def _permute_quant_params(q: "schema_fb.QuantizationParametersT", perm: List[int]) -> None:
#     if q is None:
#         return
#     if q.scale is not None and len(q.scale) == len(perm):
#         q.scale = [float(q.scale[i]) for i in perm]
#     if q.zeroPoint is not None and len(q.zeroPoint) == len(perm):
#         q.zeroPoint = [int(q.zeroPoint[i]) for i in perm]
#     if q.min is not None and len(q.min) == len(perm):
#         q.min = [float(q.min[i]) for i in perm]
#     if q.max is not None and len(q.max) == len(perm):
#         q.max = [float(q.max[i]) for i in perm]


# def _permute_tensor_buffer(model: "schema_fb.ModelT", tensor: "schema_fb.TensorT", perm: List[int], axis: int = 0) -> None:
#     shape = _as_list_int(getattr(tensor, "shape", None))
#     if not shape:
#         raise ValueError("Tensor has no shape.")
#     if axis != 0:
#         raise ValueError("Only axis=0 is supported here.")
#     if shape[0] != len(perm):
#         raise ValueError(f"perm len {len(perm)} != shape[0] {shape[0]} for tensor {tensor.name}")

#     buf_bytes = _get_buf_bytes(model, tensor.buffer)
#     if buf_bytes is None:
#         raise ValueError(f"Tensor {tensor.name} has empty buffer; cannot permute.")
#     np_dtype = _dtype_for_tensor(tensor)
#     arr = np.frombuffer(buf_bytes, dtype=np_dtype).copy().reshape(shape)
#     arr_new = arr[perm, ...].copy()
#     _set_buf_bytes(model, tensor.buffer, arr_new.tobytes())

#     if tensor.quantization is not None:
#         _permute_quant_params(tensor.quantization, perm)


# def _find_tail_pixelshuffle(subg: "schema_fb.SubGraphT", model: "schema_fb.ModelT") -> Optional[Tuple[int, int, int, int]]:
#     ops = subg.operators
#     if len(ops) < 4:
#         return None

#     def builtin_code(op_idx: int) -> int:
#         op = ops[op_idx]
#         code = model.operatorCodes[op.opcodeIndex]
#         # do NOT treat 0 as "missing", 0 is a valid enum value
#         bc = getattr(code, "builtinCode", None)
#         if bc is None:
#             bc = getattr(code, "deprecatedBuiltinCode", 0)
#         return int(bc)

#     for k in range(len(ops) - 4, -1, -1):
#         bc0, bc1, bc2, bc3 = (builtin_code(k + i) for i in range(4))
#         if (bc0 == schema_fb.BuiltinOperator.TRANSPOSE and
#             bc1 == schema_fb.BuiltinOperator.RESHAPE and
#             bc2 == schema_fb.BuiltinOperator.TRANSPOSE and
#             bc3 == schema_fb.BuiltinOperator.RESHAPE):
#             return (k, k + 1, k + 2, k + 3)
#     return None


# def _find_producer_op_idx(subg: "schema_fb.SubGraphT", tensor_id: int) -> Optional[int]:
#     for i, op in enumerate(subg.operators):
#         outs = _as_list_int(op.outputs)
#         if tensor_id in outs:
#             return i
#     return None


# def _read_transpose_perm(model: "schema_fb.ModelT", subg: "schema_fb.SubGraphT", op: "schema_fb.OperatorT") -> Optional[List[int]]:
#     ins = _as_list_int(op.inputs)
#     if len(ins) < 2:
#         return None
#     perm_tid = ins[1]
#     t = subg.tensors[perm_tid]
#     b = _get_buf_bytes(model, t.buffer)
#     if b is None:
#         return None
#     arr = np.frombuffer(b, dtype=np.int32)
#     return [int(x) for x in arr.tolist()]


# def _build_perm(out_c: int, block: int) -> List[int]:
#     # perm[new_idx] = old_idx
#     r = block
#     perm = [0] * (out_c * r * r)
#     for by in range(r):
#         for bx in range(r):
#             for c in range(out_c):
#                 new_idx = (by * r + bx) * out_c + c
#                 old_idx = c * (r * r) + by * r + bx
#                 perm[new_idx] = old_idx
#     return perm


# def _make_quant_from(ref: "schema_fb.QuantizationParametersT") -> "schema_fb.QuantizationParametersT":
#     if ref is None:
#         return schema_fb.QuantizationParametersT()
#     q = schema_fb.QuantizationParametersT()
#     if ref.scale is not None:
#         q.scale = [float(x) for x in list(ref.scale)]
#     if ref.zeroPoint is not None:
#         q.zeroPoint = [int(x) for x in list(ref.zeroPoint)]
#     if ref.min is not None:
#         q.min = [float(x) for x in list(ref.min)]
#     if ref.max is not None:
#         q.max = [float(x) for x in list(ref.max)]
#     q.quantizedDimension = int(getattr(ref, "quantizedDimension", 0))
#     return q


# def _op_code(model: "schema_fb.ModelT", op: "schema_fb.OperatorT") -> int:
#     code = model.operatorCodes[op.opcodeIndex]
#     bc = getattr(code, "builtinCode", None)
#     if bc is None:
#         bc = getattr(code, "deprecatedBuiltinCode", 0)
#     return int(bc)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_model", required=True)
#     ap.add_argument("--out_model", required=True)
#     ap.add_argument("--block", type=int, default=3)
#     ap.add_argument("--no_permconv", action="store_true")
#     args = ap.parse_args()

#     with open(args.in_model, "rb") as f:
#         buf = f.read()
#     model = _convert_buf_to_model(buf)

#     if not model.subgraphs:
#         raise SystemExit("Model has no subgraphs.")
#     subg = model.subgraphs[0]
#     tensors = subg.tensors
#     ops = subg.operators

#     tail = _find_tail_pixelshuffle(subg, model)
#     if tail is None:
#         raise SystemExit("Could not find tail pattern: TRANSPOSE->RESHAPE->TRANSPOSE->RESHAPE")
#     op0_idx, op1_idx, op2_idx, op3_idx = tail
#     op0, op1, op2, op3 = (ops[i] for i in tail)

#     # Identify tensors around tail
#     pre_id  = int(_as_list_int(op0.inputs)[0])   # input to first TRANSPOSE
#     t0_id   = int(_as_list_int(op0.outputs)[0])  # output of first TRANSPOSE
#     out_id  = int(_as_list_int(op3.outputs)[0])  # final output

#     pre_t = tensors[pre_id]
#     t0_t  = tensors[t0_id]
#     out_t = tensors[out_id]

#     pre_shape = _as_list_int(getattr(pre_t, "shape", None))
#     t0_shape  = _as_list_int(getattr(t0_t,  "shape", None))
#     out_shape = _as_list_int(getattr(out_t, "shape", None))

#     print("Matched tail pattern at ops:", tail)
#     print("  pre (op0.in)  tid:", pre_id, "shape:", pre_shape)
#     print("  t0  (op0.out) tid:", t0_id,  "shape:", t0_shape)
#     print("  out (op3.out) tid:", out_id, "shape:", out_shape)

#     block = int(args.block)

#     def is_nhwc(shape):
#         return len(shape) == 4 and shape[-1] > 0

#     # Choose DEPTH_TO_SPACE input tensor:
#     # Prefer a tensor that is clearly NHWC with channels divisible by r^2.
#     d2s_in_id = None
#     d2s_in_shape = None

#     if len(pre_shape) == 4 and (pre_shape[-1] % (block * block) == 0):
#         d2s_in_id = pre_id
#         d2s_in_shape = pre_shape
#         replace_start = op0_idx  # replace from op0
#         print("Using pre tensor as D2S input (appears NHWC).")
#     elif len(t0_shape) == 4 and (t0_shape[-1] % (block * block) == 0):
#         d2s_in_id = t0_id
#         d2s_in_shape = t0_shape
#         replace_start = op1_idx  # keep op0, replace from op1..op3
#         print("Using op0 output as D2S input (op0 likely layout transpose). Keeping op0.")
#     else:
#         raise SystemExit("Cannot find an NHWC tensor with channels divisible by r^2 for D2S input.")

#     if len(d2s_in_shape) != 4:
#         raise SystemExit("Unexpected D2S input rank (expected 4D).")
#     if d2s_in_shape[-1] % (block * block) != 0:
#         raise SystemExit("D2S input channels not divisible by r^2.")

#     inN, inH, inW, inCh = d2s_in_shape
#     out_c = inCh // (block * block)

#     # Detect desired output layout from existing output tensor shape
#     # NHWC output: shape[-1]==3; NCHW output: shape[1]==3
#     want_nhwc_out = (len(out_shape) == 4 and out_shape[-1] == 3)
#     want_nchw_out = (len(out_shape) == 4 and out_shape[1] == 3)
#     if not (want_nhwc_out or want_nchw_out):
#         # fallback: keep old out layout by heuristic
#         want_nhwc_out = True
#         print("[WARN] Cannot infer output layout, defaulting to NHWC.")

#     # Save output tensor metadata to restore
#     out_quant_orig = _make_quant_from(out_t.quantization)
#     out_type_orig = int(out_t.type)

#     # 1) Permute producer CONV_2D weights (producer should feed pre_id, not d2s_in_id)
#     if not args.no_permconv:
#         prod_idx = _find_producer_op_idx(subg, pre_id)
#         if prod_idx is None:
#             print("[WARN] Could not find producer op for pre tensor; skipping permconv.")
#         else:
#             prod_op = ops[prod_idx]
#             bc = _op_code(model, prod_op)
#             if bc != schema_fb.BuiltinOperator.CONV_2D:
#                 print(f"[WARN] Producer of pre tensor is not CONV_2D (builtinCode={bc}); skipping permconv.")
#             else:
#                 prod_inputs = _as_list_int(prod_op.inputs)
#                 if len(prod_inputs) < 2:
#                     raise SystemExit("CONV_2D has too few inputs.")
#                 w_tid = int(prod_inputs[1])
#                 b_tid = int(prod_inputs[2]) if len(prod_inputs) > 2 and prod_inputs[2] >= 0 else None

#                 w_t = tensors[w_tid]
#                 w_shape = _as_list_int(getattr(w_t, "shape", None))
#                 # TFLite CONV_2D weights are OHWI: [out_ch, kh, kw, in_ch]
#                 expected = out_c * block * block
#                 if not w_shape or w_shape[0] != expected:
#                     print(f"[WARN] Weight shape {w_shape} unexpected; expected dim0={expected}. Skipping permconv.")
#                 else:
#                     perm = _build_perm(out_c, block)
#                     print(f"Permuting CONV_2D weights: out_ch={len(perm)}")
#                     _permute_tensor_buffer(model, w_t, perm, axis=0)
#                     if b_tid is not None:
#                         b_t = tensors[b_tid]
#                         b_shape = _as_list_int(getattr(b_t, "shape", None))
#                         if b_shape and b_shape[0] == len(perm):
#                             _permute_tensor_buffer(model, b_t, perm, axis=0)
#                         else:
#                             print("[WARN] Bias shape mismatch; skipping bias permutation.")

#     # 2) Build new ops: DEPTH_TO_SPACE (+ optional TRANSPOSE)
#     # Create operator codes with version=1
#     def add_opcode(builtin):
#         oc = schema_fb.OperatorCodeT()
#         oc.builtinCode = builtin
#         oc.version = 1
#         model.operatorCodes.append(oc)
#         return len(model.operatorCodes) - 1

#     # Create DEPTH_TO_SPACE op
#     d2s_op = schema_fb.OperatorT()
#     d2s_op.opcodeIndex = add_opcode(schema_fb.BuiltinOperator.DEPTH_TO_SPACE)
#     d2s_op.inputs = [d2s_in_id]

#     nhwc_shape = [inN, inH * block, inW * block, out_c]

#     if want_nhwc_out:
#         # Directly write to existing out tensor (no extra transpose)
#         d2s_op.outputs = [out_id]
#         d2s_opt = schema_fb.DepthToSpaceOptionsT()
#         d2s_opt.blockSize = block
#         d2s_op.builtinOptionsType = schema_fb.BuiltinOptions.DepthToSpaceOptions
#         d2s_op.builtinOptions = d2s_opt
#         new_ops_seq = [d2s_op]
#         print("Patching tail to: DEPTH_TO_SPACE (NHWC output)")
#         # Ensure output tensor shape matches NHWC
#         out_t.shape = list(nhwc_shape)
#     else:
#         # Need NHWC intermediate then TRANSPOSE to NCHW output
#         # Create new intermediate tensor
#         new_tid = len(tensors)
#         mid_t = schema_fb.TensorT()
#         mid_t.name = b"d2s_out"
#         mid_t.shape = list(nhwc_shape)
#         mid_t.type = int(tensors[d2s_in_id].type)
#         mid_t.buffer = 0
#         mid_t.isVariable = False
#         mid_t.quantization = _make_quant_from(tensors[d2s_in_id].quantization)
#         tensors.append(mid_t)

#         d2s_op.outputs = [new_tid]
#         d2s_opt = schema_fb.DepthToSpaceOptionsT()
#         d2s_opt.blockSize = block
#         d2s_op.builtinOptionsType = schema_fb.BuiltinOptions.DepthToSpaceOptions
#         d2s_op.builtinOptions = d2s_opt

#         # Create perm tensor [0,3,1,2]
#         perm_arr = np.array([0, 3, 1, 2], dtype=np.int32)
#         perm_buf = schema_fb.BufferT()
#         perm_buf.data = bytearray(perm_arr.tobytes())
#         model.buffers.append(perm_buf)
#         perm_buf_idx = len(model.buffers) - 1

#         perm_tid = len(tensors)
#         perm_t = schema_fb.TensorT()
#         perm_t.name = b"perm_nhwc_to_nchw"
#         perm_t.shape = [4]
#         perm_t.type = schema_fb.TensorType.INT32
#         perm_t.buffer = perm_buf_idx
#         perm_t.isVariable = False
#         tensors.append(perm_t)

#         tr_op = schema_fb.OperatorT()
#         tr_op.opcodeIndex = add_opcode(schema_fb.BuiltinOperator.TRANSPOSE)
#         tr_op.inputs = [new_tid, perm_tid]
#         tr_op.outputs = [out_id]
#         tr_op.builtinOptionsType = schema_fb.BuiltinOptions.TransposeOptions
#         tr_op.builtinOptions = schema_fb.TransposeOptionsT()

#         new_ops_seq = [d2s_op, tr_op]
#         print("Patching tail to: DEPTH_TO_SPACE + TRANSPOSE (output NCHW)")
#         # For NCHW output tensor shape should be [N,3,H,W]
#         out_t.shape = [inN, 3, inH * block, inW * block]

#     # Replace ops in range [replace_start .. op3_idx] with new_ops_seq, keep suffix ops
#     ops_new = ops[:replace_start] + new_ops_seq + ops[op3_idx + 1:]
#     subg.operators = ops_new

#     # Restore output tensor type/quant params explicitly
#     out_t.type = out_type_orig
#     out_t.quantization = out_quant_orig

#     out_buf = _model_to_buf(model)
#     with open(args.out_model, "wb") as f:
#         f.write(out_buf)

#     print("Saved:", args.out_model)
#     print("Next: verify equivalence by running original vs patched on random int8 inputs.")


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         sys.exit(130)






#!/usr/bin/env python3
import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

try:
    from tensorflow.lite.tools import flatbuffer_utils as fbu
except Exception:
    from tensorflow.lite.python import flatbuffer_utils as fbu


def _convert_buf_to_model(buf: bytes) -> "schema_fb.ModelT":
    try:
        return fbu.convert_bytearray_to_object(buf, schema_fb.ModelT)
    except TypeError:
        return fbu.convert_bytearray_to_object(buf)

def _model_to_buf(model: "schema_fb.ModelT") -> bytes:
    return fbu.convert_object_to_bytearray(model)

def _as_list_int(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    try:
        return [int(v) for v in list(x)]
    except Exception:
        return []

def _get_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int) -> Optional[bytes]:
    buf = model.buffers[buffer_idx]
    if buf is None or buf.data is None:
        return None
    data = buf.data
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return bytes(bytearray(data))

def _set_buf_bytes(model: "schema_fb.ModelT", buffer_idx: int, data_bytes: bytes) -> None:
    model.buffers[buffer_idx].data = bytearray(data_bytes)

def _dtype_for_tensor(t: "schema_fb.TensorT"):
    tt = t.type
    if tt == schema_fb.TensorType.INT8:
        return np.int8
    if tt == schema_fb.TensorType.UINT8:
        return np.uint8
    if tt == schema_fb.TensorType.INT32:
        return np.int32
    if tt == schema_fb.TensorType.FLOAT32:
        return np.float32
    raise ValueError(f"Unsupported tensor type: {tt}")

def _permute_quant_params(q: "schema_fb.QuantizationParametersT", perm: List[int]) -> None:
    if q is None:
        return
    if q.scale is not None and len(q.scale) == len(perm):
        q.scale = [float(q.scale[i]) for i in perm]
    if q.zeroPoint is not None and len(q.zeroPoint) == len(perm):
        q.zeroPoint = [int(q.zeroPoint[i]) for i in perm]
    if q.min is not None and len(q.min) == len(perm):
        q.min = [float(q.min[i]) for i in perm]
    if q.max is not None and len(q.max) == len(perm):
        q.max = [float(q.max[i]) for i in perm]

def _permute_tensor_buffer(model: "schema_fb.ModelT", tensor: "schema_fb.TensorT", perm: List[int]) -> None:
    shape = _as_list_int(getattr(tensor, "shape", None))
    if not shape:
        raise ValueError("Tensor has no shape; cannot permute.")
    if shape[0] != len(perm):
        raise ValueError(f"perm len {len(perm)} != tensor dim0 {shape[0]} ({tensor.name})")

    buf_bytes = _get_buf_bytes(model, tensor.buffer)
    if buf_bytes is None:
        raise ValueError(f"Tensor {tensor.name} has empty buffer; cannot permute.")
    np_dtype = _dtype_for_tensor(tensor)
    arr = np.frombuffer(buf_bytes, dtype=np_dtype).copy().reshape(shape)
    arr_new = arr[perm, ...].copy()

    _set_buf_bytes(model, tensor.buffer, arr_new.tobytes())

    if tensor.quantization is not None:
        _permute_quant_params(tensor.quantization, perm)

def _build_perm(out_c: int, block: int) -> List[int]:
    r = block
    perm = [0] * (out_c * r * r)
    for by in range(r):
        for bx in range(r):
            for c in range(out_c):
                new_idx = (by * r + bx) * out_c + c
                old_idx = c * (r * r) + by * r + bx
                perm[new_idx] = old_idx
    return perm

def _find_tail_pixelshuffle(subg: "schema_fb.SubGraphT") -> Optional[Tuple[int, int, int, int]]:
    ops = subg.operators
    if len(ops) < 4:
        return None

    def op_builtin(op_idx: int) -> int:
        op = ops[op_idx]
        code = subg._model.operatorCodes[op.opcodeIndex]
        bc = int(getattr(code, "builtinCode", 0))
        if bc == 0:
            bc = int(getattr(code, "deprecatedBuiltinCode", 0))
        return bc

    for k in range(len(ops) - 4, -1, -1):
        bc0, bc1, bc2, bc3 = (op_builtin(k + i) for i in range(4))
        if (bc0 == schema_fb.BuiltinOperator.TRANSPOSE and
            bc1 == schema_fb.BuiltinOperator.RESHAPE and
            bc2 == schema_fb.BuiltinOperator.TRANSPOSE and
            bc3 == schema_fb.BuiltinOperator.RESHAPE):
            return (k, k + 1, k + 2, k + 3)
    return None

def _find_producer_op_idx(subg: "schema_fb.SubGraphT", tensor_id: int) -> Optional[int]:
    for i, op in enumerate(subg.operators):
        outs = _as_list_int(op.outputs)
        if tensor_id in outs:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--no_permconv", action="store_true")
    args = ap.parse_args()

    with open(args.in_model, "rb") as f:
        buf = f.read()
    model = _convert_buf_to_model(buf)

    for sg in model.subgraphs:
        setattr(sg, "_model", model)

    subg = model.subgraphs[0]
    tensors = subg.tensors
    ops = subg.operators

    tail = _find_tail_pixelshuffle(subg)
    if tail is None:
        raise SystemExit("Could not find tail pattern: TRANSPOSE->RESHAPE->TRANSPOSE->RESHAPE")
    op0_idx, op1_idx, op2_idx, op3_idx = tail
    op0, op3 = ops[op0_idx], ops[op3_idx]

    tail_in_id = int(_as_list_int(op0.inputs)[0])
    tail_out_id = int(_as_list_int(op3.outputs)[0])

    in_t = tensors[tail_in_id]
    out_t = tensors[tail_out_id]

    in_shape = _as_list_int(getattr(in_t, "shape", None))
    out_shape_old = _as_list_int(getattr(out_t, "shape", None))

    print("Matched tail PixelShuffle-like pattern")
    print(f"  tail input : {tail_in_id} shape: {in_shape}")
    print(f"  tail output: {tail_out_id} old shape: {out_shape_old}")

    if len(in_shape) != 4:
        raise SystemExit("Expected 4D tail input tensor.")
    block = int(args.block)
    if in_shape[-1] % (block * block) != 0:
        raise SystemExit(f"Input channels {in_shape[-1]} not divisible by block^2={block*block}")

    out_c = in_shape[-1] // (block * block)

    # 1) Permute producer conv output channels to match DepthToSpace ordering
    if not args.no_permconv:
        prod_idx = _find_producer_op_idx(subg, tail_in_id)
        if prod_idx is None:
            raise SystemExit(f"Could not find producer op for tensor {tail_in_id}")
        prod_op = ops[prod_idx]
        code = model.operatorCodes[prod_op.opcodeIndex]
        bc = int(getattr(code, "builtinCode", 0)) or int(getattr(code, "deprecatedBuiltinCode", 0))
        if bc != schema_fb.BuiltinOperator.CONV_2D:
            print(f"WARNING: producer op is not CONV_2D (builtinCode={bc}); skip permute.")
        else:
            prod_inputs = _as_list_int(prod_op.inputs)
            w_tid = int(prod_inputs[1])
            b_tid = int(prod_inputs[2]) if len(prod_inputs) > 2 and prod_inputs[2] >= 0 else None

            w_t = tensors[w_tid]
            w_shape = _as_list_int(getattr(w_t, "shape", None))  # OHWI
            expect0 = out_c * block * block
            if not w_shape or w_shape[0] != expect0:
                print(f"WARNING: unexpected weight shape {w_shape}; expected dim0={expect0}; skip permute.")
            else:
                perm = _build_perm(out_c, block)
                print(f"Permuting last CONV_2D out-channels: {len(perm)}")
                _permute_tensor_buffer(model, w_t, perm)
                if b_tid is not None:
                    b_t = tensors[b_tid]
                    b_shape = _as_list_int(getattr(b_t, "shape", None))
                    if b_shape and b_shape[0] == len(perm):
                        _permute_tensor_buffer(model, b_t, perm)
                    else:
                        print(f"WARNING: bias shape {b_shape} mismatch; skip bias permute.")

    # 2) Replace tail with DEPTH_TO_SPACE only (NHWC output)
    nhwc = [in_shape[0], in_shape[1] * block, in_shape[2] * block, out_c]
    print(f"New output NHWC shape will be: {nhwc}")

    d2s_code = schema_fb.OperatorCodeT()
    d2s_code.builtinCode = schema_fb.BuiltinOperator.DEPTH_TO_SPACE
    model.operatorCodes.append(d2s_code)

    d2s_op = schema_fb.OperatorT()
    d2s_op.opcodeIndex = len(model.operatorCodes) - 1
    d2s_op.inputs = [tail_in_id]
    d2s_op.outputs = [tail_out_id]

    d2s_opt = schema_fb.DepthToSpaceOptionsT()
    d2s_opt.blockSize = block
    d2s_op.builtinOptionsType = schema_fb.BuiltinOptions.DepthToSpaceOptions
    d2s_op.builtinOptions = d2s_opt

    subg.operators = ops[:op0_idx] + [d2s_op]

    out_t = tensors[tail_out_id]
    out_t.shape = list(nhwc)

    out_buf = _model_to_buf(model)
    with open(args.out_model, "wb") as f:
        f.write(out_buf)

    print("Saved:", args.out_model)
    print("Output is NHWC. Use expect_out=nhwc/auto in your eval script.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
