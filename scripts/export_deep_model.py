#!/usr/bin/env python3
"""
Export / Fetch Helper for Deep Palm Embedding Model (ONNX).
Supports creating a tiny deterministic dummy ONNX model for tests/scaffolding
without external network access, or pointing to a real ONNX file for production/dev.
"""

import argparse
import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_dummy_onnx(output_path: str, input_size: int = 128, in_channels: int = 3, feature_dim: int = 512):
    """
    Creates a minimal valid ONNX graph using onnx package if installed,
    or writes a stub file if onnx is unavailable.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        import onnx
        from onnx import helper, TensorProto

        # Input: (1, in_channels, input_size, input_size) float32
        # Output: (1, feature_dim) float32
        inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_channels, input_size, input_size])
        out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, feature_dim])

        # Global average pool -> flatten -> gemm / dense projection
        gap_out = helper.make_tensor_value_info('gap_out', TensorProto.FLOAT, [1, in_channels, 1, 1])
        flat_out = helper.make_tensor_value_info('flat_out', TensorProto.FLOAT, [1, in_channels])

        node_pool = helper.make_node('GlobalAveragePool', ['input'], ['gap_out'])
        node_flat = helper.make_node('Flatten', ['gap_out'], ['flat_out'])

        # Weight matrix: (in_channels, feature_dim)
        w_data = np.ones((in_channels, feature_dim), dtype=np.float32) / np.sqrt(feature_dim * in_channels)
        w_tensor = helper.make_tensor('W', TensorProto.FLOAT, [in_channels, feature_dim], w_data.flatten().tolist())

        node_gemm = helper.make_node('Gemm', ['flat_out', 'W'], ['output'], alpha=1.0, beta=0.0)

        graph = helper.make_graph(
            [node_pool, node_flat, node_gemm],
            'dummy_palm_embedder',
            [inp],
            [out],
            [w_tensor]
        )

        model = helper.make_model(graph, producer_name='palmpay_export')
        onnx.save(model, output_path)
        print(f"[Export] Created valid dummy ONNX model at: {output_path}")
        return True
    except ImportError:
        # If onnx library is not installed in the environment, create an informative marker file
        with open(output_path, "wb") as f:
            f.write(b"PALMPAY_STUB_ONNX_MODEL_FILE")
        print(f"[Export] Created stub ONNX marker file at: {output_path} (onnx python package not installed)")
        return False


def main():
    parser = argparse.ArgumentParser(description="PalmPay Deep Model Export/Setup Helper")
    parser.add_argument("--output", type=str, default=config.DEEP_MODEL_PATH, help="Path to save the ONNX model")
    parser.add_argument("--input-size", type=int, default=config.DEEP_INPUT_SIZE, help="Input spatial dimension (e.g. 128)")
    parser.add_argument("--feature-dim", type=int, default=config.DEEP_FEATURE_DIM, help="Feature embedding dimension (e.g. 512)")
    parser.add_argument("--from-file", type=str, default=None, help="Copy an existing real ONNX model into place")
    args = parser.parse_args()

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(config.BASE_DIR, out_path)

    if args.from_file:
        if not os.path.exists(args.from_file):
            print(f"[Export] Error: Source file does not exist: {args.from_file}")
            sys.exit(1)
        import shutil
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copyfile(args.from_file, out_path)
        print(f"[Export] Copied model from {args.from_file} to {out_path}")
    else:
        create_dummy_onnx(out_path, args.input_size, args.feature_dim)


if __name__ == "__main__":
    main()
