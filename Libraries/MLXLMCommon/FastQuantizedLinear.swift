//
//  FastQuantizedLinear.swift
//  mlx-swift-lm (ankerton fork)
//
//  Small-batch (2 ≤ M ≤ 8 rows) 8-bit affine quantized matmul that streams each
//  weight tile ONCE and applies it to every input row, instead of MLX's `qmv`
//  path which launches one weight-streaming threadgroup per row and so costs
//  ~M× a single-row step (measured 2026-09-12 on an M5 Max, Qwen3.8-27B 8-bit:
//  M=1 12 ms, M=2 21, M=4 46, M=8 92 for the 64 up-projections). Decode with a
//  few rows — continuous batching and MTP speculative verify — is exactly this
//  shape. Falls back to `QuantizedLinear` for anything it does not cover.
//
//  Disable with MLX_FAST_QMV=0.
//

import Foundation
import MLX
import MLXNN

enum FastQMV {
    static let enabled: Bool = ProcessInfo.processInfo.environment["MLX_FAST_QMV"] != "0"
    static let maxRows = 8
    /// Rows per kernel launch: the register-resident x tile stays flat-cost up
    /// to 4 rows (measured); larger inputs are split into 4-row launches.
    static let rowsPerLaunch = 4

    /// One compiled kernel per row count M (template-specialised).
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    static func kernel(rows M: Int) -> MLXFast.MLXFastKernel {
        lock.lock(); defer { lock.unlock() }
        if let k = kernels[M] { return k }
        let k = MLXFast.metalKernel(
            name: "fast_qmv_m\(M)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: source)
        kernels[M] = k
        return k
    }

    /// Template params: T (element type), M (rows), K (in features), N (out
    /// features). Layout matches MLX affine 8-bit, group size 64, `transpose:
    /// true`: `w` is [N, K/4] uint32 = N rows of K bytes; `scales`/`biases` are
    /// [N, K/64]. Threadgroup = 2 simdgroups; each simdgroup produces 4 output
    /// columns for ALL M rows; each thread owns 8 consecutive k per 256-wide
    /// block (the same tiling as MLX's `qmv_fast`), so the weight bytes and
    /// the per-group scale/bias are read exactly once per output column.
    static let source = """
        constexpr int VPT = 8;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int in_g = K / 64;
        const uint simd_lid = thread_index_in_simdgroup;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;

        const device uint8_t* ws = (const device uint8_t*)w + (size_t)out_row * K + simd_lid * VPT;
        const device T* sl = scales + (size_t)out_row * in_g + simd_lid / SCALE_STEP;
        const device T* bl = biases + (size_t)out_row * in_g + simd_lid / SCALE_STEP;
        const device T* xp = x + simd_lid * VPT;

        float acc[M][4];
        for (int m = 0; m < M; ++m) { for (int r = 0; r < 4; ++r) { acc[m][r] = 0.0f; } }

        for (int k = 0; k < K; k += BLOCK) {
          float xt[M][VPT];
          float xs[M];
          for (int m = 0; m < M; ++m) {
            xs[m] = 0.0f;
            const device T* xr = xp + (size_t)m * K;
            for (int i = 0; i < VPT; ++i) { float v = (float)xr[i]; xt[m][i] = v; xs[m] += v; }
          }
          for (int r = 0; r < 4; ++r) {
            const device uint8_t* wl = ws + (size_t)r * K;
            const float s = (float)sl[r * in_g];
            const float b = (float)bl[r * in_g];
            float wv[VPT];
            for (int i = 0; i < VPT; ++i) { wv[i] = (float)wl[i]; }
            for (int m = 0; m < M; ++m) {
              float d = 0.0f;
              for (int i = 0; i < VPT; ++i) { d += xt[m][i] * wv[i]; }
              acc[m][r] += s * d + b * xs[m];
            }
          }
          ws += BLOCK;
          sl += BLOCK / 64;
          bl += BLOCK / 64;
          xp += BLOCK;
        }

        for (int r = 0; r < 4; ++r) {
          for (int m = 0; m < M; ++m) {
            float total = simd_sum(acc[m][r]);
            if (simd_lid == 0) { y[(size_t)m * N + out_row + r] = (T)total; }
          }
        }
        """

    /// Whether this (x, layer) shape is served by the custom kernel.
    static func covers(x: MLXArray, layer: QuantizedLinear) -> Int? {
        guard enabled, layer.bits == 8, layer.groupSize == 64, layer.mode == .affine,
            layer.biases != nil,
            x.dtype == .bfloat16 || x.dtype == .float16 || x.dtype == .float32
        else { return nil }
        let K = x.dim(-1)
        let N = layer.scales.dim(0)
        guard K > 0, x.size % K == 0 else { return nil }
        let M = x.size / K
        guard M >= 2, M <= maxRows, K % 256 == 0, N % 8 == 0 else { return nil }
        guard layer.weight.dim(0) == N, layer.weight.dim(1) * 4 == K, layer.scales.dim(1) == K / 64 else { return nil }
        return M
    }
}

/// `QuantizedLinear` that routes small-batch 8-bit affine matmuls through
/// `FastQMV` (see the file comment) and everything else through MLX.
open class FastQuantizedLinear: QuantizedLinear {
    public convenience init(_ other: Linear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        let (q, s, b) = MLX.quantized(other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self.init(weight: q, bias: other.bias, scales: s, biases: b, groupSize: groupSize, bits: bits, mode: mode)
        self.freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let M = FastQMV.covers(x: x, layer: self), let biases else {
            return super.callAsFunction(x)
        }
        let K = x.dim(-1)
        let N = scales.dim(0)
        let x2 = x.reshaped([M, K])
        func launch(_ rows: MLXArray, _ m: Int) -> MLXArray {
            FastQMV.kernel(rows: m)(
                [weight, scales, biases, rows],
                template: [("T", x.dtype), ("M", m), ("K", K), ("N", N)],
                grid: (64, N / 8, 1),
                threadGroup: (64, 1, 1),
                outputShapes: [[m, N]],
                outputDTypes: [x.dtype])[0]
        }
        let y: MLXArray
        if M <= FastQMV.rowsPerLaunch {
            y = launch(x2, M)
        } else {
            var parts: [MLXArray] = []
            var r0 = 0
            while r0 < M {
                let m = min(FastQMV.rowsPerLaunch, M - r0)
                parts.append(launch(x2[r0 ..< (r0 + m)], m))
                r0 += m
            }
            y = concatenated(parts, axis: 0)
        }
        var out = y.reshaped(Array(x.shape.dropLast()) + [N])
        if let bias { out = out + bias }
        return out
    }
}

/// Drop-in `apply:` for `quantize(model:filter:apply:)`: plain `Linear` layers
/// become `FastQuantizedLinear`; everything else (embeddings, already-quantized
/// modules) takes the stock path.
public func fastQuantizeSingle(
    layer: Module, groupSize: Int, bits: Int, mode: QuantizationMode
) -> Module? {
    if layer is Quantized { return nil }
    if let linear = layer as? Linear, type(of: layer) == Linear.self {
        return FastQuantizedLinear(linear, groupSize: groupSize, bits: bits, mode: mode)
    }
    return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode) as? Module
}
