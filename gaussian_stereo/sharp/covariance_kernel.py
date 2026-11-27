"""Optional CUDA Jacobi solver. Original PyTorch SVD is always available.

Only video CUDA geometry uses this module. No CuPy import at startup. All
arrays are Torch-owned; CuPy launches the kernel on the current Torch stream.
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

LOG = logging.getLogger(__name__)
_KERNEL = None
_CP = None
_DECISION = {}


def _reference(m):
    u, s, _ = torch.linalg.svd(m)
    return u, s


def _kernel():
    global _KERNEL, _CP
    if _KERNEL is None:
        import cupy as cp
        _CP = cp
        code = Path(__file__).with_name('covariance_jacobi.cu').read_text(encoding='utf-8')
        _KERNEL = cp.RawKernel(code, 'solve_covariances', options=('--std=c++11',))
    return _CP, _KERNEL


def _candidate(m):
    cp, kernel = _kernel()
    flat = m.contiguous().reshape(-1, 3, 3)
    n = flat.shape[0]
    if not 0 < n < 2**31:
        raise ValueError('Unsupported number of covariances')
    u = torch.empty_like(flat)
    s = torch.empty((n, 3), device=m.device, dtype=m.dtype)
    bad = torch.empty((n,), device=m.device, dtype=torch.uint8)
    stream = torch.cuda.current_stream(m.device)
    with cp.cuda.Device(m.device.index), cp.cuda.ExternalStream(stream.cuda_stream, device_id=m.device.index):
        # DLPack views of Torch allocations; launch on the SAME stream, so the
        # producer/consumer ordering and allocator lifetime remain unchanged.
        args = tuple(cp.from_dlpack(x) for x in (flat, u, s, bad))
        kernel(((n + 127)//128,), (128,), (*args, np.int32(n)))
    ids = torch.nonzero(bad, as_tuple=False).flatten()
    rejected = ids.numel()
    # For difficult scenes avoid spending time on a large scattered fallback.
    if rejected > n * 0.10:
        ref_u, ref_s = _reference(m)
        return ref_u, ref_s, rejected, False
    if rejected:
        ref_u, ref_s = _reference(flat.index_select(0, ids))
        u.index_copy_(0, ids, ref_u)
        s.index_copy_(0, ids, ref_s)
    return u.reshape(m.shape), s.reshape(m.shape[:-1]), rejected, True


def _validate(m, u, s, ref_u, ref_s):
    # Compare the actual full frame, chunked to avoid a large temporary VRAM
    # allocation. The SVD basis can differ by signs or degenerate subspaces.
    v, w = u.reshape(-1,3,3), s.reshape(-1,3)
    rv, rw = ref_u.reshape(-1,3,3), ref_s.reshape(-1,3)
    max_cov = torch.zeros((), device=m.device)
    max_scale = torch.zeros((), device=m.device)
    for i in range(0, len(v), 65536):
        a, b = v[i:i+65536], w[i:i+65536]
        c, d = rv[i:i+65536], rw[i:i+65536]
        # Compare covariance implied by the two solvers using normalized
        # eigenvalues to avoid overflow/underflow at different scene scales.
        norm = d[:, :1].clamp_min(1e-30)
        # Explicit elementwise contraction avoids TF32/FP16 matmul settings
        # affecting the accuracy gate.
        cov_a = ((a * (b/norm)[:,None,:])[:,:,None,:] * a[:,None,:,:]).sum(-1)
        cov_b = ((c * (d/norm)[:,None,:])[:,:,None,:] * c[:,None,:,:]).sum(-1)
        max_cov = torch.maximum(max_cov, (cov_a-cov_b).abs().amax())
        rel = (b.sqrt()-d.sqrt()).abs()/d.sqrt().clamp_min(1e-30)
        max_scale = torch.maximum(max_scale, rel.amax())
    err_cov, err_scale = float(max_cov), float(max_scale)
    return err_cov <= 2e-5 and err_scale <= 1e-3, err_cov, err_scale


def _timed(fn, m):
    stream = torch.cuda.current_stream(m.device)
    stream.synchronize()
    start = time.perf_counter()
    result = fn(m)
    stream.synchronize()
    return result, time.perf_counter()-start


def solve(m):
    """Return U and descending nonnegative singular values, same shape as SVD."""
    if (not m.is_cuda or m.dtype != torch.float32 or m.shape[-2:] != (3,3)
            or os.environ.get('SHARP_VR_COV_KERNEL', '1') == '0'):
        return _reference(m)
    key = (m.device.index, m.shape, m.dtype)
    mode = _DECISION.get(key)
    if mode == 'svd':
        return _reference(m)
    if mode == 'jacobi':
        try:
            u, s, rejected, used = _candidate(m)
            if not used:
                LOG.warning('[SHARP VR Video] CUDA covariance kernel: high fallback count %d; frame used SVD', rejected)
            elif rejected:
                LOG.debug('[SHARP VR Video] CUDA covariance kernel: SVD fallback %d/%d', rejected, m.numel()//9)
            return u, s
        except Exception as exc:
            _DECISION[key] = 'svd'
            LOG.warning('[SHARP VR Video] CUDA covariance kernel disabled: %s', exc)
            return _reference(m)
    try:
        # Compile and warm the kernel on a small real-data subset before the
        # timed full-frame comparison; startup compile is not a speed sample.
        cp, kernel = _kernel()
        with cp.cuda.Device(m.device.index):
            kernel.compile()
        _candidate(m.reshape(-1,3,3)[:min(4096,m.numel()//9)])
        (ru, rs), t_ref = _timed(_reference, m)
        (u, s, rejected, used), t_new = _timed(_candidate, m)
        passed, ec, es = _validate(m, u, s, ru, rs) if used else (False, float('nan'), float('nan'))
        choose = passed and t_new < t_ref * 0.90
        _DECISION[key] = 'jacobi' if choose else 'svd'
        LOG.info('[SHARP VR Video] CUDA covariance kernel: %s (SVD=%.3fs, checked Jacobi=%.3fs, fallback=%d/%d, max covariance error=%.3g, max relative scale error=%.3g)',
                 'Jacobi selected' if choose else 'SVD retained', t_ref, t_new, rejected, m.numel()//9, ec, es)
        # Use the selected solver for the calibration frame too.
        return (u, s) if choose else (ru, rs)
    except Exception as exc:
        _DECISION[key] = 'svd'
        LOG.warning('[SHARP VR Video] CUDA covariance kernel unavailable; using SVD (%s)', exc)
        return _reference(m)
