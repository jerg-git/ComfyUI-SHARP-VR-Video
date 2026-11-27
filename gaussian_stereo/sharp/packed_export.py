"""Optional GPU export of the existing gsplat 32-byte row format.

Retains the legacy FP32 PLY attribute rounding and JS quantization. No import
of CuPy until a compatible browser requests the packed transport.
"""
import logging
import numpy as np
import torch

LOG = logging.getLogger(__name__)
MAGIC = b'SVRPACK1'
_KERNEL = None
_STATUS = {}
_SOURCE = r'''
extern "C" __global__ void pack(const float* a, unsigned int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a += i*14; out += i*8;
  for (int j=0;j<3;++j) out[j]=__float_as_uint(a[j]);
  for (int j=0;j<3;++j) out[j+3]=__float_as_uint((float)exp((double)a[j+7]));
  double c[4] = {(0.5+0.28209479177387814*(double)a[3])*255.0,
                 (0.5+0.28209479177387814*(double)a[4])*255.0,
                 (0.5+0.28209479177387814*(double)a[5])*255.0,
                 (1.0/(1.0+exp(-(double)a[6])))*255.0};
  double w=a[10], x=a[11], y=a[12], z=a[13];
  double norm=sqrt(x*x+y*y+z*z+w*w);
  double q[4]={w/norm*128.0+128.0,x/norm*128.0+128.0,y/norm*128.0+128.0,z/norm*128.0+128.0};
  unsigned int rgba=0, rot=0;
  for(int j=0;j<4;++j) {
    double v=c[j], r=q[j];
    unsigned int cv=!(v>0.0)?0:(v>=255.0?255:__double2uint_rn(v));
    unsigned int rv=!(r>0.0)?0:(r>=255.0?255:__double2uint_rn(r));
    rgba |= cv<<(j*8); rot |= rv<<(j*8);
  }
  out[6]=rgba; out[7]=rot;
}
'''


def reference(a):
    a = np.asarray(a, dtype=np.float32).astype(np.float64)
    out = np.empty((len(a), 8), dtype=np.float32)
    out[:, :3] = a[:, :3]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        out[:, 3:6] = np.exp(a[:, 7:10])
        colors = np.column_stack(((.5+.28209479177387814*a[:,3:6])*255, (1/(1+np.exp(-a[:,6])))*255))
        w,x,y,z = a[:,10:14].T
        q = a[:,10:14]/np.sqrt(x*x+y*y+z*z+w*w)[:,None]*128+128
    b = out.view(np.uint8).reshape(-1,32)
    b[:,24:28] = np.rint(np.nan_to_num(colors,nan=0).clip(0,255)).astype(np.uint8)
    b[:,28:32] = np.rint(np.nan_to_num(q,nan=0).clip(0,255)).astype(np.uint8)
    return out.tobytes()


def try_pack(attributes):
    global _KERNEL
    if not attributes.is_cuda:
        return None
    key = attributes.device.index
    if _STATUS.get(key) is False:
        return None
    try:
        import cupy as cp
        if _KERNEL is None:
            _KERNEL = cp.RawKernel(_SOURCE, 'pack', options=('--std=c++11','--fmad=false'))
        a = attributes.detach().to(torch.float32).contiguous()
        n = a.shape[0]
        out = torch.empty((n,32), dtype=torch.uint8, device=a.device)
        stream = torch.cuda.current_stream(a.device)
        with cp.cuda.Device(key), cp.cuda.ExternalStream(stream.cuda_stream, device_id=key):
            _KERNEL(((n+127)//128,), (128,), (cp.from_dlpack(a), cp.from_dlpack(out), np.int32(n)))
        raw = out.cpu().numpy().tobytes()
        if key not in _STATUS:
            # Validate the actual complete first frame against legacy conversion.
            # Allow at most one FP32 ULP in exp() across CUDA/host libraries;
            # colors, rotations and positions must match byte-for-byte.
            ref = reference(a.cpu().numpy())
            actual = np.frombuffer(raw,np.uint8).reshape(-1,32)
            expected = np.frombuffer(ref,np.uint8).reshape(-1,32)
            if not (np.array_equal(actual[:,:12],expected[:,:12]) and np.array_equal(actual[:,24:],expected[:,24:])):
                raise ValueError('packed colors, positions or rotations differ from reference')
            aa=np.frombuffer(raw,np.uint32).reshape(-1,8)[:,3:6].astype(np.int64)
            bb=np.frombuffer(ref,np.uint32).reshape(-1,8)[:,3:6].astype(np.int64)
            if np.max(np.abs(aa-bb))>1:
                raise ValueError('packed scales differ from reference')
            _STATUS[key] = True
            LOG.info('[SHARP VR Video] Packed GPU export validated (%d splats)', n)
        return MAGIC + raw
    except Exception as exc:
        _STATUS[key] = False
        LOG.warning('[SHARP VR Video] Packed GPU export unavailable; using PLY (%s)', exc)
        return None
