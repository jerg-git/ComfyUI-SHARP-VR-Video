// SHARP VR Video experimental symmetric 3x3 covariance solver.
// One independent matrix per CUDA thread. No fast-math or reduced precision.
// This code also builds as a host function for numerical regression tests.
#ifdef SHARP_HOST_TEST
#include <math.h>
// Host regression tests need the system math header. NVRTC device math
// functions are built in; do not include host headers in the CUDA path.
#define DEV inline
#else
#define DEV __device__ __forceinline__
#endif

// Apply a Jacobi rotation to rows/columns p,q, with remaining axis k.
// Keep only the six independent entries of the symmetric matrix.
DEV void rotate_pair(float &app, float &aqq, float &apq,
                     float &apk, float &aqk,
                     float &vp0, float &vp1, float &vp2,
                     float &vq0, float &vq1, float &vq2) {
    if (fabsf(apq) <= 1.0e-9f) return;
    const float delta = aqq - app;
    const float h = sqrtf(delta*delta + 4.0f*apq*apq);
    const float t = (2.0f*apq) / (delta + copysignf(h, delta));
    const float c = 1.0f / sqrtf(1.0f + t*t);
    const float s = t*c;
    app -= t*apq;
    aqq += t*apq;
    apq = 0.0f;
    const float x = apk, y = aqk;
    apk = c*x - s*y;
    aqk = s*x + c*y;
    const float x0=vp0, x1=vp1, x2=vp2;
    vp0=c*x0-s*vq0; vp1=c*x1-s*vq1; vp2=c*x2-s*vq2;
    vq0=s*x0+c*vq0; vq1=s*x1+c*vq1; vq2=s*x2+c*vq2;
}

DEV void swap_eigen(float &a, float &b, float &x0, float &x1, float &x2,
                    float &y0, float &y1, float &y2) {
    if (a < b) {
        float t=a; a=b; b=t;
        t=x0; x0=y0; y0=t; t=x1; x1=y1; y1=t; t=x2; x2=y2; y2=t;
    }
}

DEV bool check_eigen(float a, float b, float c, float d, float e, float f,
                     float l, float x, float y, float z) {
    const float rx=a*x+d*y+e*z-l*x;
    const float ry=d*x+b*y+f*z-l*y;
    const float rz=e*x+f*y+c*z-l*z;
    // Relative residual per eigenvalue, not just largest-axis covariance.
    const float bound=5.0e-4f*l;
    return fabsf(rx)<=bound && fabsf(ry)<=bound && fabsf(rz)<=bound;
}

DEV bool solve_one(const float *m, float *v, float *w) {
    float scale=0.0f;
    for (int j=0;j<9;++j) {
        if (!isfinite(m[j])) return false;
        scale=fmaxf(scale,fabsf(m[j]));
    }
    if (!(scale>1.0e-30f && scale<1.0e30f)) return false;
    const float inv=1.0f/scale;
    // The source is theoretically symmetric positive semidefinite; preserve
    // the original SVD for material asymmetry or poorly conditioned matrices.
    const float asym=fmaxf(fabsf((m[1]-m[3])*inv),
                     fmaxf(fabsf((m[2]-m[6])*inv),fabsf((m[5]-m[7])*inv)));
    if (!(asym<=2.0e-6f)) return false;
    const float a=m[0]*inv,b=m[4]*inv,c=m[8]*inv;
    const float d=(m[1]*inv+m[3]*inv)*0.5f;
    const float e=(m[2]*inv+m[6]*inv)*0.5f;
    const float f=(m[5]*inv+m[7]*inv)*0.5f;
    float p=a,q=b,r=c,xy=d,xz=e,yz=f;
    float x0=1,x1=0,x2=0,y0=0,y1=1,y2=0,z0=0,z1=0,z2=1;
    // Five cyclic sweeps suffice for well-conditioned 3x3 matrices. Residual
    // and orthogonality checks below, not iteration count, determine acceptance.
    for (int sweep=0;sweep<5;++sweep) {
        rotate_pair(p,q,xy,xz,yz,x0,x1,x2,y0,y1,y2);
        rotate_pair(p,r,xz,xy,yz,x0,x1,x2,z0,z1,z2);
        rotate_pair(q,r,yz,xy,xz,y0,y1,y2,z0,z1,z2);
    }
    swap_eigen(p,q,x0,x1,x2,y0,y1,y2);
    swap_eigen(q,r,y0,y1,y2,z0,z1,z2);
    swap_eigen(p,q,x0,x1,x2,y0,y1,y2);
    if (!(p>0 && r>=p*1.0e-4f)) return false;
    if (!(asym<=1.0e-4f*r)) return false;
    if (!check_eigen(a,b,c,d,e,f,p,x0,x1,x2) ||
        !check_eigen(a,b,c,d,e,f,q,y0,y1,y2) ||
        !check_eigen(a,b,c,d,e,f,r,z0,z1,z2)) return false;
    const float tol=3.0e-6f;
    if (!(fabsf(x0*x0+x1*x1+x2*x2-1)<=tol &&
          fabsf(y0*y0+y1*y1+y2*y2-1)<=tol &&
          fabsf(z0*z0+z1*z1+z2*z2-1)<=tol &&
          fabsf(x0*y0+x1*y1+x2*y2)<=tol &&
          fabsf(x0*z0+x1*z1+x2*z2)<=tol &&
          fabsf(y0*z0+y1*z1+y2*z2)<=tol)) return false;
    w[0]=p*scale; w[1]=q*scale; w[2]=r*scale;
    v[0]=x0; v[1]=y0; v[2]=z0;
    v[3]=x1; v[4]=y1; v[5]=z1;
    v[6]=x2; v[7]=y2; v[8]=z2;
    return true;
}

#ifdef SHARP_HOST_TEST
extern "C" void solve_covariances(const float *m, float *v, float *w,
                                 unsigned char *bad, int n) {
    for (int i=0;i<n;++i) bad[i]=!solve_one(m+9*i,v+9*i,w+3*i);
}
#else
extern "C" __global__ void solve_covariances(const float *m, float *v, float *w,
                                            unsigned char *bad, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) bad[i]=!solve_one(m+9*i,v+9*i,w+3*i);
}
#endif
