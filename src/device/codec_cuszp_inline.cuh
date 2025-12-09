// codec_cuszp_inline.cuh (hard-protected, status-coded)
// Minimal, single-header, GPU-only fixed-length packer for INT8/INT16.
// - Whole-frame unified bitw, cmpBytes pre-computable.
// - Sign-map (1bit/elem, 32->uint32) + bit-planes (LSB->MSB).
// - Hard protection: NO truncation. If even raw won't fit, return ERR_CAPACITY.
// - Host-visible size formula for smoke-check.
// - Single-thread reference implementation (<<<1,1>>>). Parallelize later.

#ifndef CODEC_CUSZP_INLINE_CUH_
#define CODEC_CUSZP_INLINE_CUH_

#include <stdint.h>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuszp_inline {

#ifndef CODEC_BLOCKL
#define CODEC_BLOCKL 256
#endif
#ifndef CODEC_MAGIC
#define CODEC_MAGIC 0xCC0DEC01u
#endif

#pragma pack(push,1)
struct CodecHdrV1 {
  uint32_t magic;     // 0xCC0DEC01
  uint8_t  codec;     // 1=cuszp-fixed
  uint8_t  dtype;     // 0=i8, 1=i16
  uint8_t  flags;     // bit0: F_FALLBACK
  uint8_t  rsvd;
  uint32_t rawBytes;
  uint32_t cmpBytes;  // payload bytes (excl. header)
  uint16_t blockL;
  uint16_t bitw;
};
#pragma pack(pop)

enum : uint8_t {
  CODEC_ID_CUSZP_FIXED = 1,
  DTYPE_I8  = 0,
  DTYPE_I16 = 1,
  F_FALLBACK = 1u << 0
};

enum CodecStatus : uint32_t {
  CODEC_OK_COMPRESSED   = 0,
  CODEC_OK_RAW_FALLBACK = 1,
  CODEC_ERR_CAPACITY    = 2
};

struct StepEncodeResult {
  uint16_t    bitw;
  uint32_t    cmpBytes;   // payload-only
  bool        fallback;
  CodecStatus status;
};

// ---- utils ----
__device__ __forceinline__ uint32_t ceil_div_u32(uint32_t a, uint32_t b){ return (a + b - 1u) / b; }
__device__ __forceinline__ uint32_t min_u32(uint32_t a, uint32_t b){return a<b?a:b;}
__device__ __forceinline__ uint32_t max_u32(uint32_t a, uint32_t b){return a>b?a:b;}
__device__ __forceinline__ uint8_t  abs_i8 (int8_t  x){ int32_t v=(int32_t)x; return (uint8_t) (v<0?-v:v); }
__device__ __forceinline__ uint16_t abs_i16(int16_t x){ int32_t v=(int32_t)x; return (uint16_t)(v<0?-v:v); }
__device__ __forceinline__ uint32_t clz_u32(uint32_t x){
#if __CUDA_ARCH__ >= 300
  return __clz(x);
#else
  uint32_t n=0; while((x & 0x80000000u)==0 && n<32){x<<=1; ++n;} return n;
#endif
}

// ---- unified bitw ----
template<typename T>
__device__ __forceinline__
uint16_t compute_bitw_unified(const T* src, uint32_t N){
  uint32_t m = 0;
  if constexpr (std::is_same<T,int8_t>::value){
    for(uint32_t i=0;i<N;++i) m = max_u32(m, (uint32_t)abs_i8(src[i]));
  } else {
    for(uint32_t i=0;i<N;++i) m = max_u32(m, (uint32_t)abs_i16(src[i]));
  }
  if (m==0) return 0;
  return (uint16_t)(32u - clz_u32(m));
}

// ---- payload size formula (host+device) ----
__host__ __device__ __forceinline__
uint32_t payload_bytes_from_formula(uint32_t N, uint16_t blockL, uint16_t bitw){
  if (N==0) return 0;
  const uint32_t K = (N + blockL - 1u) / blockL;
  const uint32_t T = N - (K-1u)*blockL;
  const uint32_t sign_full  = (blockL/32u) * 4u;
  const uint32_t plane_full = ((uint32_t)bitw) * (blockL/8u);
  const uint32_t full_bytes = sign_full + plane_full;
  const uint32_t sign_tail  = ((T + 31u)/32u) * 4u;
  const uint32_t plane_tail = ((uint32_t)bitw) * ((T + 7u)/8u);
  return (K-1u)*full_bytes + (sign_tail + plane_tail);
}

// ---- block I/O (single-thread reference) ----
template<typename T>
__device__ __forceinline__
void write_block_payload(const T* src, uint32_t valid_elems,
                         uint16_t bitw, uint8_t* dst){
  const uint32_t groups = (valid_elems + 31u) / 32u;
  // sign-map
  for(uint32_t g=0; g<groups; ++g){
    uint32_t base = g*32u, sign = 0u;
    for(uint32_t k=0;k<32u;++k){
      uint32_t idx = base+k;
      uint32_t bit = 0u;
      if (idx < valid_elems){
        int32_t v = (int32_t)src[idx];
        bit = (v<0)?1u:0u;
      }
      sign |= (bit << k);
    }
    dst[0]=(uint8_t)(sign&0xFFu); dst[1]=(uint8_t)((sign>>8)&0xFFu);
    dst[2]=(uint8_t)((sign>>16)&0xFFu); dst[3]=(uint8_t)((sign>>24)&0xFFu);
    dst+=4;
  }
  // planes
  if (bitw==0) return;
  for(uint16_t b=0;b<bitw;++b){
    for(uint32_t g=0; g<groups; ++g){
      uint32_t base=g*32u, lane=0u;
      for(uint32_t k=0;k<32u;++k){
        uint32_t idx=base+k, bit=0u;
        if (idx<valid_elems){
          uint32_t a = std::is_same<T,int8_t>::value ? (uint32_t)abs_i8(src[idx])
                                                     : (uint32_t)abs_i16(src[idx]);
          bit = (a>>b)&1u;
        }
        lane |= (bit<<k);
      }
      dst[0]=(uint8_t)(lane&0xFFu); dst[1]=(uint8_t)((lane>>8)&0xFFu);
      dst[2]=(uint8_t)((lane>>16)&0xFFu); dst[3]=(uint8_t)((lane>>24)&0xFFu);
      dst+=4;
    }
  }
}

template<typename T>
__device__ __forceinline__
void read_block_payload(const uint8_t* src, uint32_t valid_elems,
                        uint16_t bitw, T* dst){
  const uint32_t groups = (valid_elems + 31u) / 32u;
  for(uint32_t i=0;i<valid_elems;++i) dst[i]=(T)0;

  const uint8_t* p = src;
  const uint8_t* planes = p + groups*4u;

  if (bitw>0){
    for(uint16_t b=0;b<bitw;++b){
      for(uint32_t g=0; g<groups; ++g){
        const uint8_t* gp = planes + (b*groups + g)*4u;
        uint32_t lane = (uint32_t)gp[0] | ((uint32_t)gp[1]<<8) |
                        ((uint32_t)gp[2]<<16) | ((uint32_t)gp[3]<<24);
        uint32_t base=g*32u;
        for(uint32_t k=0;k<32u;++k){
          uint32_t idx=base+k;
          if (idx<valid_elems){
            uint32_t bit=(lane>>k)&1u;
            if constexpr (std::is_same<T,int8_t>::value){
              uint32_t a=(uint8_t)dst[idx]; a|=(bit<<b); dst[idx]=(T)(uint8_t)a;
            } else {
              uint32_t a=(uint16_t)dst[idx]; a|=(bit<<b); dst[idx]=(T)(uint16_t)a;
            }
          }
        }
      }
    }
  }

  const uint8_t* signp = p;
  for(uint32_t g=0; g<groups; ++g){
    uint32_t sign = (uint32_t)signp[0] | ((uint32_t)signp[1]<<8) |
                    ((uint32_t)signp[2]<<16) | ((uint32_t)signp[3]<<24);
    signp+=4;
    uint32_t base=g*32u;
    for(uint32_t k=0;k<32u;++k){
      uint32_t idx=base+k;
      if (idx<valid_elems && ((sign>>k)&1u)){
        if constexpr (std::is_same<T,int8_t>::value){
          uint8_t a=(uint8_t)dst[idx]; dst[idx]=(T)(int8_t)(-(int32_t)a);
        } else {
          uint16_t a=(uint16_t)dst[idx]; dst[idx]=(T)(int16_t)(-(int32_t)a);
        }
      }
    }
  }
}

// ---- frame encode (hard-protected) ----
template<typename T>
__device__ __forceinline__
StepEncodeResult step_encode(const T* src, uint32_t rawBytes,
                             uint8_t* dst, uint32_t dstCapBytes,
                             uint16_t blockL = CODEC_BLOCKL){
  StepEncodeResult r{0,0,false,CODEC_ERR_CAPACITY};

  const uint32_t N = rawBytes / (uint32_t)sizeof(T);
  uint16_t bitw = compute_bitw_unified<T>(src, N);
  uint32_t cmpBytes = payload_bytes_from_formula(N, blockL, bitw);

  const uint32_t need_cmp = (uint32_t)sizeof(CodecHdrV1) + cmpBytes;
  const uint32_t need_raw = (uint32_t)sizeof(CodecHdrV1) + rawBytes;

  // Case 1: compress fits and benefits
  if (cmpBytes < rawBytes && need_cmp <= dstCapBytes){
    CodecHdrV1* hdr = (CodecHdrV1*)dst;
    hdr->magic=CODEC_MAGIC; hdr->codec=CODEC_ID_CUSZP_FIXED;
    hdr->dtype=(uint8_t)(std::is_same<T,int8_t>::value?DTYPE_I8:DTYPE_I16);
    hdr->flags=0; hdr->rawBytes=rawBytes; hdr->cmpBytes=cmpBytes;
    hdr->blockL=blockL; hdr->bitw=bitw;

    uint8_t* payload = dst + sizeof(CodecHdrV1);
    uint32_t written=0, left=N;
    const T* pSrc=src;
    while(left){
      uint32_t take=min_u32(left,(uint32_t)blockL);
      write_block_payload<T>(pSrc,take,bitw,payload+written);
      uint32_t signB=((take+31u)/32u)*4u;
      uint32_t planeB=((uint32_t)bitw)*((take+7u)/8u);
      written+=signB+planeB; pSrc+=take; left-=take;
    }
    r.bitw=bitw; r.cmpBytes=cmpBytes; r.fallback=false; r.status=CODEC_OK_COMPRESSED;
    return r;
  }

  // Case 2: fallback to raw (fits)
  if (need_raw <= dstCapBytes){
    CodecHdrV1* hdr = (CodecHdrV1*)dst;
    hdr->magic=CODEC_MAGIC; hdr->codec=CODEC_ID_CUSZP_FIXED;
    hdr->dtype=(uint8_t)(std::is_same<T,int8_t>::value?DTYPE_I8:DTYPE_I16);
    hdr->flags=F_FALLBACK; hdr->rawBytes=rawBytes; hdr->cmpBytes=rawBytes;
    hdr->blockL=blockL; hdr->bitw=bitw;
    uint8_t* payload = dst + sizeof(CodecHdrV1);
    for(uint32_t i=0;i<rawBytes;++i) payload[i]=((const uint8_t*)src)[i];
    r.bitw=bitw; r.cmpBytes=rawBytes; r.fallback=true; r.status=CODEC_OK_RAW_FALLBACK;
    return r;
  }

  // Case 3: capacity error (hard protection) — DO NOT write anything
  r.bitw=bitw; r.cmpBytes=0; r.fallback=true; r.status=CODEC_ERR_CAPACITY;
  return r;
}

// ---- frame decode ----
template<typename T>
__device__ __forceinline__
void step_decode(const uint8_t* src, T* dst){
  const CodecHdrV1* hdr = (const CodecHdrV1*)src;
  const uint8_t* payload = src + sizeof(CodecHdrV1);
  const uint32_t N = hdr->rawBytes / (uint32_t)sizeof(T);
  const uint16_t L = hdr->blockL;
  const uint16_t bitw = hdr->bitw;

  if (hdr->flags & F_FALLBACK){
    for(uint32_t i=0;i<hdr->rawBytes;++i) ((uint8_t*)dst)[i]=payload[i];
    return;
  }

  const uint8_t* p=payload;
  uint32_t left=N;
  while(left){
    uint32_t take=min_u32(left,(uint32_t)L);
    uint32_t signB=((take+31u)/32u)*4u;
    uint32_t planeB=((uint32_t)bitw)*((take+7u)/8u);
    read_block_payload<T>(p,take,bitw,dst);
    p+=signB+planeB; dst+=take; left-=take;
  }
}

} // namespace cuszp_inline

// ----------------------------------------------------------------------------
// SimpleC codec wrappers with precise size semantics
// ----------------------------------------------------------------------------
namespace cuszp_inline {

__device__ __forceinline__
int encode_simplec(const uint8_t* in,
                   int            rawBytes,
                   uint8_t*       out,
                   int            outCapBytes,
                   CodecHdrV1*    hdr) {
  // Prepare header defaults; final fields may be overwritten by codec.
  hdr->magic    = CODEC_MAGIC;
  hdr->codec    = CODEC_ID_CUSZP_FIXED;
  hdr->dtype    = DTYPE_I8;
  hdr->rawBytes = (uint32_t)rawBytes;
  hdr->blockL   = CODEC_BLOCKL;
  hdr->bitw     = 8;
  hdr->flags    = 0;

  // Frame destination is the header location; payload follows the header.
  uint8_t* frameDst  = reinterpret_cast<uint8_t*>(hdr);
  uint32_t frameCap  = (uint32_t)(sizeof(CodecHdrV1) + (outCapBytes < 0 ? 0 : outCapBytes));

  // If src==dst (in-place into payload region), avoid overlap issues by forcing fallback.
  // The caller can still benefit later when a safe staging strategy is used.
  bool inPlaceOverlap = (in == out) || (outCapBytes <= 0);
  if (!inPlaceOverlap) {
    // Use the reference single-thread encode with protection.
    StepEncodeResult r = step_encode<int8_t>(reinterpret_cast<const int8_t*>(in), (uint32_t)rawBytes,
                                            frameDst, frameCap, CODEC_BLOCKL);
    if (r.status == CODEC_OK_COMPRESSED && (int)r.cmpBytes <= outCapBytes && (int)r.cmpBytes <= rawBytes) {
      // Header is already written by step_encode into hdr
      return (int)r.cmpBytes;
    }
    // If r.status == CODEC_OK_RAW_FALLBACK and fits, we could accept, but that
    // would still overwrite the source if in==out; we already prevented that above.
  }

  // Fallback: emit header indicating raw payload and copy raw bytes to payload region
  hdr->flags    = F_FALLBACK;
  hdr->cmpBytes = (uint32_t)rawBytes;
  hdr->bitw     = 8;
  // Raw copy into the payload region (out)
  for (int i = 0; i < rawBytes; ++i) out[i] = in[i];
  return rawBytes;
}

__device__ __forceinline__
void decode_simplec(const uint8_t*    in,
                    const CodecHdrV1* hdr,
                    uint8_t*          out) {
  const int rawBytes = (int)hdr->rawBytes;
  if (hdr->flags & F_FALLBACK) {
    // Payload is raw; memcpy
    for (int i = 0; i < rawBytes; ++i) out[i] = in[i];
    return;
  }
  // Decode full frame (header at hdr)
  step_decode<int8_t>(reinterpret_cast<const uint8_t*>(hdr), reinterpret_cast<int8_t*>(out));
}

} // namespace cuszp_inline

#endif // CODEC_CUSZP_INLINE_CUH_
