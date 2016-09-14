# 1 "cuda_main.cu"
# 56 "/usr/local/cuda/include/cuda_runtime.h"
#pragma GCC diagnostic push


#pragma GCC diagnostic ignored "-Wunused-function"
# 35 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility push ( default )
# 149 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility pop
# 42 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility push ( default )
# 120 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility pop
# 1888 "/usr/local/cuda/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 30 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/gthr.h" 3
#pragma GCC visibility push ( default )
# 151 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/gthr.h" 3
#pragma GCC visibility pop
# 36 "/usr/include/c++/4.8/bits/cxxabi_forced.h" 3
#pragma GCC visibility push ( default )
# 58 "/usr/include/c++/4.8/bits/cxxabi_forced.h" 3
#pragma GCC visibility pop
# 1425 "/usr/local/cuda/include/driver_types.h"
struct CUstream_st;




struct CUevent_st;
# 27 "/usr/include/xlocale.h" 3
struct __locale_struct;
# 180 "/usr/include/libio.h" 3
enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv};
# 245 "/usr/include/libio.h" 3
struct _IO_FILE;
# 51 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
enum idtype_t {
P_ALL,
P_PID,
P_PGID};
# 190 "/usr/include/math.h" 3
enum _ZUt_ {
FP_NAN,


FP_INFINITE,


FP_ZERO,


FP_SUBNORMAL,


FP_NORMAL};
# 302 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_};
# 241 "/usr/include/x86_64-linux-gnu/bits/fcntl-linux.h" 3
enum __pid_type {

F_OWNER_TID,
F_OWNER_PID,
F_OWNER_PGRP,
F_OWNER_GID = 2};
# 25 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt0_ {
_PC_LINK_MAX,

_PC_MAX_CANON,

_PC_MAX_INPUT,

_PC_NAME_MAX,

_PC_PATH_MAX,

_PC_PIPE_BUF,

_PC_CHOWN_RESTRICTED,

_PC_NO_TRUNC,

_PC_VDISABLE,

_PC_SYNC_IO,

_PC_ASYNC_IO,

_PC_PRIO_IO,

_PC_SOCK_MAXBUF,

_PC_FILESIZEBITS,

_PC_REC_INCR_XFER_SIZE,

_PC_REC_MAX_XFER_SIZE,

_PC_REC_MIN_XFER_SIZE,

_PC_REC_XFER_ALIGN,

_PC_ALLOC_SIZE_MIN,

_PC_SYMLINK_MAX,

_PC_2_SYMLINKS};
# 72 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt1_ {
_SC_ARG_MAX,

_SC_CHILD_MAX,

_SC_CLK_TCK,

_SC_NGROUPS_MAX,

_SC_OPEN_MAX,

_SC_STREAM_MAX,

_SC_TZNAME_MAX,

_SC_JOB_CONTROL,

_SC_SAVED_IDS,

_SC_REALTIME_SIGNALS,

_SC_PRIORITY_SCHEDULING,

_SC_TIMERS,

_SC_ASYNCHRONOUS_IO,

_SC_PRIORITIZED_IO,

_SC_SYNCHRONIZED_IO,

_SC_FSYNC,

_SC_MAPPED_FILES,

_SC_MEMLOCK,

_SC_MEMLOCK_RANGE,

_SC_MEMORY_PROTECTION,

_SC_MESSAGE_PASSING,

_SC_SEMAPHORES,

_SC_SHARED_MEMORY_OBJECTS,

_SC_AIO_LISTIO_MAX,

_SC_AIO_MAX,

_SC_AIO_PRIO_DELTA_MAX,

_SC_DELAYTIMER_MAX,

_SC_MQ_OPEN_MAX,

_SC_MQ_PRIO_MAX,

_SC_VERSION,

_SC_PAGESIZE,


_SC_RTSIG_MAX,

_SC_SEM_NSEMS_MAX,

_SC_SEM_VALUE_MAX,

_SC_SIGQUEUE_MAX,

_SC_TIMER_MAX,




_SC_BC_BASE_MAX,

_SC_BC_DIM_MAX,

_SC_BC_SCALE_MAX,

_SC_BC_STRING_MAX,

_SC_COLL_WEIGHTS_MAX,

_SC_EQUIV_CLASS_MAX,

_SC_EXPR_NEST_MAX,

_SC_LINE_MAX,

_SC_RE_DUP_MAX,

_SC_CHARCLASS_NAME_MAX,


_SC_2_VERSION,

_SC_2_C_BIND,

_SC_2_C_DEV,

_SC_2_FORT_DEV,

_SC_2_FORT_RUN,

_SC_2_SW_DEV,

_SC_2_LOCALEDEF,


_SC_PII,

_SC_PII_XTI,

_SC_PII_SOCKET,

_SC_PII_INTERNET,

_SC_PII_OSI,

_SC_POLL,

_SC_SELECT,

_SC_UIO_MAXIOV,

_SC_IOV_MAX = 60,

_SC_PII_INTERNET_STREAM,

_SC_PII_INTERNET_DGRAM,

_SC_PII_OSI_COTS,

_SC_PII_OSI_CLTS,

_SC_PII_OSI_M,

_SC_T_IOV_MAX,



_SC_THREADS,

_SC_THREAD_SAFE_FUNCTIONS,

_SC_GETGR_R_SIZE_MAX,

_SC_GETPW_R_SIZE_MAX,

_SC_LOGIN_NAME_MAX,

_SC_TTY_NAME_MAX,

_SC_THREAD_DESTRUCTOR_ITERATIONS,

_SC_THREAD_KEYS_MAX,

_SC_THREAD_STACK_MIN,

_SC_THREAD_THREADS_MAX,

_SC_THREAD_ATTR_STACKADDR,

_SC_THREAD_ATTR_STACKSIZE,

_SC_THREAD_PRIORITY_SCHEDULING,

_SC_THREAD_PRIO_INHERIT,

_SC_THREAD_PRIO_PROTECT,

_SC_THREAD_PROCESS_SHARED,


_SC_NPROCESSORS_CONF,

_SC_NPROCESSORS_ONLN,

_SC_PHYS_PAGES,

_SC_AVPHYS_PAGES,

_SC_ATEXIT_MAX,

_SC_PASS_MAX,


_SC_XOPEN_VERSION,

_SC_XOPEN_XCU_VERSION,

_SC_XOPEN_UNIX,

_SC_XOPEN_CRYPT,

_SC_XOPEN_ENH_I18N,

_SC_XOPEN_SHM,


_SC_2_CHAR_TERM,

_SC_2_C_VERSION,

_SC_2_UPE,


_SC_XOPEN_XPG2,

_SC_XOPEN_XPG3,

_SC_XOPEN_XPG4,


_SC_CHAR_BIT,

_SC_CHAR_MAX,

_SC_CHAR_MIN,

_SC_INT_MAX,

_SC_INT_MIN,

_SC_LONG_BIT,

_SC_WORD_BIT,

_SC_MB_LEN_MAX,

_SC_NZERO,

_SC_SSIZE_MAX,

_SC_SCHAR_MAX,

_SC_SCHAR_MIN,

_SC_SHRT_MAX,

_SC_SHRT_MIN,

_SC_UCHAR_MAX,

_SC_UINT_MAX,

_SC_ULONG_MAX,

_SC_USHRT_MAX,


_SC_NL_ARGMAX,

_SC_NL_LANGMAX,

_SC_NL_MSGMAX,

_SC_NL_NMAX,

_SC_NL_SETMAX,

_SC_NL_TEXTMAX,


_SC_XBS5_ILP32_OFF32,

_SC_XBS5_ILP32_OFFBIG,

_SC_XBS5_LP64_OFF64,

_SC_XBS5_LPBIG_OFFBIG,


_SC_XOPEN_LEGACY,

_SC_XOPEN_REALTIME,

_SC_XOPEN_REALTIME_THREADS,


_SC_ADVISORY_INFO,

_SC_BARRIERS,

_SC_BASE,

_SC_C_LANG_SUPPORT,

_SC_C_LANG_SUPPORT_R,

_SC_CLOCK_SELECTION,

_SC_CPUTIME,

_SC_THREAD_CPUTIME,

_SC_DEVICE_IO,

_SC_DEVICE_SPECIFIC,

_SC_DEVICE_SPECIFIC_R,

_SC_FD_MGMT,

_SC_FIFO,

_SC_PIPE,

_SC_FILE_ATTRIBUTES,

_SC_FILE_LOCKING,

_SC_FILE_SYSTEM,

_SC_MONOTONIC_CLOCK,

_SC_MULTI_PROCESS,

_SC_SINGLE_PROCESS,

_SC_NETWORKING,

_SC_READER_WRITER_LOCKS,

_SC_SPIN_LOCKS,

_SC_REGEXP,

_SC_REGEX_VERSION,

_SC_SHELL,

_SC_SIGNALS,

_SC_SPAWN,

_SC_SPORADIC_SERVER,

_SC_THREAD_SPORADIC_SERVER,

_SC_SYSTEM_DATABASE,

_SC_SYSTEM_DATABASE_R,

_SC_TIMEOUTS,

_SC_TYPED_MEMORY_OBJECTS,

_SC_USER_GROUPS,

_SC_USER_GROUPS_R,

_SC_2_PBS,

_SC_2_PBS_ACCOUNTING,

_SC_2_PBS_LOCATE,

_SC_2_PBS_MESSAGE,

_SC_2_PBS_TRACK,

_SC_SYMLOOP_MAX,

_SC_STREAMS,

_SC_2_PBS_CHECKPOINT,


_SC_V6_ILP32_OFF32,

_SC_V6_ILP32_OFFBIG,

_SC_V6_LP64_OFF64,

_SC_V6_LPBIG_OFFBIG,


_SC_HOST_NAME_MAX,

_SC_TRACE,

_SC_TRACE_EVENT_FILTER,

_SC_TRACE_INHERIT,

_SC_TRACE_LOG,


_SC_LEVEL1_ICACHE_SIZE,

_SC_LEVEL1_ICACHE_ASSOC,

_SC_LEVEL1_ICACHE_LINESIZE,

_SC_LEVEL1_DCACHE_SIZE,

_SC_LEVEL1_DCACHE_ASSOC,

_SC_LEVEL1_DCACHE_LINESIZE,

_SC_LEVEL2_CACHE_SIZE,

_SC_LEVEL2_CACHE_ASSOC,

_SC_LEVEL2_CACHE_LINESIZE,

_SC_LEVEL3_CACHE_SIZE,

_SC_LEVEL3_CACHE_ASSOC,

_SC_LEVEL3_CACHE_LINESIZE,

_SC_LEVEL4_CACHE_SIZE,

_SC_LEVEL4_CACHE_ASSOC,

_SC_LEVEL4_CACHE_LINESIZE,



_SC_IPV6 = 235,

_SC_RAW_SOCKETS,


_SC_V7_ILP32_OFF32,

_SC_V7_ILP32_OFFBIG,

_SC_V7_LP64_OFF64,

_SC_V7_LPBIG_OFFBIG,


_SC_SS_REPL_MAX,


_SC_TRACE_EVENT_NAME_MAX,

_SC_TRACE_NAME_MAX,

_SC_TRACE_SYS_MAX,

_SC_TRACE_USER_EVENT_MAX,


_SC_XOPEN_STREAMS,


_SC_THREAD_ROBUST_PRIO_INHERIT,

_SC_THREAD_ROBUST_PRIO_PROTECT};
# 534 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt2_ {
_CS_PATH,


_CS_V6_WIDTH_RESTRICTED_ENVS,



_CS_GNU_LIBC_VERSION,

_CS_GNU_LIBPTHREAD_VERSION,


_CS_V5_WIDTH_RESTRICTED_ENVS,



_CS_V7_WIDTH_RESTRICTED_ENVS,



_CS_LFS_CFLAGS = 1000,

_CS_LFS_LDFLAGS,

_CS_LFS_LIBS,

_CS_LFS_LINTFLAGS,

_CS_LFS64_CFLAGS,

_CS_LFS64_LDFLAGS,

_CS_LFS64_LIBS,

_CS_LFS64_LINTFLAGS,


_CS_XBS5_ILP32_OFF32_CFLAGS = 1100,

_CS_XBS5_ILP32_OFF32_LDFLAGS,

_CS_XBS5_ILP32_OFF32_LIBS,

_CS_XBS5_ILP32_OFF32_LINTFLAGS,

_CS_XBS5_ILP32_OFFBIG_CFLAGS,

_CS_XBS5_ILP32_OFFBIG_LDFLAGS,

_CS_XBS5_ILP32_OFFBIG_LIBS,

_CS_XBS5_ILP32_OFFBIG_LINTFLAGS,

_CS_XBS5_LP64_OFF64_CFLAGS,

_CS_XBS5_LP64_OFF64_LDFLAGS,

_CS_XBS5_LP64_OFF64_LIBS,

_CS_XBS5_LP64_OFF64_LINTFLAGS,

_CS_XBS5_LPBIG_OFFBIG_CFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LDFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LIBS,

_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,


_CS_POSIX_V6_ILP32_OFF32_CFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LDFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LIBS,

_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LIBS,

_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,

_CS_POSIX_V6_LP64_OFF64_CFLAGS,

_CS_POSIX_V6_LP64_OFF64_LDFLAGS,

_CS_POSIX_V6_LP64_OFF64_LIBS,

_CS_POSIX_V6_LP64_OFF64_LINTFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LIBS,

_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS,


_CS_POSIX_V7_ILP32_OFF32_CFLAGS,

_CS_POSIX_V7_ILP32_OFF32_LDFLAGS,

_CS_POSIX_V7_ILP32_OFF32_LIBS,

_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_LIBS,

_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS,

_CS_POSIX_V7_LP64_OFF64_CFLAGS,

_CS_POSIX_V7_LP64_OFF64_LDFLAGS,

_CS_POSIX_V7_LP64_OFF64_LIBS,

_CS_POSIX_V7_LP64_OFF64_LINTFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_LIBS,

_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS,


_CS_V6_ENV,

_CS_V7_ENV};
# 47 "/usr/include/ctype.h" 3
enum _ZUt3_ {
_ISupper = 256,
_ISlower = 512,
_ISalpha = 1024,
_ISdigit = 2048,
_ISxdigit = 4096,
_ISspace = 8192,
_ISprint = 16384,
_ISgraph = 32768,
_ISblank = 1,
_IScntrl,
_ISpunct = 4,
_ISalnum = 8};
# 33 "/usr/include/pthread.h" 3
enum _ZUt4_ {
PTHREAD_CREATE_JOINABLE,

PTHREAD_CREATE_DETACHED};
# 43 "/usr/include/pthread.h" 3
enum _ZUt5_ {
PTHREAD_MUTEX_TIMED_NP,
PTHREAD_MUTEX_RECURSIVE_NP,
PTHREAD_MUTEX_ERRORCHECK_NP,
PTHREAD_MUTEX_ADAPTIVE_NP,


PTHREAD_MUTEX_NORMAL = 0,
PTHREAD_MUTEX_RECURSIVE,
PTHREAD_MUTEX_ERRORCHECK,
PTHREAD_MUTEX_DEFAULT = 0,



PTHREAD_MUTEX_FAST_NP = 0};
# 65 "/usr/include/pthread.h" 3
enum _ZUt6_ {
PTHREAD_MUTEX_STALLED,
PTHREAD_MUTEX_STALLED_NP = 0,
PTHREAD_MUTEX_ROBUST,
PTHREAD_MUTEX_ROBUST_NP = 1};
# 77 "/usr/include/pthread.h" 3
enum _ZUt7_ {
PTHREAD_PRIO_NONE,
PTHREAD_PRIO_INHERIT,
PTHREAD_PRIO_PROTECT};
# 126 "/usr/include/pthread.h" 3
enum _ZUt8_ {
PTHREAD_RWLOCK_PREFER_READER_NP,
PTHREAD_RWLOCK_PREFER_WRITER_NP,
PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
PTHREAD_RWLOCK_DEFAULT_NP = 0};
# 167 "/usr/include/pthread.h" 3
enum _ZUt9_ {
PTHREAD_INHERIT_SCHED,

PTHREAD_EXPLICIT_SCHED};
# 177 "/usr/include/pthread.h" 3
enum _ZUt10_ {
PTHREAD_SCOPE_SYSTEM,

PTHREAD_SCOPE_PROCESS};
# 187 "/usr/include/pthread.h" 3
enum _ZUt11_ {
PTHREAD_PROCESS_PRIVATE,

PTHREAD_PROCESS_SHARED};
# 211 "/usr/include/pthread.h" 3
enum _ZUt12_ {
PTHREAD_CANCEL_ENABLE,

PTHREAD_CANCEL_DISABLE};



enum _ZUt13_ {
PTHREAD_CANCEL_DEFERRED,

PTHREAD_CANCEL_ASYNCHRONOUS};
# 72 "/usr/include/wctype.h" 3
enum _ZUt14_ {
__ISwupper,
__ISwlower,
__ISwalpha,
__ISwdigit,
__ISwxdigit,
__ISwspace,
__ISwprint,
__ISwgraph,
__ISwblank,
__ISwcntrl,
__ISwpunct,
__ISwalnum,

_ISwupper = 16777216,
_ISwlower = 33554432,
_ISwalpha = 67108864,
_ISwdigit = 134217728,
_ISwxdigit = 268435456,
_ISwspace = 536870912,
_ISwprint = 1073741824,
_ISwgraph = (-2147483647-1),
_ISwblank = 65536,
_ISwcntrl = 131072,
_ISwpunct = 262144,
_ISwalnum = 524288};
# 91 "/usr/include/x86_64-linux-gnu/sys/time.h" 3
enum __itimer_which {


ITIMER_REAL,


ITIMER_VIRTUAL,



ITIMER_PROF};
# 128 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E { _ZNSt9__is_voidIvE7__valueE = 1};
# 148 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E { _ZNSt12__is_integerIbE7__valueE = 1};
# 155 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E { _ZNSt12__is_integerIcE7__valueE = 1};
# 162 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E { _ZNSt12__is_integerIaE7__valueE = 1};
# 169 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E { _ZNSt12__is_integerIhE7__valueE = 1};
# 177 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E { _ZNSt12__is_integerIwE7__valueE = 1};
# 201 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E { _ZNSt12__is_integerIsE7__valueE = 1};
# 208 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E { _ZNSt12__is_integerItE7__valueE = 1};
# 215 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E { _ZNSt12__is_integerIiE7__valueE = 1};
# 222 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E { _ZNSt12__is_integerIjE7__valueE = 1};
# 229 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E { _ZNSt12__is_integerIlE7__valueE = 1};
# 236 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E { _ZNSt12__is_integerImE7__valueE = 1};
# 243 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E { _ZNSt12__is_integerIxE7__valueE = 1};
# 250 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E { _ZNSt12__is_integerIyE7__valueE = 1};
# 268 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E { _ZNSt13__is_floatingIfE7__valueE = 1};
# 275 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E { _ZNSt13__is_floatingIdE7__valueE = 1};
# 282 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E { _ZNSt13__is_floatingIeE7__valueE = 1};
# 358 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E { _ZNSt9__is_charIcE7__valueE = 1};
# 366 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E { _ZNSt9__is_charIwE7__valueE = 1};
# 381 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E { _ZNSt9__is_byteIcE7__valueE = 1};
# 388 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E { _ZNSt9__is_byteIaE7__valueE = 1};
# 395 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E { _ZNSt9__is_byteIhE7__valueE = 1};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIeEUt_E { _ZNSt12__is_integerIeE7__valueE}; enum _ZNSt12__is_integerIdEUt_E { _ZNSt12__is_integerIdE7__valueE}; enum _ZNSt12__is_integerIfEUt_E { _ZNSt12__is_integerIfE7__valueE};
# 233 "/usr/include/c++/4.8/bits/char_traits.h" 3
struct _ZSt11char_traitsIcE;
# 338 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5facetE; struct __SO__NSt6locale5facetE;
# 475 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5_ImplE;
# 304 "/usr/include/c++/4.8/bits/locale_classes.h" 3
enum _ZNSt6localeUt_E { _ZNSt6locale18_S_categories_sizeE = 12};
# 62 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZSt6locale;
# 51 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt13_Ios_Fmtflags {

_ZSt12_S_boolalpha = 1,
_ZSt6_S_dec,
_ZSt8_S_fixed = 4,
_ZSt6_S_hex = 8,
_ZSt11_S_internal = 16,
_ZSt7_S_left = 32,
_ZSt6_S_oct = 64,
_ZSt8_S_right = 128,
_ZSt13_S_scientific = 256,
_ZSt11_S_showbase = 512,
_ZSt12_S_showpoint = 1024,
_ZSt10_S_showpos = 2048,
_ZSt9_S_skipws = 4096,
_ZSt10_S_unitbuf = 8192,
_ZSt12_S_uppercase = 16384,
_ZSt14_S_adjustfield = 176,
_ZSt12_S_basefield = 74,
_ZSt13_S_floatfield = 260,
_ZSt19_S_ios_fmtflags_end = 65536,
_ZSt19_S_ios_fmtflags_max = 2147483647,
_ZSt19_S_ios_fmtflags_min = (-2147483647-1)};
# 105 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt13_Ios_Openmode {

_ZSt6_S_app = 1,
_ZSt6_S_ate,
_ZSt6_S_bin = 4,
_ZSt5_S_in = 8,
_ZSt6_S_out = 16,
_ZSt8_S_trunc = 32,
_ZSt19_S_ios_openmode_end = 65536,
_ZSt19_S_ios_openmode_max = 2147483647,
_ZSt19_S_ios_openmode_min = (-2147483647-1)};
# 147 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt12_Ios_Iostate {

_ZSt10_S_goodbit,
_ZSt9_S_badbit,
_ZSt9_S_eofbit,
_ZSt10_S_failbit = 4,
_ZSt18_S_ios_iostate_end = 65536,
_ZSt18_S_ios_iostate_max = 2147483647,
_ZSt18_S_ios_iostate_min = (-2147483647-1)};
# 187 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt12_Ios_Seekdir {

_ZSt6_S_beg,
_ZSt6_S_cur,
_ZSt6_S_end,
_ZSt18_S_ios_seekdir_end = 65536};
# 425 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZNSt8ios_base5eventE {

_ZNSt8ios_base11erase_eventE,
_ZNSt8ios_base11imbue_eventE,
_ZNSt8ios_base13copyfmt_eventE};
# 466 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base14_Callback_listE;
# 505 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE;
# 517 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZNSt8ios_baseUt_E { _ZNSt8ios_base18_S_local_word_sizeE = 8};
# 539 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base4InitE;
# 205 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt8ios_base;
# 120 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSt19istreambuf_iteratorIcSt11char_traitsIcEE;


struct _ZSt19ostreambuf_iteratorIcSt11char_traitsIcEE;
# 80 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSt15basic_streambufIcSt11char_traitsIcEE;
# 41 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/ctype_base.h" 3
struct _ZSt10ctype_base;
# 674 "/usr/include/c++/4.8/bits/locale_facets.h" 3
struct _ZSt5ctypeIcE;
# 1524 "/usr/include/c++/4.8/bits/locale_facets.h" 3
enum _ZNSt10__num_baseUt_E {
_ZNSt10__num_base9_S_ominusE,
_ZNSt10__num_base8_S_oplusE,
_ZNSt10__num_base5_S_oxE,
_ZNSt10__num_base5_S_oXE,
_ZNSt10__num_base10_S_odigitsE,
_ZNSt10__num_base14_S_odigits_endE = 20,
_ZNSt10__num_base11_S_oudigitsE = 20,
_ZNSt10__num_base15_S_oudigits_endE = 36,
_ZNSt10__num_base5_S_oeE = 18,
_ZNSt10__num_base5_S_oEE = 34,
_ZNSt10__num_base7_S_oendE = 36};
# 1550 "/usr/include/c++/4.8/bits/locale_facets.h" 3
enum _ZNSt10__num_baseUt0_E {
_ZNSt10__num_base9_S_iminusE,
_ZNSt10__num_base8_S_iplusE,
_ZNSt10__num_base5_S_ixE,
_ZNSt10__num_base5_S_iXE,
_ZNSt10__num_base8_S_izeroE,
_ZNSt10__num_base5_S_ieE = 18,
_ZNSt10__num_base5_S_iEE = 24,
_ZNSt10__num_base7_S_iendE = 26};
# 1915 "/usr/include/c++/4.8/bits/locale_facets.h" 3
struct _ZSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE;
# 2254 "/usr/include/c++/4.8/bits/locale_facets.h" 3
struct _ZSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE;
# 77 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSt9basic_iosIcSt11char_traitsIcEE;
# 86 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSo;
# 49 "/usr/include/c++/4.8/bits/codecvt.h" 3
enum _ZNSt12codecvt_base6resultE {

_ZNSt12codecvt_base2okE,
_ZNSt12codecvt_base7partialE,
_ZNSt12codecvt_base5errorE,
_ZNSt12codecvt_base6noconvE};
# 68 "/usr/include/c++/4.8/bits/stl_bvector.h" 3
enum _ZStUt_ { _ZSt11_S_word_bit = 64};
# 2201 "/usr/include/c++/4.8/bits/stl_algo.h" 3
enum _ZStUt0_ { _ZSt12_S_threshold = 16};
# 3375 "/usr/include/c++/4.8/bits/stl_algo.h" 3
enum _ZStUt1_ { _ZSt13_S_chunk_size = 7};
# 309 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt20__is_normal_iteratorIPmEUt_E { _ZNSt20__is_normal_iteratorIPmE7__valueE};
# 260 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIiEUt_E { _ZNSt13__is_floatingIiE7__valueE};
# 98 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__traitorISt12__is_integerIiESt13__is_floatingIiEEUt_E { _ZNSt9__traitorISt12__is_integerIiESt13__is_floatingIiEE7__valueE = 1};
# 292 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_pointerIiEUt_E { _ZNSt12__is_pointerIiE7__valueE};
# 98 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__traitorISt15__is_arithmeticIiESt12__is_pointerIiEEUt_E { _ZNSt9__traitorISt15__is_arithmeticIiESt12__is_pointerIiEE7__valueE = 1};
# 153 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_;
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
# 39 "/usr/include/xlocale.h" 3
typedef struct __locale_struct *__locale_t;
# 48 "/usr/include/stdio.h" 3
typedef struct _IO_FILE FILE;
# 32 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/atomic_word.h" 3
typedef int _Atomic_word;
# 187 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++config.h" 3
typedef long _ZSt9ptrdiff_t;
# 98 "/usr/include/c++/4.8/bits/postypes.h" 3
typedef _ZSt9ptrdiff_t _ZSt10streamsize;
# 136 "/usr/include/c++/4.8/iosfwd" 3
typedef struct _ZSo _ZSt7ostream;
# 62 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++locale.h" 3
typedef __locale_t _ZSt10__c_locale;
# 338 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5facetE { const long *__vptr;
# 344 "/usr/include/c++/4.8/bits/locale_classes.h" 3
_Atomic_word _M_refcount;char __nv_no_debug_dummy_end_padding_0[4];};
# 338 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct __SO__NSt6locale5facetE { const long *__vptr;
# 344 "/usr/include/c++/4.8/bits/locale_classes.h" 3
_Atomic_word _M_refcount;};
# 62 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZSt6locale {
# 280 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5_ImplE *_M_impl;};
# 261 "/usr/include/c++/4.8/bits/ios_base.h" 3
typedef enum _ZSt13_Ios_Fmtflags _ZNSt8ios_base8fmtflagsE;
# 336 "/usr/include/c++/4.8/bits/ios_base.h" 3
typedef enum _ZSt12_Ios_Iostate _ZNSt8ios_base7iostateE;
# 505 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE {

void *_M_pword;
long _M_iword;};
# 539 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base4InitE {char __nv_no_debug_dummy_end_padding_0;};
# 205 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt8ios_base { const long *__vptr;
# 458 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10streamsize _M_precision;
_ZSt10streamsize _M_width;
_ZNSt8ios_base8fmtflagsE _M_flags;
_ZNSt8ios_base7iostateE _M_exception;
_ZNSt8ios_base7iostateE _M_streambuf_state;
# 496 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base14_Callback_listE *_M_callbacks;
# 513 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE _M_word_zero;




struct _ZNSt8ios_base6_WordsE _M_local_word[8];


int _M_word_size;
struct _ZNSt8ios_base6_WordsE *_M_word;
# 528 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt6locale _M_ios_locale;};
# 129 "/usr/include/c++/4.8/streambuf" 3
typedef char _ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE;
typedef struct _ZSt11char_traitsIcE _ZNSt15basic_streambufIcSt11char_traitsIcEE11traits_typeE;
# 80 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSt15basic_streambufIcSt11char_traitsIcEE { const long *__vptr;
# 184 "/usr/include/c++/4.8/streambuf" 3
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_in_beg;
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_in_cur;
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_in_end;
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_out_beg;
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_out_cur;
_ZNSt15basic_streambufIcSt11char_traitsIcEE9char_typeE *_M_out_end;


struct _ZSt6locale _M_buf_locale;};
# 44 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/ctype_base.h" 3
typedef const int *_ZNSt10ctype_base9__to_typeE;



typedef unsigned short _ZNSt10ctype_base4maskE;
# 41 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/ctype_base.h" 3
struct _ZSt10ctype_base {char __nv_no_debug_dummy_end_padding_0;};
# 679 "/usr/include/c++/4.8/bits/locale_facets.h" 3
typedef char _ZNSt5ctypeIcE9char_typeE;
# 674 "/usr/include/c++/4.8/bits/locale_facets.h" 3
struct _ZSt5ctypeIcE {  const long *__b_NSt6locale5facetE___vptr;
# 344 "/usr/include/c++/4.8/bits/locale_classes.h" 3
_Atomic_word __b_NSt6locale5facetE__M_refcount;
# 683 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZSt10__c_locale _M_c_locale_ctype;
char _M_del;
_ZNSt10ctype_base9__to_typeE _M_toupper;
_ZNSt10ctype_base9__to_typeE _M_tolower;
const _ZNSt10ctype_base4maskE *_M_table;
char _M_widen_ok;
char _M_widen[256];
char _M_narrow[256];
char _M_narrow_ok;char __nv_no_debug_dummy_end_padding_0[6];};
# 75 "/usr/include/c++/4.8/bits/basic_ios.h" 3
typedef char _ZNSt9basic_iosIcSt11char_traitsIcEE9char_typeE;
# 86 "/usr/include/c++/4.8/bits/basic_ios.h" 3
typedef struct _ZSt5ctypeIcE _ZNSt9basic_iosIcSt11char_traitsIcEE12__ctype_typeE;

typedef struct _ZSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE _ZNSt9basic_iosIcSt11char_traitsIcEE14__num_put_typeE;

typedef struct _ZSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE _ZNSt9basic_iosIcSt11char_traitsIcEE14__num_get_typeE;
# 77 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSt9basic_iosIcSt11char_traitsIcEE { struct _ZSt8ios_base __b_St8ios_base;
# 95 "/usr/include/c++/4.8/bits/basic_ios.h" 3
struct _ZSo *_M_tie;
_ZNSt9basic_iosIcSt11char_traitsIcEE9char_typeE _M_fill;
char _M_fill_init;
struct _ZSt15basic_streambufIcSt11char_traitsIcEE *_M_streambuf;


const _ZNSt9basic_iosIcSt11char_traitsIcEE12__ctype_typeE *_M_ctype;

const _ZNSt9basic_iosIcSt11char_traitsIcEE14__num_put_typeE *_M_num_put;

const _ZNSt9basic_iosIcSt11char_traitsIcEE14__num_get_typeE *_M_num_get;};
# 62 "/usr/include/c++/4.8/ostream" 3
typedef char _ZNSo9char_typeE;
# 71 "/usr/include/c++/4.8/ostream" 3
typedef struct _ZSo _ZNSo14__ostream_typeE;
# 86 "/usr/include/c++/4.8/iosfwd" 3
struct _ZSo { const long *__vptr; struct _ZSt9basic_iosIcSt11char_traitsIcEE __v_St9basic_iosIcSt11char_traitsIcEE;};
# 153 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_ { long double __l; int __i[3];};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 251 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaDeviceReset(void);
# 1410 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaGetDeviceProperties(struct cudaDeviceProp *, int);
# 2193 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaEventCreate(struct CUevent_st **);
# 2322 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaEventSynchronize(struct CUevent_st *);
# 2388 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaEventElapsedTime(float *, struct CUevent_st *, struct CUevent_st *);
# 2782 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaConfigureCall(struct dim3, struct dim3, size_t, struct CUstream_st *);
# 2993 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaMallocHost(void **, size_t);
# 3121 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaFreeHost(void *);
# 3999 "/usr/local/cuda/include/cuda_runtime_api.h"
extern enum cudaError cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
# 85 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 3
extern int __fprintf_chk(FILE *__restrict__, int, const char *__restrict__, ...);

extern int __printf_chk(int, const char *__restrict__, ...);
# 543 "/usr/include/stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void exit(int);
# 36 "/usr/local/cuda/samples/common/inc/helper_cuda.h"
static const char *_Z17_cudaGetErrorEnum9cudaError(enum cudaError);
# 967 "/usr/local/cuda/samples/common/inc/helper_cuda.h"
extern  __attribute__((__weak__)) /* COMDAT group: _Z5checkI9cudaErrorEvT_PKcS3_i */ void _Z5checkI9cudaErrorEvT_PKcS3_i(enum cudaError, const char *const, const char *const, const int);
# 21 "cuda_main.cu"
extern void _Z6init2DPfiif(float *, int, int, float);
# 30 "cuda_main.cu"
extern void _Z7print2DPfii(float *, int, int);
# 41 "cuda_main.cu"
extern void _Z6init1DPfif(float *, int, float);
# 48 "cuda_main.cu"
extern void _Z7print1DPfi(float *, int);
# 57 "cuda_main.cu"
extern void _Z11d2h_print1dPfS_i(float *, float *, const int);
# 66 "cuda_main.cu"
extern int _Z5checkPfS_ii(float *, float *, const int, const int);
# 84 "cuda_main.cu"
extern void _Z8h2d_copyPfS_i(float *, float *, const int);
# 275 "cuda_main.cu"
extern void _Z8test_v1aii(int, int);
# 355 "cuda_main.cu"
extern int main(int, char **);
extern int __cudaSetupArgSimple();
extern int __cudaLaunch();
# 543 "/usr/include/c++/4.8/bits/ios_base.h" 3
extern __attribute__((visibility("default"))) void _ZNSt8ios_base4InitC1Ev(struct _ZNSt8ios_base4InitE *const);
extern __attribute__((visibility("default"))) void _ZNSt8ios_base4InitD1Ev(struct _ZNSt8ios_base4InitE *const);
# 865 "/usr/include/c++/4.8/bits/locale_facets.h" 3
extern  __attribute__((__weak__)) /* COMDAT group: _ZNKSt5ctypeIcE5widenEc */ __inline__ __attribute__((visibility("default"))) _ZNSt5ctypeIcE9char_typeE _ZNKSt5ctypeIcE5widenEc(const struct _ZSt5ctypeIcE *const, char);
# 1159 "/usr/include/c++/4.8/bits/locale_facets.h" 3
extern __attribute__((visibility("default"))) void _ZNKSt5ctypeIcE13_M_widen_initEv(const struct _ZSt5ctypeIcE *const);
# 142 "/usr/include/c++/4.8/bits/basic_ios.h" 3
extern __attribute__((visibility("default"))) void _ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(struct _ZSt9basic_iosIcSt11char_traitsIcEE *const, _ZNSt8ios_base7iostateE);
# 108 "/usr/include/c++/4.8/ostream" 3
extern __inline__ __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSolsEPFRSoS_E(struct _ZSo *const, _ZNSo14__ostream_typeE *(*)(_ZNSo14__ostream_typeE *));
# 224 "/usr/include/c++/4.8/ostream" 3
extern __inline__ __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSolsEf(struct _ZSo *const, float);
# 303 "/usr/include/c++/4.8/ostream" 3
extern __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSo3putEc(struct _ZSo *const, _ZNSo9char_typeE);
# 348 "/usr/include/c++/4.8/ostream" 3
extern __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSo5flushEv(struct _ZSo *const);
# 389 "/usr/include/c++/4.8/ostream" 3
extern __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSo9_M_insertIdEERSoT_(struct _ZSo *const, double);
# 56 "/usr/include/c++/4.8/bits/functexcept.h" 3
extern __attribute__((__noreturn__)) __attribute__((visibility("default"))) void _ZSt16__throw_bad_castv(void);
# 76 "/usr/include/c++/4.8/bits/ostream_insert.h" 3
extern struct _ZSo *_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(struct _ZSo *, const char *, _ZSt10streamsize);
# 564 "/usr/include/c++/4.8/ostream" 3
extern __inline__ struct _ZSo *_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(struct _ZSo *);
# 530 "/usr/include/c++/4.8/ostream" 3
extern __inline__ struct _ZSo *_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(struct _ZSo *, const char *);
extern void __nv_dummy_param_ref();
extern void __nv_save_fatbinhandle_for_managed_rt();
extern int __cudaRegisterEntry();
extern int __cudaRegisterBinary();
static void __sti___17_cuda_main_cpp1_ii_8e0b8fcd(void) __attribute__((__constructor__));
extern int __cxa_atexit();
# 170 "/usr/include/stdio.h" 3
extern struct _IO_FILE *stderr;
# 61 "/usr/include/c++/4.8/iostream" 3
extern _ZSt7ostream _ZSt4cout __attribute__((visibility("default")));
# 74 "/usr/include/c++/4.8/iostream" 3
static struct _ZNSt8ios_base4InitE _ZSt8__ioinit __attribute__((visibility("default"))) = {0};
extern void *__dso_handle __attribute__((visibility("hidden")));
# 36 "/usr/local/cuda/samples/common/inc/helper_cuda.h"
static const char *_Z17_cudaGetErrorEnum9cudaError( enum cudaError error)
{
switch ((int)error)
{
case 0:
return (const char *)("cudaSuccess");

case 1:
return (const char *)("cudaErrorMissingConfiguration");

case 2:
return (const char *)("cudaErrorMemoryAllocation");

case 3:
return (const char *)("cudaErrorInitializationError");

case 4:
return (const char *)("cudaErrorLaunchFailure");

case 5:
return (const char *)("cudaErrorPriorLaunchFailure");

case 6:
return (const char *)("cudaErrorLaunchTimeout");

case 7:
return (const char *)("cudaErrorLaunchOutOfResources");

case 8:
return (const char *)("cudaErrorInvalidDeviceFunction");

case 9:
return (const char *)("cudaErrorInvalidConfiguration");

case 10:
return (const char *)("cudaErrorInvalidDevice");

case 11:
return (const char *)("cudaErrorInvalidValue");

case 12:
return (const char *)("cudaErrorInvalidPitchValue");

case 13:
return (const char *)("cudaErrorInvalidSymbol");

case 14:
return (const char *)("cudaErrorMapBufferObjectFailed");

case 15:
return (const char *)("cudaErrorUnmapBufferObjectFailed");

case 16:
return (const char *)("cudaErrorInvalidHostPointer");

case 17:
return (const char *)("cudaErrorInvalidDevicePointer");

case 18:
return (const char *)("cudaErrorInvalidTexture");

case 19:
return (const char *)("cudaErrorInvalidTextureBinding");

case 20:
return (const char *)("cudaErrorInvalidChannelDescriptor");

case 21:
return (const char *)("cudaErrorInvalidMemcpyDirection");

case 22:
return (const char *)("cudaErrorAddressOfConstant");

case 23:
return (const char *)("cudaErrorTextureFetchFailed");

case 24:
return (const char *)("cudaErrorTextureNotBound");

case 25:
return (const char *)("cudaErrorSynchronizationError");

case 26:
return (const char *)("cudaErrorInvalidFilterSetting");

case 27:
return (const char *)("cudaErrorInvalidNormSetting");

case 28:
return (const char *)("cudaErrorMixedDeviceExecution");

case 29:
return (const char *)("cudaErrorCudartUnloading");

case 30:
return (const char *)("cudaErrorUnknown");

case 31:
return (const char *)("cudaErrorNotYetImplemented");

case 32:
return (const char *)("cudaErrorMemoryValueTooLarge");

case 33:
return (const char *)("cudaErrorInvalidResourceHandle");

case 34:
return (const char *)("cudaErrorNotReady");

case 35:
return (const char *)("cudaErrorInsufficientDriver");

case 36:
return (const char *)("cudaErrorSetOnActiveProcess");

case 37:
return (const char *)("cudaErrorInvalidSurface");

case 38:
return (const char *)("cudaErrorNoDevice");

case 39:
return (const char *)("cudaErrorECCUncorrectable");

case 40:
return (const char *)("cudaErrorSharedObjectSymbolNotFound");

case 41:
return (const char *)("cudaErrorSharedObjectInitFailed");

case 42:
return (const char *)("cudaErrorUnsupportedLimit");

case 43:
return (const char *)("cudaErrorDuplicateVariableName");

case 44:
return (const char *)("cudaErrorDuplicateTextureName");

case 45:
return (const char *)("cudaErrorDuplicateSurfaceName");

case 46:
return (const char *)("cudaErrorDevicesUnavailable");

case 47:
return (const char *)("cudaErrorInvalidKernelImage");

case 48:
return (const char *)("cudaErrorNoKernelImageForDevice");

case 49:
return (const char *)("cudaErrorIncompatibleDriverContext");

case 50:
return (const char *)("cudaErrorPeerAccessAlreadyEnabled");

case 51:
return (const char *)("cudaErrorPeerAccessNotEnabled");

case 54:
return (const char *)("cudaErrorDeviceAlreadyInUse");

case 55:
return (const char *)("cudaErrorProfilerDisabled");

case 56:
return (const char *)("cudaErrorProfilerNotInitialized");

case 57:
return (const char *)("cudaErrorProfilerAlreadyStarted");

case 58:
return (const char *)("cudaErrorProfilerAlreadyStopped");


case 59:
return (const char *)("cudaErrorAssert");

case 60:
return (const char *)("cudaErrorTooManyPeers");

case 61:
return (const char *)("cudaErrorHostMemoryAlreadyRegistered");

case 62:
return (const char *)("cudaErrorHostMemoryNotRegistered");


case 63:
return (const char *)("cudaErrorOperatingSystem");

case 64:
return (const char *)("cudaErrorPeerAccessUnsupported");

case 65:
return (const char *)("cudaErrorLaunchMaxDepthExceeded");

case 66:
return (const char *)("cudaErrorLaunchFileScopedTex");

case 67:
return (const char *)("cudaErrorLaunchFileScopedSurf");

case 68:
return (const char *)("cudaErrorSyncDepthExceeded");

case 69:
return (const char *)("cudaErrorLaunchPendingCountExceeded");

case 70:
return (const char *)("cudaErrorNotPermitted");

case 71:
return (const char *)("cudaErrorNotSupported");


case 72:
return (const char *)("cudaErrorHardwareStackError");

case 73:
return (const char *)("cudaErrorIllegalInstruction");

case 74:
return (const char *)("cudaErrorMisalignedAddress");

case 75:
return (const char *)("cudaErrorInvalidAddressSpace");

case 76:
return (const char *)("cudaErrorInvalidPc");

case 77:
return (const char *)("cudaErrorIllegalAddress");


case 78:
return (const char *)("cudaErrorInvalidPtx");

case 79:
return (const char *)("cudaErrorInvalidGraphicsContext");

case 127:
return (const char *)("cudaErrorStartupFailure");

case 10000:
return (const char *)("cudaErrorApiFailureBase");
}

return (const char *)("<unknown>");
}
# 967 "/usr/local/cuda/samples/common/inc/helper_cuda.h"
 __attribute__((__weak__)) /* COMDAT group: _Z5checkI9cudaErrorEvT_PKcS3_i */ void _Z5checkI9cudaErrorEvT_PKcS3_i( enum cudaError result,  const char *const func,  const char *const file,  const int line)
{
if (result)
{
fprintf(stderr, ((const char *)"CUDA error at %s:%d code=%d(%s) \"%s\" \n"), file, line, ((unsigned)result), (_Z17_cudaGetErrorEnum9cudaError(result)), func);

cudaDeviceReset();

exit(1);
} 
}
# 21 "cuda_main.cu"
void _Z6init2DPfiif( float *array,  int rows,  int cols,  float value)
{  {
 int i;
# 23 "cuda_main.cu"
i = 0; for (; (i < rows); i++) { {
 int j;
# 24 "cuda_main.cu"
j = 0; for (; (j < cols); j++) {
(array[((i * cols) + j)]) = value;
} }
} } 
}

void _Z7print2DPfii( float *array,  int rows,  int cols)
{
printf(((const char *)"\n")); {
 int i;
# 33 "cuda_main.cu"
i = 0; for (; (i < rows); i++) { {
 int j;
# 34 "cuda_main.cu"
j = 0; for (; (j < cols); j++) {
printf(((const char *)"%5.3f "), ((double)(array[((i * cols) + j)])));
} }
printf(((const char *)"\n"));
} } 
}

void _Z6init1DPfif( float *data,  int len,  float value)
{  {
 int i;
# 43 "cuda_main.cu"
i = 0; for (; (i < len); i++) {
(data[i]) = value;
} } 
}

void _Z7print1DPfi( float *data,  int len)
{
printf(((const char *)"\n")); {
 int i;
# 51 "cuda_main.cu"
i = 0; for (; (i < len); i++) {
printf(((const char *)"%5.3f "), ((double)(data[i])));
} }
printf(((const char *)"\n")); 
}

void _Z11d2h_print1dPfS_i( float *d_data,  float *h_data,  const int rows)
{
cudaMemcpy(((void *)h_data), ((const void *)d_data), (4UL * ((unsigned long)rows)), cudaMemcpyDeviceToHost); {
 int i;
# 60 "cuda_main.cu"
i = 0; for (; (i < rows); i++) {
printf(((const char *)"%f "), ((double)(h_data[i])));
} }
printf(((const char *)"\n")); 
}

int _Z5checkPfS_ii( float *d_data,  float *h_data,  const int rows,  const int cols)
{
 float __cuda_local_var_75823_8_non_const_cpu;


 int __cuda_local_var_75826_6_non_const_correct;
# 68 "cuda_main.cu"
__cuda_local_var_75823_8_non_const_cpu = ((float)(((double)cols) * (0.02000000000000000042)));
cudaMemcpy(((void *)h_data), ((const void *)d_data), (4UL * ((unsigned long)rows)), cudaMemcpyDeviceToHost);

__cuda_local_var_75826_6_non_const_correct = 1; {
 int i;
# 72 "cuda_main.cu"
i = 0; for (; (i < rows); i++) {  float __T20;

if (((double)((__T20 = ((h_data[i]) - __cuda_local_var_75823_8_non_const_cpu)) , (__builtin_fabsf(__T20)))) > (1.000000000000000082e-05)) {
fprintf(stderr, ((const char *)"result doesn\'t match! pos : %d, gpu %12.8f , cpu %12.8f\n"), i, ((double)(h_data[i])), ((double)__cuda_local_var_75823_8_non_const_cpu));

__cuda_local_var_75826_6_non_const_correct = 0;
goto __T21;
}
} } __T21:;
return __cuda_local_var_75826_6_non_const_correct;
}

void _Z8h2d_copyPfS_i( float *h_data,  float *d_data,  const int len)
{
cudaMemcpy(((void *)d_data), ((const void *)h_data), (4UL * ((unsigned long)len)), cudaMemcpyHostToDevice); 
}
# 275 "cuda_main.cu"
void _Z8test_v1aii( int rows,  int cols)
{  unsigned __T25;
 struct CUevent_st *__cuda_local_var_75919_14_non_const_startEvent;
# 277 "cuda_main.cu"
 struct CUevent_st *__cuda_local_var_75919_26_non_const_stopEvent;




 float *__cuda_local_var_75924_9_non_const_A;
 float *__cuda_local_var_75925_9_non_const_B;
 float *__cuda_local_var_75926_9_non_const_C;
# 295 "cuda_main.cu"
 float *__cuda_local_var_75937_9_non_const_d_A;
 float *__cuda_local_var_75938_9_non_const_d_B;
 float *__cuda_local_var_75939_9_non_const_d_C;
# 314 "cuda_main.cu"
 struct dim3 __cuda_local_var_75956_10_non_const_Blk_config;
 struct dim3 __cuda_local_var_75957_10_non_const_Grd_config;
# 331 "cuda_main.cu"
 float __cuda_local_var_75973_8_non_const_milliseconds;
# 278 "cuda_main.cu"
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaEventCreate((&__cuda_local_var_75919_14_non_const_startEvent))), ((const char *)"cudaEventCreate(&startEvent)"), ((const char *)"cuda_main.cu"), 278);
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaEventCreate((&__cuda_local_var_75919_26_non_const_stopEvent))), ((const char *)"cudaEventCreate(&stopEvent)"), ((const char *)"cuda_main.cu"), 279);
# 285 "cuda_main.cu"
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMallocHost(((void **)(&__cuda_local_var_75924_9_non_const_A)), (((unsigned long)(rows * cols)) * 4UL))), ((const char *)"cudaMallocHost((void **)&A, rows * cols * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 285);
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMallocHost(((void **)(&__cuda_local_var_75925_9_non_const_B)), (((unsigned long)cols) * 4UL))), ((const char *)"cudaMallocHost((void **)&B, cols * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 286);
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMallocHost(((void **)(&__cuda_local_var_75926_9_non_const_C)), (((unsigned long)rows) * 4UL))), ((const char *)"cudaMallocHost((void **)&C, rows * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 287);

_Z6init2DPfiif(__cuda_local_var_75924_9_non_const_A, rows, cols, (0.200000003F));
_Z6init1DPfif(__cuda_local_var_75925_9_non_const_B, cols, (0.1000000015F));
# 298 "cuda_main.cu"
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMalloc(((void **)(&__cuda_local_var_75937_9_non_const_d_A)), (((unsigned long)(rows * cols)) * 4UL))), ((const char *)"cudaMalloc((void **)&d_A, rows * cols * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 298);
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMalloc(((void **)(&__cuda_local_var_75938_9_non_const_d_B)), (((unsigned long)cols) * 4UL))), ((const char *)"cudaMalloc((void **)&d_B, cols * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 299);
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaMalloc(((void **)(&__cuda_local_var_75939_9_non_const_d_C)), (((unsigned long)rows) * 4UL))), ((const char *)"cudaMalloc((void **)&d_C, rows * FLT_SIZE)"), ((const char *)"cuda_main.cu"), 300);

_Z8h2d_copyPfS_i(__cuda_local_var_75924_9_non_const_A, __cuda_local_var_75937_9_non_const_d_A, (rows * cols));
_Z8h2d_copyPfS_i(__cuda_local_var_75925_9_non_const_B, __cuda_local_var_75938_9_non_const_d_B, cols);



cudaEventRecord(__cuda_local_var_75919_14_non_const_startEvent, ((struct CUstream_st *)0LL));
# 314 "cuda_main.cu"
{
# 421 "/usr/local/cuda/include/vector_types.h"
(__cuda_local_var_75956_10_non_const_Blk_config.x) = 128U; (__cuda_local_var_75956_10_non_const_Blk_config.y) = 1U; (__cuda_local_var_75956_10_non_const_Blk_config.z) = 1U;
# 314 "cuda_main.cu"
}
{ __T25 = ((unsigned)(((rows + 128) - 1) / 128));
# 421 "/usr/local/cuda/include/vector_types.h"
{ (__cuda_local_var_75957_10_non_const_Grd_config.x) = __T25; (__cuda_local_var_75957_10_non_const_Grd_config.y) = 1U; (__cuda_local_var_75957_10_non_const_Grd_config.z) = 1U; }
# 315 "cuda_main.cu"
}




(cudaConfigureCall(__cuda_local_var_75957_10_non_const_Grd_config, __cuda_local_var_75956_10_non_const_Blk_config, 0UL, ((struct CUstream_st *)0LL))) ? ((void)0) : (__device_stub__Z17kernel_sgemv_128biiiPKfS0_Pf(rows, cols, (((cols + 4) - 1) / 4), ((const float *)__cuda_local_var_75937_9_non_const_d_A), ((const float *)__cuda_local_var_75938_9_non_const_d_B), __cuda_local_var_75939_9_non_const_d_C));
# 328 "cuda_main.cu"
cudaEventRecord(__cuda_local_var_75919_26_non_const_stopEvent, ((struct CUstream_st *)0LL));
cudaEventSynchronize(__cuda_local_var_75919_26_non_const_stopEvent);

__cuda_local_var_75973_8_non_const_milliseconds = (0.0F);
cudaEventElapsedTime((&__cuda_local_var_75973_8_non_const_milliseconds), __cuda_local_var_75919_14_non_const_startEvent, __cuda_local_var_75919_26_non_const_stopEvent);
_ZNSolsEPFRSoS_E((_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((_ZNSolsEf((&_ZSt4cout), __cuda_local_var_75973_8_non_const_milliseconds)), ((const char *)" (ms)"))), _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);


if (_Z5checkPfS_ii(__cuda_local_var_75939_9_non_const_d_C, __cuda_local_var_75926_9_non_const_C, rows, cols)) {
printf(((const char *)"success!\n"));
}



if (__cuda_local_var_75924_9_non_const_A != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFreeHost(((void *)__cuda_local_var_75924_9_non_const_A))), ((const char *)"cudaFreeHost(A)"), ((const char *)"cuda_main.cu"), 342); }
if (__cuda_local_var_75925_9_non_const_B != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFreeHost(((void *)__cuda_local_var_75925_9_non_const_B))), ((const char *)"cudaFreeHost(B)"), ((const char *)"cuda_main.cu"), 343); }
if (__cuda_local_var_75926_9_non_const_C != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFreeHost(((void *)__cuda_local_var_75926_9_non_const_C))), ((const char *)"cudaFreeHost(C)"), ((const char *)"cuda_main.cu"), 344); }

if (__cuda_local_var_75937_9_non_const_d_A != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFree(((void *)__cuda_local_var_75937_9_non_const_d_A))), ((const char *)"cudaFree(d_A)"), ((const char *)"cuda_main.cu"), 346); }
if (__cuda_local_var_75938_9_non_const_d_B != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFree(((void *)__cuda_local_var_75938_9_non_const_d_B))), ((const char *)"cudaFree(d_B)"), ((const char *)"cuda_main.cu"), 347); }
if (__cuda_local_var_75939_9_non_const_d_C != ((float *)0LL)) { _Z5checkI9cudaErrorEvT_PKcS3_i((cudaFree(((void *)__cuda_local_var_75939_9_non_const_d_C))), ((const char *)"cudaFree(d_C)"), ((const char *)"cuda_main.cu"), 348); }

cudaDeviceReset(); 
}



int main( int argc,  char **argv) {

 struct cudaDeviceProp __cuda_local_var_75999_17_non_const_prop;
_Z5checkI9cudaErrorEvT_PKcS3_i((cudaGetDeviceProperties((&__cuda_local_var_75999_17_non_const_prop), 0)), ((const char *)"cudaGetDeviceProperties(&prop, 0)"), ((const char *)"cuda_main.cu"), 358);
printf(((const char *)"Device: %s\n"), ((__cuda_local_var_75999_17_non_const_prop.name)));


_Z8test_v1aii(100, 100);
# 380 "cuda_main.cu"
return 0;
}
__asm__(".align 2");
# 865 "/usr/include/c++/4.8/bits/locale_facets.h" 3
 __attribute__((__weak__)) /* COMDAT group: _ZNKSt5ctypeIcE5widenEc */ __inline__ __attribute__((visibility("default"))) _ZNSt5ctypeIcE9char_typeE _ZNKSt5ctypeIcE5widenEc( const struct _ZSt5ctypeIcE *const this,  char __c)
{
if (((struct _ZSt5ctypeIcE *)this)->_M_widen_ok) {
return ((((struct _ZSt5ctypeIcE *)this)->_M_widen))[((unsigned char)__c)]; }
_ZNKSt5ctypeIcE13_M_widen_initEv(this);
return (*((_ZNSt5ctypeIcE9char_typeE (**)(const struct _ZSt5ctypeIcE *const, char))((((*(struct __SO__NSt6locale5facetE *)&(this->__b_NSt6locale5facetE___vptr))).__vptr) + 6)))(this, __c);
}
__asm__(".align 2");
# 108 "/usr/include/c++/4.8/ostream" 3
extern __inline__ __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSolsEPFRSoS_E( struct _ZSo *const this,  _ZNSo14__ostream_typeE *(*__pf)(_ZNSo14__ostream_typeE *))
{



return __pf(this);
}
__asm__(".align 2");
# 224 "/usr/include/c++/4.8/ostream" 3
extern __inline__ __attribute__((visibility("default"))) _ZNSo14__ostream_typeE *_ZNSolsEf( struct _ZSo *const this,  float __f)
{


return _ZNSo9_M_insertIdEERSoT_(this, ((double)__f));
}
# 564 "/usr/include/c++/4.8/ostream" 3
extern __inline__ struct _ZSo *_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_( struct _ZSo *__os)
{  const struct _ZSt9basic_iosIcSt11char_traitsIcEE *__T26;
 const _ZNSt9basic_iosIcSt11char_traitsIcEE12__ctype_typeE *__T27;
 struct _ZSo *__T28;
# 565 "/usr/include/c++/4.8/ostream" 3
return (__T28 = (_ZNSo3putEc(__os, ((__T26 = ((const struct _ZSt9basic_iosIcSt11char_traitsIcEE *)((__os) ? ((struct _ZSt9basic_iosIcSt11char_traitsIcEE *)(((char *)__os) + ((__os->__vptr)[(-3L)]))) : ((struct _ZSt9basic_iosIcSt11char_traitsIcEE *)0LL)))) , (_ZNKSt5ctypeIcE5widenEc(((__T27 = (__T26->_M_ctype)) , (((!(__T27)) ? (_ZSt16__throw_bad_castv()) : ((void)0)) , __T27)), ((char)10))))))) , (_ZNSo5flushEv(__T28)); }
# 530 "/usr/include/c++/4.8/ostream" 3
extern __inline__ struct _ZSo *_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( struct _ZSo *__out,  const char *__s)
{  struct _ZSt9basic_iosIcSt11char_traitsIcEE *__T29;
if (!(__s)) {
{ __T29 = ((__out) ? ((struct _ZSt9basic_iosIcSt11char_traitsIcEE *)(((char *)__out) + ((__out->__vptr)[(-3L)]))) : ((struct _ZSt9basic_iosIcSt11char_traitsIcEE *)0LL));
# 152 "/usr/include/c++/4.8/bits/basic_ios.h" 3
{ _ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(__T29, ((enum _ZSt12_Ios_Iostate)(((int)((((const struct _ZSt9basic_iosIcSt11char_traitsIcEE *)__T29)->__b_St8ios_base)._M_streambuf_state)) | ((int)((_ZNSt8ios_base7iostateE)1))))); }
# 533 "/usr/include/c++/4.8/ostream" 3
} } else  {

_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(__out, __s, ((_ZSt10streamsize)(__builtin_strlen(__s)))); }

return __out;
}
static void __sti___17_cuda_main_cpp1_ii_8e0b8fcd(void) {
# 74 "/usr/include/c++/4.8/iostream" 3
_ZNSt8ios_base4InitC1Ev((&_ZSt8__ioinit)); __cxa_atexit(_ZNSt8ios_base4InitD1Ev, ((void *)(&_ZSt8__ioinit)), (&__dso_handle));  }

#include "cuda_main.cudafe1.stub.c"
