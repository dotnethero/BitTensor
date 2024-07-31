/* automatically generated by rust-bindgen 0.69.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _iobuf {
    pub _Placeholder: *mut ::std::os::raw::c_void,
}
#[test]
fn bindgen_test_layout__iobuf() {
    const UNINIT: ::std::mem::MaybeUninit<_iobuf> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<_iobuf>(),
        8usize,
        concat!("Size of: ", stringify!(_iobuf))
    );
    assert_eq!(
        ::std::mem::align_of::<_iobuf>(),
        8usize,
        concat!("Alignment of ", stringify!(_iobuf))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr)._Placeholder) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(_iobuf),
            "::",
            stringify!(_Placeholder)
        )
    );
}
pub type FILE = _iobuf;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorDataType_t {
    CUTENSOR_R_16F = 2,
    CUTENSOR_C_16F = 6,
    CUTENSOR_R_16BF = 14,
    CUTENSOR_C_16BF = 15,
    CUTENSOR_R_32F = 0,
    CUTENSOR_C_32F = 4,
    CUTENSOR_R_64F = 1,
    CUTENSOR_C_64F = 5,
    CUTENSOR_R_4I = 16,
    CUTENSOR_C_4I = 17,
    CUTENSOR_R_4U = 18,
    CUTENSOR_C_4U = 19,
    CUTENSOR_R_8I = 3,
    CUTENSOR_C_8I = 7,
    CUTENSOR_R_8U = 8,
    CUTENSOR_C_8U = 9,
    CUTENSOR_R_16I = 20,
    CUTENSOR_C_16I = 21,
    CUTENSOR_R_16U = 22,
    CUTENSOR_C_16U = 23,
    CUTENSOR_R_32I = 10,
    CUTENSOR_C_32I = 11,
    CUTENSOR_R_32U = 12,
    CUTENSOR_C_32U = 13,
    CUTENSOR_R_64I = 24,
    CUTENSOR_C_64I = 25,
    CUTENSOR_R_64U = 26,
    CUTENSOR_C_64U = 27,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorOperator_t {
    CUTENSOR_OP_IDENTITY = 1,
    CUTENSOR_OP_SQRT = 2,
    CUTENSOR_OP_RELU = 8,
    CUTENSOR_OP_CONJ = 9,
    CUTENSOR_OP_RCP = 10,
    CUTENSOR_OP_SIGMOID = 11,
    CUTENSOR_OP_TANH = 12,
    CUTENSOR_OP_EXP = 22,
    CUTENSOR_OP_LOG = 23,
    CUTENSOR_OP_ABS = 24,
    CUTENSOR_OP_NEG = 25,
    CUTENSOR_OP_SIN = 26,
    CUTENSOR_OP_COS = 27,
    CUTENSOR_OP_TAN = 28,
    CUTENSOR_OP_SINH = 29,
    CUTENSOR_OP_COSH = 30,
    CUTENSOR_OP_ASIN = 31,
    CUTENSOR_OP_ACOS = 32,
    CUTENSOR_OP_ATAN = 33,
    CUTENSOR_OP_ASINH = 34,
    CUTENSOR_OP_ACOSH = 35,
    CUTENSOR_OP_ATANH = 36,
    CUTENSOR_OP_CEIL = 37,
    CUTENSOR_OP_FLOOR = 38,
    CUTENSOR_OP_MISH = 39,
    CUTENSOR_OP_SWISH = 40,
    CUTENSOR_OP_SOFT_PLUS = 41,
    CUTENSOR_OP_SOFT_SIGN = 42,
    CUTENSOR_OP_ADD = 3,
    CUTENSOR_OP_MUL = 5,
    CUTENSOR_OP_MAX = 6,
    CUTENSOR_OP_MIN = 7,
    CUTENSOR_OP_UNKNOWN = 126,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorStatus_t {
    CUTENSOR_STATUS_SUCCESS = 0,
    CUTENSOR_STATUS_NOT_INITIALIZED = 1,
    CUTENSOR_STATUS_ALLOC_FAILED = 3,
    CUTENSOR_STATUS_INVALID_VALUE = 7,
    CUTENSOR_STATUS_ARCH_MISMATCH = 8,
    CUTENSOR_STATUS_MAPPING_ERROR = 11,
    CUTENSOR_STATUS_EXECUTION_FAILED = 13,
    CUTENSOR_STATUS_INTERNAL_ERROR = 14,
    CUTENSOR_STATUS_NOT_SUPPORTED = 15,
    CUTENSOR_STATUS_LICENSE_ERROR = 16,
    CUTENSOR_STATUS_CUBLAS_ERROR = 17,
    CUTENSOR_STATUS_CUDA_ERROR = 18,
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    CUTENSOR_STATUS_IO_ERROR = 21,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorAlgo_t {
    CUTENSOR_ALGO_DEFAULT_PATIENT = -6,
    CUTENSOR_ALGO_GETT = -4,
    CUTENSOR_ALGO_TGETT = -3,
    CUTENSOR_ALGO_TTGT = -2,
    CUTENSOR_ALGO_DEFAULT = -1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorWorksizePreference_t {
    CUTENSOR_WORKSPACE_MIN = 1,
    CUTENSOR_WORKSPACE_DEFAULT = 2,
    CUTENSOR_WORKSPACE_MAX = 3,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorComputeDescriptor {
    _unused: [u8; 0],
}
pub type cutensorComputeDescriptor_t = *mut cutensorComputeDescriptor;
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorOperationDescriptorAttribute_t {
    CUTENSOR_OPERATION_DESCRIPTOR_TAG = 0,
    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1,
    CUTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,
    CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorPlanPreferenceAttribute_t {
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0,
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1,
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
    CUTENSOR_PLAN_PREFERENCE_ALGO = 3,
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4,
    CUTENSOR_PLAN_PREFERENCE_JIT = 5,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorAutotuneMode_t {
    CUTENSOR_AUTOTUNE_MODE_NONE = 0,
    CUTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorJitMode_t {
    CUTENSOR_JIT_MODE_NONE = 0,
    CUTENSOR_JIT_MODE_DEFAULT = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorCacheMode_t {
    CUTENSOR_CACHE_MODE_NONE = 0,
    CUTENSOR_CACHE_MODE_PEDANTIC = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cutensorPlanAttribute_t {
    CUTENSOR_PLAN_REQUIRED_WORKSPACE = 0,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorOperationDescriptor {
    _unused: [u8; 0],
}
pub type cutensorOperationDescriptor_t = *mut cutensorOperationDescriptor;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorPlan {
    _unused: [u8; 0],
}
pub type cutensorPlan_t = *mut cutensorPlan;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorPlanPreference {
    _unused: [u8; 0],
}
pub type cutensorPlanPreference_t = *mut cutensorPlanPreference;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorHandle {
    _unused: [u8; 0],
}
pub type cutensorHandle_t = *mut cutensorHandle;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorTensorDescriptor {
    _unused: [u8; 0],
}
pub type cutensorTensorDescriptor_t = *mut cutensorTensorDescriptor;
pub type cutensorLoggerCallback_t = ::std::option::Option<
    unsafe extern "C" fn(
        logLevel: i32,
        functionName: *const ::std::os::raw::c_char,
        message: *const ::std::os::raw::c_char,
    ),
>;
extern "C" {
    pub fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorHandleResizePlanCache(
        handle: cutensorHandle_t,
        numEntries: u32,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorHandleWritePlanCacheToFile(
        handle: cutensorHandle_t,
        filename: *const ::std::os::raw::c_char,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorHandleReadPlanCacheFromFile(
        handle: cutensorHandle_t,
        filename: *const ::std::os::raw::c_char,
        numCachelinesRead: *mut u32,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorWriteKernelCacheToFile(
        handle: cutensorHandle_t,
        filename: *const ::std::os::raw::c_char,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorReadKernelCacheFromFile(
        handle: cutensorHandle_t,
        filename: *const ::std::os::raw::c_char,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreateTensorDescriptor(
        handle: cutensorHandle_t,
        desc: *mut cutensorTensorDescriptor_t,
        numModes: u32,
        extent: *const i64,
        stride: *const i64,
        dataType: cutensorDataType_t,
        alignmentRequirement: u32,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreateElementwiseTrinary(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        descA: cutensorTensorDescriptor_t,
        modeA: *const i32,
        opA: cutensorOperator_t,
        descB: cutensorTensorDescriptor_t,
        modeB: *const i32,
        opB: cutensorOperator_t,
        descC: cutensorTensorDescriptor_t,
        modeC: *const i32,
        opC: cutensorOperator_t,
        descD: cutensorTensorDescriptor_t,
        modeD: *const i32,
        opAB: cutensorOperator_t,
        opABC: cutensorOperator_t,
        descCompute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorElementwiseTrinaryExecute(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const ::std::os::raw::c_void,
        A: *const ::std::os::raw::c_void,
        beta: *const ::std::os::raw::c_void,
        B: *const ::std::os::raw::c_void,
        gamma: *const ::std::os::raw::c_void,
        C: *const ::std::os::raw::c_void,
        D: *mut ::std::os::raw::c_void,
        stream: cudaStream_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreateElementwiseBinary(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        descA: cutensorTensorDescriptor_t,
        modeA: *const i32,
        opA: cutensorOperator_t,
        descC: cutensorTensorDescriptor_t,
        modeC: *const i32,
        opC: cutensorOperator_t,
        descD: cutensorTensorDescriptor_t,
        modeD: *const i32,
        opAC: cutensorOperator_t,
        descCompute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorElementwiseBinaryExecute(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const ::std::os::raw::c_void,
        A: *const ::std::os::raw::c_void,
        gamma: *const ::std::os::raw::c_void,
        C: *const ::std::os::raw::c_void,
        D: *mut ::std::os::raw::c_void,
        stream: cudaStream_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreatePermutation(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        descA: cutensorTensorDescriptor_t,
        modeA: *const i32,
        opA: cutensorOperator_t,
        descB: cutensorTensorDescriptor_t,
        modeB: *const i32,
        descCompute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorPermute(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const ::std::os::raw::c_void,
        A: *const ::std::os::raw::c_void,
        B: *mut ::std::os::raw::c_void,
        stream: cudaStream_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreateContraction(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        descA: cutensorTensorDescriptor_t,
        modeA: *const i32,
        opA: cutensorOperator_t,
        descB: cutensorTensorDescriptor_t,
        modeB: *const i32,
        opB: cutensorOperator_t,
        descC: cutensorTensorDescriptor_t,
        modeC: *const i32,
        opC: cutensorOperator_t,
        descD: cutensorTensorDescriptor_t,
        modeD: *const i32,
        descCompute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorDestroyOperationDescriptor(
        desc: cutensorOperationDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorOperationDescriptorSetAttribute(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        attr: cutensorOperationDescriptorAttribute_t,
        buf: *const ::std::os::raw::c_void,
        sizeInBytes: usize,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorOperationDescriptorGetAttribute(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        attr: cutensorOperationDescriptorAttribute_t,
        buf: *mut ::std::os::raw::c_void,
        sizeInBytes: usize,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreatePlanPreference(
        handle: cutensorHandle_t,
        pref: *mut cutensorPlanPreference_t,
        algo: cutensorAlgo_t,
        jitMode: cutensorJitMode_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorPlanPreferenceSetAttribute(
        handle: cutensorHandle_t,
        pref: cutensorPlanPreference_t,
        attr: cutensorPlanPreferenceAttribute_t,
        buf: *const ::std::os::raw::c_void,
        sizeInBytes: usize,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorPlanGetAttribute(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        attr: cutensorPlanAttribute_t,
        buf: *mut ::std::os::raw::c_void,
        sizeInBytes: usize,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorEstimateWorkspaceSize(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        planPref: cutensorPlanPreference_t,
        workspacePref: cutensorWorksizePreference_t,
        workspaceSizeEstimate: *mut u64,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreatePlan(
        handle: cutensorHandle_t,
        plan: *mut cutensorPlan_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        workspaceSizeLimit: u64,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorContract(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const ::std::os::raw::c_void,
        A: *const ::std::os::raw::c_void,
        B: *const ::std::os::raw::c_void,
        beta: *const ::std::os::raw::c_void,
        C: *const ::std::os::raw::c_void,
        D: *mut ::std::os::raw::c_void,
        workspace: *mut ::std::os::raw::c_void,
        workspaceSize: u64,
        stream: cudaStream_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorCreateReduction(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        descA: cutensorTensorDescriptor_t,
        modeA: *const i32,
        opA: cutensorOperator_t,
        descC: cutensorTensorDescriptor_t,
        modeC: *const i32,
        opC: cutensorOperator_t,
        descD: cutensorTensorDescriptor_t,
        modeD: *const i32,
        opReduce: cutensorOperator_t,
        descCompute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorReduce(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const ::std::os::raw::c_void,
        A: *const ::std::os::raw::c_void,
        beta: *const ::std::os::raw::c_void,
        C: *const ::std::os::raw::c_void,
        D: *mut ::std::os::raw::c_void,
        workspace: *mut ::std::os::raw::c_void,
        workspaceSize: u64,
        stream: cudaStream_t,
    ) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorGetErrorString(error: cutensorStatus_t) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn cutensorGetVersion() -> usize;
}
extern "C" {
    pub fn cutensorGetCudartVersion() -> usize;
}
extern "C" {
    pub fn cutensorLoggerSetCallback(callback: cutensorLoggerCallback_t) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorLoggerSetFile(file: *mut FILE) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorLoggerOpenFile(logFile: *const ::std::os::raw::c_char) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorLoggerSetLevel(level: i32) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorLoggerSetMask(mask: i32) -> cutensorStatus_t;
}
extern "C" {
    pub fn cutensorLoggerForceDisable() -> cutensorStatus_t;
}
