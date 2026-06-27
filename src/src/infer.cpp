#include "infer.h"

#ifdef _WIN32
#include <onnxruntime/dml_provider_factory.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace
{
std::string trimString(const std::string &value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::wstring widenPath(const std::string &path)
{
#ifdef _WIN32
    if (path.empty()) {
        return {};
    }

    const int wideLength = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        path.data(),
        static_cast<int>(path.size()),
        nullptr,
        0);
    if (wideLength <= 0) {
        return std::wstring(path.begin(), path.end());
    }

    std::wstring result(static_cast<size_t>(wideLength), L'\0');
    MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        path.data(),
        static_cast<int>(path.size()),
        &result[0],
        wideLength);
    return result;
#else
    return std::wstring(path.begin(), path.end());
#endif
}

size_t elementTypeByteSize(ONNXTensorElementDataType elementType)
{
    switch (elementType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return sizeof(Ort::Float16_t);
    default:
        return 0;
    }
}

bool envFlag(const char *name, bool fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return fallback;
    }

    std::string value = trimString(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return fallback;
}

bool fileExists(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::string lowercaseFilename(const std::string &path)
{
    const size_t separator = path.find_last_of("/\\");
    const size_t nameBegin = separator == std::string::npos ? 0 : separator + 1;
    std::string filename = path.substr(nameBegin);
    std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return filename;
}

std::string fp32SiblingPath(const std::string &path)
{
    const size_t separator = path.find_last_of("/\\");
    const size_t nameBegin = separator == std::string::npos ? 0 : separator + 1;
    const std::string filename = lowercaseFilename(path);

    const size_t fp16Position = filename.rfind("fp16");
    if (fp16Position == std::string::npos) {
        return {};
    }

    std::string candidate = path;
    candidate.replace(nameBegin + fp16Position, 4, "fp32");
    return fileExists(candidate) ? candidate : std::string{};
}

bool isKnownDirectMLFp16ProblemModel(const std::string &path)
{
    const std::string filename = lowercaseFilename(path);
    return filename.find("fp16") != std::string::npos
        && (filename.find("yolov10") != std::string::npos
            || filename.find("yolo10") != std::string::npos
            || filename.find("yolo26") != std::string::npos);
}

const char *graphOptimizationName(GraphOptimizationLevel level)
{
    switch (level) {
    case ORT_DISABLE_ALL:
        return "disable_all";
    case ORT_ENABLE_BASIC:
        return "enable_basic";
    case ORT_ENABLE_EXTENDED:
        return "enable_extended";
    case ORT_ENABLE_LAYOUT:
        return "enable_layout";
    case ORT_ENABLE_ALL:
        return "enable_all";
    default:
        return "unknown";
    }
}

int graphOptimizationRank(GraphOptimizationLevel level)
{
    switch (level) {
    case ORT_DISABLE_ALL:
        return 0;
    case ORT_ENABLE_BASIC:
        return 1;
    case ORT_ENABLE_EXTENDED:
        return 2;
    case ORT_ENABLE_LAYOUT:
        return 3;
    case ORT_ENABLE_ALL:
        return 4;
    default:
        return 4;
    }
}

GraphOptimizationLevel envGraphOptimizationLevel(const char *name, GraphOptimizationLevel fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return fallback;
    }

    std::string value = trimString(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (value == "disable" || value == "disabled" || value == "disable_all" || value == "none" || value == "0") {
        return ORT_DISABLE_ALL;
    }
    if (value == "basic" || value == "enable_basic" || value == "1") {
        return ORT_ENABLE_BASIC;
    }
    if (value == "extended" || value == "enable_extended" || value == "2") {
        return ORT_ENABLE_EXTENDED;
    }
    if (value == "layout" || value == "enable_layout" || value == "3") {
        return ORT_ENABLE_LAYOUT;
    }
    if (value == "all" || value == "enable_all" || value == "99") {
        return ORT_ENABLE_ALL;
    }

    std::cerr << "[YoloOrtDml] Ignoring invalid " << name << " value: " << raw << std::endl;
    return fallback;
}

std::vector<GraphOptimizationLevel> dmlGraphOptimizationAttempts(
    GraphOptimizationLevel preferred,
    bool allowFallbacks,
    bool includePreferred)
{
    std::vector<GraphOptimizationLevel> attempts;
    if (includePreferred) {
        attempts.push_back(preferred);
    }
    if (!allowFallbacks) {
        return attempts;
    }

    const GraphOptimizationLevel fallbackOrder[] = {
        ORT_ENABLE_LAYOUT,
        ORT_ENABLE_EXTENDED,
        ORT_ENABLE_BASIC,
        ORT_DISABLE_ALL
    };
    const int preferredRank = graphOptimizationRank(preferred);
    for (const GraphOptimizationLevel candidate : fallbackOrder) {
        if (candidate != preferred && graphOptimizationRank(candidate) < preferredRank) {
            attempts.push_back(candidate);
        }
    }
    return attempts;
}
}

struct OrtInferEngine::Impl
{
    Impl()
        : env(ORT_LOGGING_LEVEL_ERROR, "YoloOrtDml"),
          memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
    }

    void setModel(const std::string &path)
    {
        modelPathValue = trimString(path);
        if (modelPathValue.empty()) {
            throw std::runtime_error("ONNX model path is empty.");
        }
        requestedModelPathValue = modelPathValue;

        resetState();
        readRuntimeOptions();

        if (!createDirectMLKnownFp16CompatibilitySession()
            && !createDirectMLSessionWithFallbacks(true, false)
            && !createDirectMLFp32SiblingSession()
            && !createDirectMLSessionWithFallbacks(false, useDmlGraphOptimizationFallbacks)) {
            createCpuSession();
        }

        readNodeNames();
        readInputInfo();
        prepareOutputTensors();
        runOptions = Ort::RunOptions();
        sessionReady = true;

        if (requestedModelPathValue != modelPathValue) {
            std::cout << "[YoloOrtDml] requested model: " << requestedModelPathValue << std::endl;
        }
        std::cout << "[YoloOrtDml] model: " << modelPathValue << std::endl;
        std::cout << "[YoloOrtDml] device: " << deviceNameValue << std::endl;
        std::cout << "[YoloOrtDml] graph optimization: " << graphOptimizationName(activeGraphOptimizationLevel) << std::endl;
        std::cout << "[YoloOrtDml] input: " << inputInfoValue.size.width << "x" << inputInfoValue.size.height
                  << (inputInfoValue.dynamicSize ? " dynamic" : "") << std::endl;
    }

    std::vector<Ort::Value> infer(InputTensorData &input)
    {
        ensureSession();
        Ort::Value &inputTensor = inputTensorValue(input);
        return session.Run(
            runOptions,
            inputNames.data(),
            &inputTensor,
            inputNames.size(),
            outputNames.data(),
            outputNames.size());
    }

    const std::vector<OutputTensorView> &inferViews(InputTensorData &input)
    {
        ensureSession();
        Ort::Value &inputTensor = inputTensorValue(input);
        if (!outputTensorsPreallocated) {
            resetOutputTensorSlots();
        }

        if (outputTensorsPreallocated && useIoBinding) {
            runWithIoBinding(input, inputTensor);
        } else {
            session.Run(
                runOptions,
                inputNames.data(),
                &inputTensor,
                inputNames.size(),
                outputNames.data(),
                outputTensors.data(),
                outputTensors.size());
        }
        if (!outputTensorsPreallocated) {
            refreshOutputViewsFromTensors();
        }
        return outputViews;
    }

    void ensureSession()
    {
        if (!sessionReady) {
            setModel(modelPathValue);
        }
    }

    void resetState()
    {
        inputNameAllocs.clear();
        inputNames.clear();
        outputNameAllocs.clear();
        outputNames.clear();
        inputTensorReady = false;
        cachedInputTensor = Ort::Value{nullptr};
        cachedData = nullptr;
        cachedElementCount = 0;
        cachedShape = {1, 3, 0, 0};
        cachedElementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        outputBuffers.clear();
        outputTensors.clear();
        outputViews.clear();
        outputTensorsPreallocated = false;
        ioBinding = Ort::IoBinding{nullptr};
        ioBindingInputReady = false;
        ioBindingOutputsReady = false;
        boundInputData = nullptr;
        boundInputElementCount = 0;
        boundInputShape = {1, 3, 0, 0};
        boundInputElementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        sessionReady = false;
        session = Ort::Session{nullptr};
        runOptions = Ort::RunOptions{nullptr};
    }

    void readRuntimeOptions()
    {
        preallocateOutputs = envFlag("YOLO_PREALLOCATE_OUTPUTS", true);
        useIoBinding = envFlag("YOLO_USE_IO_BINDING", true);
        useStrictDmlSession = envFlag("YOLO_DML_STRICT_SESSION", true);
        useHighPerformanceDml = envFlag("YOLO_DML_HIGH_PERFORMANCE", false);
        useDmlGraphOptimizationFallbacks = envFlag("YOLO_DML_GRAPH_OPT_FALLBACK", true);
        useFp32DmlModelFallback = envFlag("YOLO_DML_FP32_MODEL_FALLBACK", true);
        preferredDmlGraphOptimizationLevel = envGraphOptimizationLevel("YOLO_DML_GRAPH_OPT", ORT_ENABLE_ALL);
        cpuGraphOptimizationLevel = envGraphOptimizationLevel("YOLO_CPU_GRAPH_OPT", ORT_ENABLE_ALL);
    }

    void resetSessionOptions(bool directMl, GraphOptimizationLevel graphOptimizationLevel)
    {
        sessionOptions = Ort::SessionOptions();
        const unsigned int hardwareThreads = std::thread::hardware_concurrency();
        const int automaticThreads = directMl
            ? 1
            : static_cast<int>(std::min(6u, hardwareThreads == 0 ? 1u : hardwareThreads));
        sessionOptions.SetIntraOpNumThreads(automaticThreads);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(graphOptimizationLevel);
        activeGraphOptimizationLevel = graphOptimizationLevel;
    }

    void createSession()
    {
#ifdef _WIN32
        const std::wstring wideModelPath = widenPath(modelPathValue);
        session = Ort::Session(env, wideModelPath.c_str(), sessionOptions);
#else
        session = Ort::Session(env, modelPathValue.c_str(), sessionOptions);
#endif
    }

    bool createDirectMLSessionWithFallbacks(bool includePreferred, bool allowGraphOptimizationFallbacks)
    {
#ifdef _WIN32
        const std::vector<GraphOptimizationLevel> attempts = dmlGraphOptimizationAttempts(
            preferredDmlGraphOptimizationLevel,
            allowGraphOptimizationFallbacks,
            includePreferred);

        for (size_t attemptIndex = 0; attemptIndex < attempts.size(); ++attemptIndex) {
            const GraphOptimizationLevel graphOptimizationLevel = attempts[attemptIndex];
            session = Ort::Session{nullptr};
            deviceNameValue = "CPU";
            resetSessionOptions(true, graphOptimizationLevel);

            if (!appendDirectMLProviderIfAvailable()) {
                return false;
            }

            try {
                createSession();
                deviceNameValue = "DirectML";
                if (graphOptimizationLevel != preferredDmlGraphOptimizationLevel) {
                    std::cerr << "[YoloOrtDml] DirectML initialized with graph optimization "
                              << graphOptimizationName(graphOptimizationLevel) << "." << std::endl;
                }
                return true;
            } catch (const Ort::Exception &exception) {
                std::cerr << "[YoloOrtDml] DirectML session failed with graph optimization "
                          << graphOptimizationName(graphOptimizationLevel) << ": " << exception.what();
                if (attemptIndex + 1 < attempts.size()) {
                    std::cerr << "; retrying with "
                              << graphOptimizationName(attempts[attemptIndex + 1]) << ".";
                } else {
                    std::cerr << "; no more DirectML graph optimization levels in this attempt.";
                }
                std::cerr << std::endl;
            }
        }
#endif
        return false;
    }

    bool createDirectMLKnownFp16CompatibilitySession()
    {
        if (!useFp32DmlModelFallback || !isKnownDirectMLFp16ProblemModel(modelPathValue)) {
            return false;
        }

        const std::string originalModelPath = modelPathValue;
        const std::string fallbackModelPath = fp32SiblingPath(originalModelPath);
        if (fallbackModelPath.empty()) {
            return false;
        }

        std::cout << "[YoloOrtDml] DirectML compatibility: using FP32 sibling for this YOLO FP16 model: "
                  << fallbackModelPath << std::endl;

        modelPathValue = fallbackModelPath;
        if (createDirectMLSessionWithFallbacks(true, false)
            || createDirectMLSessionWithFallbacks(false, useDmlGraphOptimizationFallbacks)) {
            return true;
        }

        modelPathValue = originalModelPath;
        return false;
    }

    bool createDirectMLFp32SiblingSession()
    {
        if (!useFp32DmlModelFallback) {
            return false;
        }

        const std::string originalModelPath = modelPathValue;
        const std::string fallbackModelPath = fp32SiblingPath(originalModelPath);
        if (fallbackModelPath.empty()) {
            return false;
        }

        std::cout << "[YoloOrtDml] DirectML could not initialize the requested model; retrying DirectML with FP32 sibling: "
                  << fallbackModelPath << std::endl;

        modelPathValue = fallbackModelPath;
        if (createDirectMLSessionWithFallbacks(true, false)
            || createDirectMLSessionWithFallbacks(false, useDmlGraphOptimizationFallbacks)) {
            std::cout << "[YoloOrtDml] DirectML is using the FP32 sibling because the requested FP16 model failed to initialize."
                      << std::endl;
            return true;
        }

        modelPathValue = originalModelPath;
        return false;
    }

    void createCpuSession()
    {
        session = Ort::Session{nullptr};
        deviceNameValue = "CPU";
        resetSessionOptions(false, cpuGraphOptimizationLevel);
        createSession();
    }

    void readNodeNames()
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t inputCount = session.GetInputCount();
        const size_t outputCount = session.GetOutputCount();
        if (inputCount == 0 || outputCount == 0) {
            throw std::runtime_error("Invalid ONNX model: missing input or output nodes.");
        }

        inputNameAllocs.reserve(inputCount);
        inputNames.reserve(inputCount);
        for (size_t i = 0; i < inputCount; ++i) {
            inputNameAllocs.push_back(session.GetInputNameAllocated(i, allocator));
            inputNames.push_back(inputNameAllocs.back().get());
        }

        outputNameAllocs.reserve(outputCount);
        outputNames.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            outputNameAllocs.push_back(session.GetOutputNameAllocated(i, allocator));
            outputNames.push_back(outputNameAllocs.back().get());
        }
    }

    void readInputInfo()
    {
        const Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        const auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = tensorInfo.GetShape();
        inputInfoValue.elementType = tensorInfo.GetElementType();

        if (shape.size() < 4) {
            throw std::runtime_error("Invalid ONNX input shape: expected [N, C, H, W].");
        }

        inputInfoValue.channels = shape[1] > 0 ? static_cast<int>(shape[1]) : 3;
        if (inputInfoValue.channels != 1 && inputInfoValue.channels != 3) {
            throw std::runtime_error("Only 1-channel and 3-channel model inputs are supported.");
        }

        inputInfoValue.dynamicSize = shape[2] <= 0 || shape[3] <= 0;
        inputInfoValue.size = cv::Size(
            shape[3] > 0 ? static_cast<int>(shape[3]) : 640,
            shape[2] > 0 ? static_cast<int>(shape[2]) : 640);

        if (inputInfoValue.elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && inputInfoValue.elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            throw std::runtime_error("Only float32 and float16 model inputs are supported.");
        }
    }

    Ort::Value &inputTensorValue(InputTensorData &input)
    {
        const void *data = input.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ? static_cast<const void *>(input.floatData.data())
            : static_cast<const void *>(input.halfData.data());
        const bool cacheValid = inputTensorReady
            && cachedElementType == input.elementType
            && cachedElementCount == input.elementCount
            && cachedShape == input.shape
            && cachedData == data;
        if (cacheValid) {
            return cachedInputTensor;
        }

        cachedInputTensor = createInputTensor(input);
        cachedElementType = input.elementType;
        cachedElementCount = input.elementCount;
        cachedShape = input.shape;
        cachedData = data;
        inputTensorReady = true;
        return cachedInputTensor;
    }

    Ort::Value createInputTensor(InputTensorData &input)
    {
        if (input.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return Ort::Value::CreateTensor<float>(
                memoryInfo,
                input.floatData.data(),
                input.elementCount,
                input.shape.data(),
                input.shape.size());
        }

        if (input.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return Ort::Value::CreateTensor<Ort::Float16_t>(
                memoryInfo,
                input.halfData.data(),
                input.elementCount,
                input.shape.data(),
                input.shape.size());
        }

        throw std::runtime_error("Only float32 and float16 model inputs are supported.");
    }

    void prepareOutputTensors()
    {
        outputBuffers.clear();
        outputTensors.clear();
        outputViews.clear();
        outputTensorsPreallocated = false;

        if (!preallocateOutputs) {
            resetOutputTensorSlots();
            return;
        }

        const size_t outputCount = session.GetOutputCount();
        outputBuffers.reserve(outputCount);

        for (size_t i = 0; i < outputCount; ++i) {
            const Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
            const auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

            OutputBuffer buffer;
            buffer.shape = tensorInfo.GetShape();
            buffer.elementType = tensorInfo.GetElementType();

            size_t elementCount = 1;
            for (const int64_t dim : buffer.shape) {
                if (dim <= 0) {
                    resetOutputTensorSlots();
                    return;
                }

                const size_t dimension = static_cast<size_t>(dim);
                if (elementCount > std::numeric_limits<size_t>::max() / dimension) {
                    throw std::runtime_error("ONNX output tensor is too large.");
                }
                elementCount *= dimension;
            }

            const size_t elementBytes = elementTypeByteSize(buffer.elementType);
            if (elementBytes == 0) {
                resetOutputTensorSlots();
                return;
            }
            if (elementCount > std::numeric_limits<size_t>::max() / elementBytes) {
                throw std::runtime_error("ONNX output tensor byte size is too large.");
            }

            buffer.storage.resize(elementCount * elementBytes);
            outputBuffers.push_back(std::move(buffer));
        }

        outputTensors.reserve(outputBuffers.size());
        outputViews.reserve(outputBuffers.size());
        for (OutputBuffer &buffer : outputBuffers) {
            outputTensors.push_back(Ort::Value::CreateTensor(
                memoryInfo,
                buffer.storage.data(),
                buffer.storage.size(),
                buffer.shape.data(),
                buffer.shape.size(),
                buffer.elementType));
            outputViews.push_back(OutputTensorView{
                buffer.shape,
                buffer.elementType,
                buffer.storage.data()});
        }
        outputTensorsPreallocated = true;
    }

    void resetOutputTensorSlots()
    {
        outputTensors.clear();
        outputViews.clear();
        ioBindingOutputsReady = false;
        outputTensors.reserve(outputNames.size());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            outputTensors.emplace_back(nullptr);
        }
    }

    void refreshOutputViewsFromTensors()
    {
        outputViews.clear();
        outputViews.reserve(outputTensors.size());
        for (const Ort::Value &tensor : outputTensors) {
            const auto info = tensor.GetTensorTypeAndShapeInfo();
            OutputTensorView view;
            view.shape = info.GetShape();
            view.elementType = info.GetElementType();
            if (view.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                view.data = tensor.GetTensorData<float>();
            } else if (view.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                view.data = tensor.GetTensorData<Ort::Float16_t>();
            } else {
                throw std::runtime_error("Only float32 and float16 model outputs are supported.");
            }
            outputViews.push_back(std::move(view));
        }
    }

    void runWithIoBinding(InputTensorData &input, Ort::Value &inputTensor)
    {
        if (!ioBinding) {
            ioBinding = Ort::IoBinding(session);
            ioBindingInputReady = false;
            ioBindingOutputsReady = false;
        }

        const void *inputData = input.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ? static_cast<const void *>(input.floatData.data())
            : static_cast<const void *>(input.halfData.data());
        const bool inputStillBound = ioBindingInputReady
            && boundInputData == inputData
            && boundInputElementCount == input.elementCount
            && boundInputShape == input.shape
            && boundInputElementType == input.elementType;
        if (!inputStillBound) {
            ioBinding.ClearBoundInputs();
            ioBinding.BindInput(inputNames[0], inputTensor);
            boundInputData = inputData;
            boundInputElementCount = input.elementCount;
            boundInputShape = input.shape;
            boundInputElementType = input.elementType;
            ioBindingInputReady = true;
        }

        if (!ioBindingOutputsReady) {
            ioBinding.ClearBoundOutputs();
            for (size_t i = 0; i < outputTensors.size(); ++i) {
                ioBinding.BindOutput(outputNames[i], outputTensors[i]);
            }
            ioBindingOutputsReady = true;
        }

        session.Run(runOptions, ioBinding);
    }

    bool appendDirectMLProviderIfAvailable()
    {
#ifdef _WIN32
        const std::vector<std::string> providers = Ort::GetAvailableProviders();
        const bool hasDml = std::find(providers.begin(), providers.end(), "DmlExecutionProvider") != providers.end()
            || std::find(providers.begin(), providers.end(), "DMLExecutionProvider") != providers.end();
        if (!hasDml) {
            std::cerr << "[YoloOrtDml] DirectML provider is unavailable; using CPU." << std::endl;
            return false;
        }

        try {
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetInterOpNumThreads(1);
            if (useStrictDmlSession) {
                sessionOptions.DisableMemPattern();
                sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            }
            if (useHighPerformanceDml && appendDirectMLHighPerformanceProvider()) {
                deviceNameValue = "DirectML";
            } else {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
                deviceNameValue = "DirectML";
            }
            return true;
        } catch (const Ort::Exception &exception) {
            deviceNameValue = "CPU";
            std::cerr << "[YoloOrtDml] DirectML setup failed: " << exception.what() << "; using CPU." << std::endl;
            return false;
        }
#else
        return false;
#endif
    }

    bool appendDirectMLHighPerformanceProvider()
    {
#ifdef _WIN32
        const void *providerApi = nullptr;
        OrtStatus *status = Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, &providerApi);
        if (status != nullptr || providerApi == nullptr) {
            if (status != nullptr) {
                Ort::GetApi().ReleaseStatus(status);
            }
            return false;
        }

        OrtDmlDeviceOptions options;
        options.Preference = HighPerformance;
        options.Filter = Gpu;
        Ort::ThrowOnError(static_cast<const OrtDmlApi *>(providerApi)->SessionOptionsAppendExecutionProvider_DML2(sessionOptions, &options));
        return true;
#else
        return false;
#endif
    }

    std::string requestedModelPathValue;
    std::string modelPathValue;
    std::string deviceNameValue{"CPU"};
    ModelInputInfo inputInfoValue;
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    Ort::RunOptions runOptions{nullptr};
    Ort::MemoryInfo memoryInfo{nullptr};
    bool sessionReady{false};
    bool inputTensorReady{false};
    ONNXTensorElementDataType cachedElementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::array<int64_t, 4> cachedShape{1, 3, 0, 0};
    size_t cachedElementCount{0};
    const void *cachedData{nullptr};
    Ort::Value cachedInputTensor{nullptr};
    std::vector<Ort::AllocatedStringPtr> inputNameAllocs;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs;
    std::vector<const char *> outputNames;
    struct OutputBuffer
    {
        std::vector<int64_t> shape;
        ONNXTensorElementDataType elementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
        std::vector<std::uint8_t> storage;
    };
    std::vector<OutputBuffer> outputBuffers;
    std::vector<Ort::Value> outputTensors;
    std::vector<OutputTensorView> outputViews;
    bool preallocateOutputs{true};
    bool useIoBinding{true};
    bool useStrictDmlSession{true};
    bool useHighPerformanceDml{true};
    bool useDmlGraphOptimizationFallbacks{true};
    bool useFp32DmlModelFallback{true};
    GraphOptimizationLevel preferredDmlGraphOptimizationLevel{ORT_ENABLE_ALL};
    GraphOptimizationLevel cpuGraphOptimizationLevel{ORT_ENABLE_ALL};
    GraphOptimizationLevel activeGraphOptimizationLevel{ORT_ENABLE_ALL};
    bool outputTensorsPreallocated{false};
    Ort::IoBinding ioBinding{nullptr};
    bool ioBindingInputReady{false};
    bool ioBindingOutputsReady{false};
    const void *boundInputData{nullptr};
    size_t boundInputElementCount{0};
    std::array<int64_t, 4> boundInputShape{1, 3, 0, 0};
    ONNXTensorElementDataType boundInputElementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
};

OrtInferEngine::OrtInferEngine()
    : impl_(std::make_unique<Impl>())
{
}

OrtInferEngine::~OrtInferEngine() = default;

void OrtInferEngine::setModel(const std::string &modelPath)
{
    impl_->setModel(modelPath);
}

const ModelInputInfo &OrtInferEngine::inputInfo() const
{
    return impl_->inputInfoValue;
}

const std::string &OrtInferEngine::modelPath() const
{
    return impl_->modelPathValue;
}

const std::string &OrtInferEngine::deviceName() const
{
    return impl_->deviceNameValue;
}

std::vector<Ort::Value> OrtInferEngine::infer(InputTensorData &input)
{
    return impl_->infer(input);
}

const std::vector<OutputTensorView> &OrtInferEngine::inferViews(InputTensorData &input)
{
    return impl_->inferViews(input);
}
