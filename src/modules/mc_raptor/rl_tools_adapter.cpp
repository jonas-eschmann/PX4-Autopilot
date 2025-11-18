#include <rl_tools/operations/arm.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/inference/executor/executor.h>

#include "blob/policy.h"

#include <rl_tools/persist/backends/tar/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>



namespace rlt = rl_tools;

namespace other{
    using DEV_SPEC = rlt::devices::DefaultARMSpecification;
    using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
}

struct RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG{
    using DEVICE = other::DEVICE;
    using TI = typename other::DEVICE::index_t;
    using RNG = other::DEVICE::SPEC::RANDOM::ENGINE<>;
    static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
    static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
    using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
    using POLICY_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
    using POLICY_BATCH_SIZE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>;
    using POLICY = POLICY_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<false, false>>;
    inline static POLICY policy_copy;
    using TYPE_POLICY = typename POLICY::TYPE_POLICY;
    static auto& policy() {
        // return rlt::checkpoint::actor::module;
        return policy_copy;
    }
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 2.5 * 1000 * 1000; // Inference is at 500hz
    static constexpr TI CONTROL_INTERVAL_NATIVE_NS = 10 * 1000 * 1000; // Training is 100hz
    static constexpr TI TIMING_STATS_NUM_STEPS = 100;
    static constexpr bool FORCE_SYNC_INTERMEDIATE = true;
    static constexpr bool FORCE_SYNC_NATIVE_RUNTIME = true; //
    static constexpr TI FORCE_SYNC_NATIVE = 4;
    static constexpr bool DYNAMIC_ALLOCATION = false;
#if defined(__PX4_POSIX)
// Relax warning levels for Gazebo sitl. Because Gazebo SITL runs at 250Hz IMU rate, it is not a clean multiple of the training frequency (100hz), hence if the thresholds are set too strict, warnings will be triggered all the time. Generally, Raptor is quite robuts agains control frequency deviations.
    struct WARNING_LEVELS: rlt::inference::executor::WarningLevelsDefault<TYPE_POLICY>{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr T INTERMEDIATE_TIMING_JITTER_HIGH_THRESHOLD_NS = 2.0;
        static constexpr T INTERMEDIATE_TIMING_JITTER_LOW_THRESHOLD_NS = 0.5;
        static constexpr T INTERMEDIATE_TIMING_BIAS_HIGH_THRESHOLD = 2.0;
        static constexpr T INTERMEDIATE_TIMING_BIAS_LOW_THRESHOLD = 0.5;
        static constexpr T NATIVE_TIMING_JITTER_HIGH_THRESHOLD_NS = 2.0;
        static constexpr T NATIVE_TIMING_JITTER_LOW_THRESHOLD_NS = 0.5;
        static constexpr T NATIVE_TIMING_BIAS_HIGH_THRESHOLD = 2.0;
        static constexpr T NATIVE_TIMING_BIAS_LOW_THRESHOLD = 0.5;
    };
#else
    using WARNING_LEVELS = rlt::inference::executor::WarningLevelsDefault<TYPE_POLICY>;
#endif
};

bool rl_tools_inference_applications_l2f_init_policy(char* data, size_t size){
    RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::DEVICE device;
    if(size > 0){
        rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::TI>> reader_group;
        reader_group.data = data;
        reader_group.size = size;
        auto actor_group = rlt::get_group(device, reader_group, "actor");
        return rlt::load(device, RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::policy(), actor_group);
    }
    else{
		rlt::copy(device, device, rl_tools::checkpoint::actor::module, RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::policy_copy);
        return true;
    }
}

// #define RL_TOOLS_DISABLE_TEST
#include <rl_tools/inference/applications/l2f/c_backend.h>
