/**
 * Read-only validation for the official S10 low-level ONNX policy.
 *
 * This executable deliberately has no JointsDataCmd include and creates no
 * /JOINTS_CMD publisher. It consumes the real DDS state, applies the same S10
 * calibration and policy runner as rl_deploy, and reports inference health.
 */

#include "s10_policy_runner.hpp"

#include "drdds/msg/imu_data.hpp"
#include "drdds/msg/joints_data.hpp"
#include "drdds/msg/steer.hpp"
#include "rclcpp/rclcpp.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    double duration_seconds = 30.0;
    std::string policy_path;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--duration-seconds" && index + 1 < argc) {
            options.duration_seconds = std::stod(argv[++index]);
        } else if (argument == "--policy" && index + 1 < argc) {
            options.policy_path = argv[++index];
        } else if (argument == "--help") {
            std::cout << "Usage: s10_low_level_dry_run "
                         "[--duration-seconds N] [--policy PATH]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (!(options.duration_seconds > 0.0)) {
        throw std::invalid_argument("--duration-seconds must be positive");
    }
    if (options.policy_path.empty()) {
        const auto source = std::filesystem::path(__FILE__).parent_path();
        options.policy_path = std::filesystem::canonical(
            source / ".." / "policy" / "policy.onnx").string();
    }
    return options;
}

bool Finite(const VecXf& values) {
    return values.size() > 0 && values.allFinite();
}

class S10LowLevelDryRun : public rclcpp::Node {
public:
    explicit S10LowLevelDryRun(const Options& options)
        : Node("s10_low_level_dry_run"),
          options_(options),
          policy_("s10_official_dry_run", options.policy_path),
          started_(Clock::now()),
          last_report_(started_) {
        position_offsets_deg_ = {
            -35.0f, -145.0f, 156.0f, 0.0f,
             35.0f, -145.0f, 156.0f, 0.0f,
            -35.0f,  145.0f,-156.0f, 0.0f,
             35.0f,  145.0f,-156.0f, 0.0f,
        };
        joint_directions_ = {
             1.0f,  1.0f, -1.0f,  1.0f,
             1.0f, -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f, -1.0f,
        };

        policy_.OnEnter();
        joints_subscription_ = create_subscription<drdds::msg::JointsData>(
            "/JOINTS_DATA", rclcpp::SensorDataQoS(),
            std::bind(&S10LowLevelDryRun::OnJoints, this, std::placeholders::_1));
        imu_subscription_ = create_subscription<drdds::msg::ImuData>(
            "/IMU_DATA", rclcpp::SensorDataQoS(),
            std::bind(&S10LowLevelDryRun::OnImu, this, std::placeholders::_1));
        steer_subscription_ = create_subscription<drdds::msg::Steer>(
            "/STEER", 10,
            std::bind(&S10LowLevelDryRun::OnSteer, this, std::placeholders::_1));
        timer_ = create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&S10LowLevelDryRun::Tick, this));

        std::cout << "S10_LOW_LEVEL_DRY_RUN_READ_ONLY policy="
                  << options_.policy_path
                  << " duration_s=" << options_.duration_seconds
                  << " publishers=none" << std::endl;
    }

    bool passed() const {
        return !failed_ && inference_count_ > 0;
    }

private:
    void CalibrateOffsets(const drdds::msg::JointsData& message) {
        for (int index : {1, 5, 9, 13}) {
            const float position =
                message.data.joints_data[index].position * joint_directions_[index]
                + Deg2Rad(position_offsets_deg_[index]);
            if (position < Deg2Rad(-140.0f)) {
                position_offsets_deg_[index] += 360.0f;
            } else if (position > Deg2Rad(140.0f)) {
                position_offsets_deg_[index] -= 360.0f;
            }
        }
        for (int index : {2, 6, 10, 14}) {
            const float position =
                message.data.joints_data[index].position * joint_directions_[index]
                + Deg2Rad(position_offsets_deg_[index]);
            if (position < Deg2Rad(-164.0f)) {
                position_offsets_deg_[index] += 360.0f;
            } else if (position > Deg2Rad(164.0f)) {
                position_offsets_deg_[index] -= 360.0f;
            }
        }
        offsets_calibrated_ = true;
    }

    void OnJoints(const drdds::msg::JointsData::SharedPtr message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!offsets_calibrated_) {
            CalibrateOffsets(*message);
        }
        for (int index = 0; index < 16; ++index) {
            const auto& joint = message->data.joints_data[index];
            state_.joint_pos(index) =
                joint.position * joint_directions_[index]
                + Deg2Rad(position_offsets_deg_[index]);
            state_.joint_vel(index) = joint.velocity * joint_directions_[index];
            state_.joint_tau(index) = joint.torque * joint_directions_[index];
        }
        ++joint_count_;
        last_joint_receipt_ = Clock::now();
        have_joints_ = true;
    }

    void OnImu(const drdds::msg::ImuData::SharedPtr message) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.base_rpy = Vec3f(
            Deg2Rad(message->data.roll),
            Deg2Rad(message->data.pitch),
            Deg2Rad(message->data.yaw));
        state_.base_rot_mat = RpyToRm(state_.base_rpy);
        state_.base_omega = Vec3f(
            message->data.omega_x,
            message->data.omega_y,
            message->data.omega_z);
        state_.base_acc = Vec3f(
            message->data.acc_x,
            message->data.acc_y,
            message->data.acc_z);
        ++imu_count_;
        last_imu_receipt_ = Clock::now();
        have_imu_ = true;
    }

    void OnSteer(const drdds::msg::Steer::SharedPtr message) {
        std::lock_guard<std::mutex> lock(mutex_);
        command_.forward_vel_scale = message->data.x;
        command_.side_vel_scale = message->data.y;
        command_.turnning_vel_scale = message->data.yaw;
        ++steer_count_;
        last_steer_receipt_ = Clock::now();
        have_steer_ = true;
    }

    void Tick() {
        const auto now = Clock::now();
        RobotBasicState state;
        UserCommand command{};
        double joint_age = 0.0;
        double imu_age = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!have_joints_ || !have_imu_) {
                FinishIfDue(now);
                return;
            }
            joint_age = std::chrono::duration<double>(now - last_joint_receipt_).count();
            imu_age = std::chrono::duration<double>(now - last_imu_receipt_).count();
            state = state_;
            command = command_;
            if (!have_steer_ ||
                std::chrono::duration<double>(now - last_steer_receipt_).count() > 0.5) {
                command.forward_vel_scale = 0.0f;
                command.side_vel_scale = 0.0f;
                command.turnning_vel_scale = 0.0f;
            }
        }

        if (joint_age > 0.2 || imu_age > 0.2) {
            failed_ = true;
            failure_reason_ = "sensor_stale_over_200ms";
            Finish();
            return;
        }

        const auto inference_start = Clock::now();
        RobotAction robot_action = policy_.getRobotAction(state, command);
        const double inference_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - inference_start).count();
        const VecXf observation = policy_.GetLastObservation();
        const VecXf action = policy_.GetLastPolicyAction();
        const MatXf target = robot_action.ConvertToMat();
        if (!Finite(observation) || !Finite(action) || !target.allFinite()) {
            failed_ = true;
            failure_reason_ = "nonfinite_policy_io";
            Finish();
            return;
        }
        ++inference_count_;
        inference_sum_ms_ += inference_ms;
        inference_max_ms_ = std::max(inference_max_ms_, inference_ms);
        last_observation_ = observation;
        last_action_ = action;
        last_target_ = target;
        last_command_ = command;

        if (std::chrono::duration<double>(now - last_report_).count() >= 1.0) {
            Report(now, joint_age, imu_age);
            last_report_ = now;
        }
        FinishIfDue(now);
    }

    void Report(const Clock::time_point& now, double joint_age, double imu_age) const {
        const double elapsed = std::chrono::duration<double>(now - started_).count();
        std::cout << std::fixed << std::setprecision(6)
                  << "S10_LOW_LEVEL_DRY_RUN {"
                  << "\"elapsed_s\":" << elapsed
                  << ",\"joint_hz\":" << joint_count_ / elapsed
                  << ",\"imu_hz\":" << imu_count_ / elapsed
                  << ",\"steer_hz\":" << steer_count_ / elapsed
                  << ",\"policy_hz\":" << inference_count_ / elapsed
                  << ",\"joint_age_s\":" << joint_age
                  << ",\"imu_age_s\":" << imu_age
                  << ",\"inference_mean_ms\":"
                  << inference_sum_ms_ / std::max<uint64_t>(1, inference_count_)
                  << ",\"inference_max_ms\":" << inference_max_ms_
                  << ",\"obs_min\":" << last_observation_.minCoeff()
                  << ",\"obs_max\":" << last_observation_.maxCoeff()
                  << ",\"action_min\":" << last_action_.minCoeff()
                  << ",\"action_max\":" << last_action_.maxCoeff()
                  << ",\"target_pos_min\":" << last_target_.col(1).minCoeff()
                  << ",\"target_pos_max\":" << last_target_.col(1).maxCoeff()
                  << ",\"target_vel_min\":" << last_target_.col(3).minCoeff()
                  << ",\"target_vel_max\":" << last_target_.col(3).maxCoeff()
                  << ",\"command\":[" << last_command_.forward_vel_scale << ','
                  << last_command_.side_vel_scale << ','
                  << last_command_.turnning_vel_scale << "]}"
                  << std::endl;
    }

    void FinishIfDue(const Clock::time_point& now) {
        if (std::chrono::duration<double>(now - started_).count()
            >= options_.duration_seconds) {
            Finish();
        }
    }

    void Finish() {
        if (finished_.exchange(true)) {
            return;
        }
        std::cout << "S10_LOW_LEVEL_DRY_RUN_RESULT {\"passed\":"
                  << (passed() ? "true" : "false")
                  << ",\"inferences\":" << inference_count_
                  << ",\"failure\":\"" << failure_reason_ << "\"}"
                  << std::endl;
        rclcpp::shutdown();
    }

    Options options_;
    S10PolicyRunner policy_;
    RobotBasicState state_;
    UserCommand command_{};
    UserCommand last_command_{};
    std::array<float, 16> position_offsets_deg_{};
    std::array<float, 16> joint_directions_{};
    bool offsets_calibrated_ = false;
    bool have_joints_ = false;
    bool have_imu_ = false;
    bool have_steer_ = false;
    bool failed_ = false;
    std::string failure_reason_;
    std::atomic<bool> finished_{false};
    mutable std::mutex mutex_;
    Clock::time_point started_;
    Clock::time_point last_report_;
    Clock::time_point last_joint_receipt_{};
    Clock::time_point last_imu_receipt_{};
    Clock::time_point last_steer_receipt_{};
    uint64_t joint_count_ = 0;
    uint64_t imu_count_ = 0;
    uint64_t steer_count_ = 0;
    uint64_t inference_count_ = 0;
    double inference_sum_ms_ = 0.0;
    double inference_max_ms_ = 0.0;
    VecXf last_observation_ = VecXf::Zero(57);
    VecXf last_action_ = VecXf::Zero(16);
    MatXf last_target_ = MatXf::Zero(16, 5);

    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joints_subscription_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_subscription_;
    rclcpp::Subscription<drdds::msg::Steer>::SharedPtr steer_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        rclcpp::init(0, nullptr);
        auto node = std::make_shared<S10LowLevelDryRun>(options);
        rclcpp::spin(node);
        const bool passed = node->passed();
        node.reset();
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "S10_LOW_LEVEL_DRY_RUN_FATAL " << error.what() << std::endl;
        return 2;
    }
}
