#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <drdds/msg/steer.hpp>
#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

namespace {

constexpr size_t kRasterCells = 96 * 90;
constexpr size_t kExternalProprio = 13;
constexpr size_t kPackedInput = kExternalProprio + 4 * kRasterCells;
constexpr size_t kActorProprio = 15;
constexpr size_t kLatentCells = 64 * 5 * 8;
constexpr size_t kActorObservation = kActorProprio + kLatentCells;
constexpr size_t kHiddenCells = 512;

class SruHighLevelDryRun final : public rclcpp::Node {
 public:
  SruHighLevelDryRun()
      : Node("sru_high_level_onnx_dry_run"),
        env_(ORT_LOGGING_LEVEL_WARNING, "sru_high_level_dry_run") {
    const auto encoder_path = declare_parameter<std::string>("encoder_model", "");
    const auto policy_path = declare_parameter<std::string>("policy_model", "");
    enable_command_output_ =
        declare_parameter<bool>("enable_command_output", false);
    if (encoder_path.empty() || policy_path.empty()) {
      throw std::runtime_error("encoder_model and policy_model parameters are required");
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    encoder_ = std::make_unique<Ort::Session>(env_, encoder_path.c_str(), options);
    policy_ = std::make_unique<Ort::Session>(env_, policy_path.c_str(), options);
    validate_models();

    candidate_pub_ = create_publisher<geometry_msgs::msg::Twist>(
        "/s10/navigation/candidate_cmd", 10);
    if (enable_command_output_) {
      command_pub_ = create_publisher<drdds::msg::Steer>(
          "/s10/navigation/steer", 10);
    }
    action_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/s10/navigation/onnx_raw_action", 10);
    diagnostics_pub_ = create_publisher<std_msgs::msg::String>(
        "/s10/navigation/onnx_diagnostics", 10);
    input_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/s10/navigation/onnx_input", 2,
        std::bind(&SruHighLevelDryRun::on_input, this, std::placeholders::_1));
    if (enable_command_output_) {
      RCLCPP_ERROR(
          get_logger(),
          "ONNX COMMAND OUTPUT ENABLED: publishing /s10/navigation/steer; "
          "keep the robot suspended for the first control test");
    } else {
      RCLCPP_WARN(
          get_logger(),
          "ONNX dry-run ready: output is isolated at /s10/navigation/candidate_cmd; "
          "this node never publishes /cmd_vel, /STEER, /s10/navigation/steer, or /JOINTS_CMD");
    }
  }

 private:
  static std::vector<int64_t> shape(const Ort::Session& session, size_t index, bool input) {
    const auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    return info.GetTensorTypeAndShapeInfo().GetShape();
  }

  void validate_models() {
    if (encoder_->GetInputCount() != 4 || encoder_->GetOutputCount() != 1) {
      throw std::runtime_error("encoder ONNX must expose four inputs and one output");
    }
    if (policy_->GetInputCount() != 3 || policy_->GetOutputCount() != 3) {
      throw std::runtime_error("policy ONNX must expose obs/h/c inputs and action/h/c outputs");
    }
    const auto latent_shape = shape(*encoder_, 0, false);
    const auto action_shape = shape(*policy_, 0, false);
    if (latent_shape.size() != 4 || latent_shape[1] != 64 ||
        latent_shape[2] != 5 || latent_shape[3] != 8) {
      throw std::runtime_error("encoder output is not [B,64,5,8]");
    }
    if (action_shape.size() != 2 || action_shape[1] != 2) {
      throw std::runtime_error("policy output is not [B,2]");
    }
  }

  void on_input(const std_msgs::msg::Float32MultiArray::SharedPtr message) {
    if (message->data.size() != kPackedInput) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Rejected ONNX input with %zu values; expected %zu",
          message->data.size(), kPackedInput);
      return;
    }
    if (!std::all_of(message->data.begin(), message->data.end(),
                     [](float value) { return std::isfinite(value); })) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Rejected non-finite ONNX input");
      return;
    }

    const auto started = std::chrono::steady_clock::now();
    try {
      run_once(message->data);
    } catch (const std::exception& error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "ONNX inference failed: %s", error.what());
      return;
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++frames_;
    max_latency_ms_ = std::max(max_latency_ms_, elapsed);
    latency_sum_ms_ += elapsed;
    if (frames_ == 1 || frames_ % 5 == 0) {
      std_msgs::msg::String diagnostic;
      std::ostringstream stream;
      stream << "{\"mode\":\""
             << (enable_command_output_ ? "command_output" : "dry_run")
             << "\",\"frames\":" << frames_
             << ",\"last_latency_ms\":" << elapsed
             << ",\"mean_latency_ms\":" << latency_sum_ms_ / frames_
             << ",\"max_latency_ms\":" << max_latency_ms_
             << ",\"raw_action\":[" << last_action_[0] << "," << last_action_[1]
             << "],\"candidate_cmd\":[" << filtered_command_[0] << ",0,"
             << filtered_command_[2] << "]}";
      diagnostic.data = stream.str();
      diagnostics_pub_->publish(diagnostic);
      RCLCPP_INFO(get_logger(), "%s", diagnostic.data.c_str());
    }
  }

  void run_once(const std::vector<float>& packed) {
    const std::array<int64_t, 3> raster_shape{1, 96, 90};
    const std::array<int64_t, 2> obs_shape{1, static_cast<int64_t>(kActorObservation)};
    const std::array<int64_t, 3> state_shape{1, 1, static_cast<int64_t>(kHiddenCells)};
    const char* encoder_inputs[] = {
        "front_distance", "rear_distance", "front_world_z", "rear_world_z"};
    const char* encoder_outputs[] = {"fused_latent"};
    std::vector<Ort::Value> encoder_values;
    encoder_values.reserve(4);
    for (size_t index = 0; index < 4; ++index) {
      auto* data = const_cast<float*>(packed.data() + kExternalProprio + index * kRasterCells);
      encoder_values.push_back(Ort::Value::CreateTensor<float>(
          memory_info_, data, kRasterCells, raster_shape.data(), raster_shape.size()));
    }
    auto latent = encoder_->Run(
        Ort::RunOptions{nullptr}, encoder_inputs, encoder_values.data(),
        encoder_values.size(), encoder_outputs, 1);
    const float* latent_data = latent[0].GetTensorData<float>();

    std::copy_n(packed.data(), 9, actor_observation_.data());
    actor_observation_[9] = last_action_[0];
    actor_observation_[10] = last_action_[1];
    std::copy_n(packed.data() + 9, 4, actor_observation_.data() + 11);
    std::copy_n(latent_data, kLatentCells, actor_observation_.data() + kActorProprio);

    auto obs_value = Ort::Value::CreateTensor<float>(
        memory_info_, actor_observation_.data(), actor_observation_.size(),
        obs_shape.data(), obs_shape.size());
    auto hidden_value = Ort::Value::CreateTensor<float>(
        memory_info_, hidden_.data(), hidden_.size(), state_shape.data(), state_shape.size());
    auto cell_value = Ort::Value::CreateTensor<float>(
        memory_info_, cell_.data(), cell_.size(), state_shape.data(), state_shape.size());
    std::array<Ort::Value, 3> policy_values{
        std::move(obs_value), std::move(hidden_value), std::move(cell_value)};
    const char* policy_inputs[] = {"obs", "h_in", "c_in"};
    const char* policy_outputs[] = {"actions", "h_out", "c_out"};
    auto outputs = policy_->Run(
        Ort::RunOptions{nullptr}, policy_inputs, policy_values.data(),
        policy_values.size(), policy_outputs, 3);
    const float* action = outputs[0].GetTensorData<float>();
    std::copy_n(outputs[1].GetTensorData<float>(), kHiddenCells, hidden_.data());
    std::copy_n(outputs[2].GetTensorData<float>(), kHiddenCells, cell_.data());
    last_action_[0] = action[0];
    last_action_[1] = action[1];

    const float target_vx = std::clamp(1.5F * std::tanh(action[0]), -1.0F, 1.0F);
    const float target_yaw = std::clamp(std::tanh(action[1]), -1.0F, 1.0F);
    filtered_command_[0] = 0.5F * filtered_command_[0] + 0.5F * target_vx;
    filtered_command_[1] = 0.0F;
    filtered_command_[2] = 0.5F * filtered_command_[2] + 0.5F * target_yaw;

    geometry_msgs::msg::Twist candidate;
    candidate.linear.x = filtered_command_[0];
    candidate.angular.z = filtered_command_[2];
    candidate_pub_->publish(candidate);
    if (command_pub_) {
      drdds::msg::Steer command;
      command.data.x = filtered_command_[0];
      command.data.y = filtered_command_[1];
      command.data.yaw = filtered_command_[2];
      command_pub_->publish(command);
    }
    std_msgs::msg::Float32MultiArray raw_action;
    raw_action.data = {last_action_[0], last_action_[1]};
    action_pub_->publish(raw_action);
  }

  Ort::Env env_;
  Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> encoder_;
  std::unique_ptr<Ort::Session> policy_;
  std::array<float, kActorObservation> actor_observation_{};
  std::array<float, kHiddenCells> hidden_{};
  std::array<float, kHiddenCells> cell_{};
  std::array<float, 2> last_action_{};
  std::array<float, 3> filtered_command_{};
  size_t frames_ = 0;
  double latency_sum_ms_ = 0.0;
  double max_latency_ms_ = 0.0;
  bool enable_command_output_ = false;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr input_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr candidate_pub_;
  rclcpp::Publisher<drdds::msg::Steer>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr action_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SruHighLevelDryRun>());
  } catch (const std::exception& error) {
    std::cerr << "SRU_HIGH_LEVEL_DRY_RUN_FATAL: " << error.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
