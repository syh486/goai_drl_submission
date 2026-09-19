/**
 * ROS2 command input used by the official Jetson/ARM deployment path.
 */
#pragma once

#include "user_command_interface.h"
#include "drdds/msg/steer.hpp"
#include "std_msgs/msg/string.hpp"
#include "rclcpp/rclcpp.hpp"

#include <cstring>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace interface;
using namespace types;

class DdsCommandInterface : public UserCommandInterface {
private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<drdds::msg::Steer>::SharedPtr steer_sub_;
    rclcpp::Subscription<drdds::msg::Steer>::SharedPtr navigation_steer_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr key_sub_;
    std::mutex command_mutex_;
    float manual_forward_ = 0.0f;
    float manual_side_ = 0.0f;
    float manual_yaw_ = 0.0f;
    float navigation_forward_ = 0.0f;
    float navigation_side_ = 0.0f;
    float navigation_yaw_ = 0.0f;
    bool manual_received_ = false;
    bool navigation_received_ = false;
    std::chrono::steady_clock::time_point last_manual_;
    std::chrono::steady_clock::time_point last_navigation_;
    static constexpr double kCommandTimeoutSeconds = 0.5;

    inline static const std::unordered_map<std::string, KeyCode> key_map_ = {
        {"G20_KEY_L1", KeyCode::L1}, {"G20_KEY_L2", KeyCode::L2},
        {"G20_KEY_R1", KeyCode::R1}, {"G20_KEY_R2", KeyCode::R2},
        {"G12_KEY_C", KeyCode::L1}, {"G12_KEY_A", KeyCode::L2},
        {"G12_KEY_B", KeyCode::R1}, {"G12_KEY_D", KeyCode::R2},
    };

    void SteerCallback(const drdds::msg::Steer::SharedPtr message) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        manual_forward_ = message->data.x;
        manual_side_ = message->data.y;
        manual_yaw_ = message->data.yaw;
        last_manual_ = std::chrono::steady_clock::now();
        manual_received_ = true;
    }

    void NavigationSteerCallback(const drdds::msg::Steer::SharedPtr message) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        navigation_forward_ = message->data.x;
        navigation_side_ = message->data.y;
        navigation_yaw_ = message->data.yaw;
        last_navigation_ = std::chrono::steady_clock::now();
        navigation_received_ = true;
    }

    void KeyCallback(const std_msgs::msg::String::SharedPtr message) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        auto item = key_map_.find(message->data);
        if (item == key_map_.end()) return;
        switch (item->second) {
            case KeyCode::L1:
                if (msfb_->GetCurrentState() == RobotMotionState::WaitingForStand
                    || msfb_->GetCurrentState() == RobotMotionState::LieDown) {
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
                }
                break;
            case KeyCode::L2:
                if (msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
                }
                break;
            case KeyCode::R1:
                if (msfb_->GetCurrentState() == RobotMotionState::StandingUp
                    || msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::LieDown);
                }
                break;
            case KeyCode::R2:
                usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
                break;
            default:
                break;
        }
    }

public:
    explicit DdsCommandInterface(RobotName robot_name) : UserCommandInterface(robot_name) {
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
    }

    ~DdsCommandInterface() override { Stop(); }

    void SetNode(rclcpp::Node::SharedPtr node) { node_ = std::move(node); }

    void Start() override {
        if (!node_) {
            throw std::runtime_error("DdsCommandInterface requires a ROS2 node");
        }
        steer_sub_ = node_->create_subscription<drdds::msg::Steer>(
            "/STEER", 10,
            std::bind(&DdsCommandInterface::SteerCallback, this, std::placeholders::_1));
        navigation_steer_sub_ = node_->create_subscription<drdds::msg::Steer>(
            "/s10/navigation/steer", 10,
            std::bind(&DdsCommandInterface::NavigationSteerCallback, this, std::placeholders::_1));
        key_sub_ = node_->create_subscription<std_msgs::msg::String>(
            "/GAMEPAD_KEY", 10,
            std::bind(&DdsCommandInterface::KeyCallback, this, std::placeholders::_1));
        std::cout << "DDS command interface: manual /STEER, autonomous "
                  << "/s10/navigation/steer, mode /GAMEPAD_KEY" << std::endl;
    }

    void Stop() override {
        steer_sub_.reset();
        navigation_steer_sub_.reset();
        key_sub_.reset();
        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    UserCommand* GetUserCommand() override {
        std::lock_guard<std::mutex> lock(command_mutex_);
        const auto now = std::chrono::steady_clock::now();
        const bool navigation_fresh = navigation_received_ &&
            std::chrono::duration<double>(now - last_navigation_).count() <= kCommandTimeoutSeconds;
        const bool manual_fresh = manual_received_ &&
            std::chrono::duration<double>(now - last_manual_).count() <= kCommandTimeoutSeconds;
        if (navigation_fresh) {
            usr_cmd_->forward_vel_scale = navigation_forward_;
            usr_cmd_->side_vel_scale = navigation_side_;
            usr_cmd_->turnning_vel_scale = navigation_yaw_;
        } else if (manual_fresh) {
            usr_cmd_->forward_vel_scale = manual_forward_;
            usr_cmd_->side_vel_scale = manual_side_;
            usr_cmd_->turnning_vel_scale = manual_yaw_;
        } else {
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale = 0.0f;
            usr_cmd_->turnning_vel_scale = 0.0f;
        }
        return usr_cmd_;
    }
};
