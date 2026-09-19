// keyboard_interface.hpp
#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace interface;
using namespace types;

class KeyboardInterface : public UserCommandInterface
{
private:
    std::atomic<bool> running_{false};
    std::thread kb_thread_;
    mutable std::mutex keys_mutex_;
    mutable std::mutex ros_cmd_mutex_;

    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    float ros_forward_ = 0.0f;
    float ros_side_ = 0.0f;
    float ros_yaw_ = 0.0f;
    std::chrono::steady_clock::time_point last_ros_cmd_time_{};
    bool received_ros_cmd_ = false;
    bool auto_start_ = false;
    bool auto_stand_requested_ = false;
    bool auto_rl_requested_ = false;

    float max_forward_ = 1.0f;
    float max_side_    = 0.6f;
    float max_yaw_     = 1.0f;

    std::unordered_set<char> held_keys_;
    std::unordered_map<char, double> last_seen_time_;
    
    const std::unordered_set<char> velocity_keys_ = {'w', 's', 'a', 'd', 'q', 'e'};
    const double key_timeout_ms_ = 500.0;
    const double ros_cmd_timeout_ms_ = 500.0;

    void ClipNumber(float& num, float low, float high)
    {
        if (num < low) num = low;
        if (num > high) num = high;
    }

    double GetCurrentTimeStamp()
    {
        static auto start = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }

    static void setup_raw_mode()
    {
        termios t{};
        tcgetattr(STDIN_FILENO, &t);
        termios raw = t;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    static void restore_terminal()
    {
        termios t{};
        tcgetattr(STDIN_FILENO, &t);
        t.c_lflag |= (ECHO | ICANON);
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
    }

    void compute_velocity_from_held_keys(float& fwd, float& side, float& yaw)
    {
        fwd = 0.0f;
        side = 0.0f;
        yaw = 0.0f;

        std::lock_guard<std::mutex> lock(keys_mutex_);
        
        if (held_keys_.count('w')) fwd += max_forward_;
        if (held_keys_.count('s')) fwd -= max_forward_;
        if (held_keys_.count('a')) side += max_side_;
        if (held_keys_.count('d')) side -= max_side_;
        if (held_keys_.count('q')) yaw += max_yaw_;
        if (held_keys_.count('e')) yaw -= max_yaw_;
        
        ClipNumber(fwd, -max_forward_, max_forward_);
        ClipNumber(side, -max_side_, max_side_);
        ClipNumber(yaw, -max_yaw_, max_yaw_);
    }

    bool compute_velocity_from_ros(float& fwd, float& side, float& yaw)
    {
        std::lock_guard<std::mutex> lock(ros_cmd_mutex_);
        if (!received_ros_cmd_) return false;

        const auto age_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - last_ros_cmd_time_).count();
        if (age_ms > ros_cmd_timeout_ms_) return false;

        fwd = ros_forward_;
        side = ros_side_;
        yaw = ros_yaw_;
        return true;
    }

    bool keyboard_velocity_active() const
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        return !held_keys_.empty();
    }

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(ros_cmd_mutex_);
        ros_forward_ = static_cast<float>(msg->linear.x);
        ros_side_ = static_cast<float>(msg->linear.y);
        ros_yaw_ = static_cast<float>(msg->angular.z);
        ClipNumber(ros_forward_, -max_forward_, max_forward_);
        ClipNumber(ros_side_, -max_side_, max_side_);
        ClipNumber(ros_yaw_, -max_yaw_, max_yaw_);
        last_ros_cmd_time_ = std::chrono::steady_clock::now();
        received_ros_cmd_ = true;
        RCLCPP_INFO_THROTTLE(
            ros_node_->get_logger(),
            *ros_node_->get_clock(),
            2000,
            "Received /cmd_vel: forward=%.3f lateral=%.3f yaw=%.3f",
            ros_forward_,
            ros_side_,
            ros_yaw_);
    }

    void process_mode_command(char k)
    {
        if (k == 'r') {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            std::cout << "[MODE] Joint Damping\n";
        }
        else if (k == 'z' && (msfb_->GetCurrentState() == RobotMotionState::WaitingForStand
            || msfb_->GetCurrentState() == RobotMotionState::LieDown)) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
            std::cout << "[MODE] Standing Up\n";
        }
        else if (k == 'c' && msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
            std::cout << "[MODE] RL Control\n";
        }
        else if (k == 'x' && (msfb_->GetCurrentState() == RobotMotionState::StandingUp 
            || msfb_->GetCurrentState() == RobotMotionState::RLControlMode)) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::LieDown);
            std::cout << "[MODE] Lie Down\n";
        }
    }

    void process_auto_start()
    {
        if (!auto_start_ || msfb_ == nullptr || usr_cmd_->safe_control_mode != 0) {
            return;
        }

        const auto current_state = msfb_->GetCurrentState();
        if (!auto_stand_requested_
            && (current_state == RobotMotionState::WaitingForStand
                || current_state == RobotMotionState::LieDown)) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
            auto_stand_requested_ = true;
            std::cout << "[AUTO START] Standing Up requested\n";
        }
        else if (!auto_rl_requested_ && current_state == RobotMotionState::StandingUp) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
            auto_rl_requested_ = true;
            std::cout << "[AUTO START] RL Control requested\n";
        }
    }

    void keyboard_loop()
    {
        setup_raw_mode();

        std::cout << "\n╔════════════════════════════════════════════════╗\n"
                  << "║      KEYBOARD TELEOP - MULTI-KEY READY         ║\n"
                  << "╚════════════════════════════════════════════════╝\n"
                  << "  Movement:  W/S (forward/back)  A/D (left/right)\n"
                  << "  Rotation:  Q (CCW)  E (CW)\n"
                  << "  Mode:      R (damping)  Z (stand)  C (control)\n"
                  << "\n";

        char ch;

        while (running_) {
            double now = GetCurrentTimeStamp();
            usr_cmd_->time_stamp = now;
            process_auto_start();

            // Read all available keyboard input
            while (read(STDIN_FILENO, &ch, 1) == 1) {
                char k = std::tolower(static_cast<unsigned char>(ch));

                // Handle mode commands
                if (k == 'r' || k == 'z' || k == 'c' || k == 'x') {
                    process_mode_command(k);
                    continue;
                }

                // Track velocity keys
                if (velocity_keys_.count(k)) {
                    std::lock_guard<std::mutex> lock(keys_mutex_);
                    held_keys_.insert(k);
                    last_seen_time_[k] = now;
                }
            }

            // Remove keys that haven't been seen recently (released)
            {
                std::lock_guard<std::mutex> lock(keys_mutex_);
                std::vector<char> to_remove;
                
                for (char k : held_keys_) {
                    if (now - last_seen_time_[k] > key_timeout_ms_) {
                        to_remove.push_back(k);
                    }
                }
                
                for (char k : to_remove) {
                    held_keys_.erase(k);
                    last_seen_time_.erase(k);
                }
            }

            // Compute velocity from all currently held keys
            float fwd = 0.0f, side = 0.0f, yaw = 0.0f;
            
            if (msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                if (keyboard_velocity_active()) {
                    compute_velocity_from_held_keys(fwd, side, yaw);
                } else {
                    compute_velocity_from_ros(fwd, side, yaw);
                }
            }
            
            usr_cmd_->forward_vel_scale  = fwd;
            usr_cmd_->side_vel_scale     = side;
            usr_cmd_->turnning_vel_scale = yaw;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        restore_terminal();
        std::cout << "\n[KEYBOARD] Stopped.\n";
    }

public:
    KeyboardInterface(RobotName robot_name) : UserCommandInterface(robot_name)
    {
        std::cout << "[KeyboardInterface] Initialized with multi-key support\n";
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
        const char* auto_start_value = std::getenv("S10_AUTO_START");
        if (auto_start_value != nullptr) {
            const std::string value(auto_start_value);
            auto_start_ = value == "1" || value == "true" || value == "TRUE"
                || value == "yes" || value == "YES";
        }
        if (auto_start_) {
            std::cout << "[KeyboardInterface] Automatic stand and RL control enabled\n";
        }
    }

    ~KeyboardInterface() 
    { 
        Stop(); 
    }

    void Start() override
    {
        if (running_) return;
        running_ = true;
        kb_thread_ = std::thread(&KeyboardInterface::keyboard_loop, this);
    }

    void Stop() override
    {
        running_ = false;
        if (kb_thread_.joinable()) {
            kb_thread_.join();
        }
        
        std::lock_guard<std::mutex> lock(keys_mutex_);
        held_keys_.clear();
        last_seen_time_.clear();
        
        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    UserCommand* GetUserCommand() override 
    { 
        return usr_cmd_; 
    }

    void set_max_velocities(float fwd, float side, float yaw)
    {
        max_forward_ = std::abs(fwd);
        max_side_    = std::abs(side);
        max_yaw_     = std::abs(yaw);
        std::cout << "[CONFIG] Max velocities: fwd=" << max_forward_ 
                  << " side=" << max_side_ 
                  << " yaw=" << max_yaw_ << "\n";
    }

    void AttachRosNode(const rclcpp::Node::SharedPtr& node)
    {
        ros_node_ = node;
        cmd_vel_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel",
            10,
            std::bind(&KeyboardInterface::cmd_vel_callback, this, std::placeholders::_1));
        RCLCPP_INFO(
            ros_node_->get_logger(),
            "High-level velocity input ready on /cmd_vel (Twist), timeout %.1f s",
            ros_cmd_timeout_ms_ / 1000.0);
    }
};
