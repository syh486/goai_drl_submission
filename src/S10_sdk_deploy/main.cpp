#include "quadruped_wheel/qw_state_machine.hpp"

#ifdef USE_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward{
        backward::SignalHandling sh;
    }
#endif

using namespace types;
MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

int main(){
    std::cout << "State Machine Start Running" << std::endl;
    rclcpp::init(0, 0);
#if defined(S10_COMMAND_INTERFACE_DDS)
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(RobotName::S10, RemoteCommandType::kDDS);
#elif defined(S10_COMMAND_INTERFACE_GAMEPAD)
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(RobotName::S10, RemoteCommandType::kGamepad);
#else
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(RobotName::S10, RemoteCommandType::kKeyBoard);
#endif
    
    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
