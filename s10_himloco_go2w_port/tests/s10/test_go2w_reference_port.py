"""Static audits for the final Go2W-reference S10 port."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CFG = (ROOT / "locowheeledlegged/config/s10/go2w_him_env_cfg.py").read_text()
CORE = (ROOT / "locowheeledlegged/him/core.py").read_text()
RUNNER = (ROOT / "locowheeledlegged/him/runner.py").read_text()
ENV = (ROOT / "locowheeledlegged/envs.py").read_text()
PROTOCOL = (ROOT / "s10_policy_protocol.py").read_text()
OBSERVATIONS = (ROOT / "locowheeledlegged/mdp/s10_observations.py").read_text()
REFERENCE_REWARDS = (ROOT / "locowheeledlegged/mdp/go2w_reference.py").read_text()
ACTUATOR = (ROOT / "locowheeledlegged/assets/s10/delayed_implicit.py").read_text()
ASSET = (ROOT / "locowheeledlegged/assets/s10_robot.py").read_text()


def test_reference_is_commit_locked():
    assert 'REFERENCE_REPOSITORY = "TrackinBIT/HIMLoco-for-Go2W"' in CFG
    assert 'REFERENCE_COMMIT = "011693738c61603c3f22f2bce755098dd36fa7eb"' in CFG


def test_protocol_dimensions_and_order_are_explicit():
    policy = CFG[CFG.index("class PolicyCfg"):CFG.index("class CriticCfg")]
    critic = CFG[CFG.index("class CriticCfg"):CFG.index("class Go2WActionsCfg")]
    assert 'proprio = _proprio_term("producer")' in policy
    assert 'proprio = _proprio_term("consumer")' in critic
    assert "history_length = 6" in policy
    assert "target_slices=((3, proprio_dim + 3),)" in RUNNER
    assert "critic_dim != 262" in (ROOT / "scripts/s10/train_go2w_him.py").read_text()
    assert '0.125 if "hipx" in name else 0.25' in CFG
    assert "scale=5.0" in CFG


def test_s10_policy_robot_permutations_are_bidirectional():
    namespace: dict[str, object] = {}
    exec(compile(PROTOCOL, "s10_policy_protocol.py", "exec"), namespace)
    policy_to_robot = namespace["POLICY_TO_ROBOT_INDICES"]
    robot_to_policy = namespace["ROBOT_TO_POLICY_INDICES"]
    assert policy_to_robot == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15)
    assert robot_to_policy == (0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15)
    values = tuple(range(16))
    robot = tuple(values[index] for index in robot_to_policy)
    restored = tuple(robot[index] for index in policy_to_robot)
    assert restored == values
    assert namespace["BASE_COM_BODY"] == (-0.000512, 0.057317, 0.001182)


def test_reference_proprio_is_one_shared_57d_frame():
    function = OBSERVATIONS[
        OBSERVATIONS.index("def go2w_proprio_reference"):
        OBSERVATIONS.index("def joint_pos_rel_without_wheel_policy_order")
    ]
    order = (
        "asset.data.root_ang_vel_b * 0.25",
        "asset.data.projected_gravity_b",
        "command * command.new_tensor((2.0, 2.0, 0.25))",
        "joint_pos",
        "joint_vel * 0.05",
        "last_action",
    )
    frame = function[function.index("frame = torch.cat"):]
    assert [frame.index(term) for term in order] == sorted(frame.index(term) for term in order)
    assert 'joint_pos[:, -4:] = 0.0' in OBSERVATIONS
    assert 'joint_vel[:, -4:] = 0.0' in OBSERVATIONS
    assert 'env._go2w_shared_proprio = frame' in OBSERVATIONS
    assert 'if add_noise:' in OBSERVATIONS
    assert 'frame.shape[1] != 57' in OBSERVATIONS


def test_reference_history_reset_and_clipping_semantics_are_preserved():
    assert "saved_buffer = history._buffer[:, env_ids].clone()" in ENV
    assert "history._buffer[:, env_ids] = saved_buffer" in ENV
    assert "history._num_pushes.fill_(1)" in ENV
    assert "history._buffer[index].zero_()" in ENV
    assert "value.clamp_(min=-100.0, max=100.0)" in ENV
    assert "rewards.clamp_(min=0.0)" in ENV
    assert "clip_actions=100.0" in (ROOT / "scripts/s10/train_go2w_him.py").read_text()
    assert "saved_action = self.action_manager._action[env_ids].clone()" in ENV
    assert "terminal-action delay carryover audit passed" in RUNNER


def test_actuator_reset_targets_are_resolved_by_name_not_policy_position():
    assert "reset_position_target=DEFAULT_JOINT_POSITIONS" in ASSET
    assert "tuple(target[name] for name in self.joint_names)" in ACTUATOR
    assert "prime_command_targets" in ACTUATOR


def test_reference_reward_side_effects_are_made_explicit():
    assert "func=mdp.dof_acc_reference" in CFG
    assert "previous[:, -wheel_count:] = 0.0" in REFERENCE_REWARDS
    assert "acceleration = (effective_previous - current) / env.step_dt" in REFERENCE_REWARDS
    assert "wheel-velocity observation" in OBSERVATIONS
    # class_to_dict() in the reference iterates dir(scales), therefore dof_acc
    # executes alphabetically before the mutating dof_vel reward.
    assert sorted(("dof_acc", "dof_vel")) == ["dof_acc", "dof_vel"]


def test_runner_never_recomputes_history_without_physics():
    assert "def _cached_observations" in RUNNER
    learn_block = RUNNER[RUNNER.index("def learn("):]
    assert "self.env.get_observations()" not in learn_block
    assert "pre-reset history was cleared instead of preserved" in RUNNER
    assert "policy and critic did not receive the same noisy Go2W proprio frame" in RUNNER


def test_terminal_critic_and_heading_semantics_match_reference_execution():
    assert 'extras["go2w_terminal_critic"]' in ENV
    assert 'infos.get("go2w_terminal_critic")' in RUNNER
    assert "torch.ones_like(dones, dtype=torch.bool)" in CORE
    assert "min=-2.0" in REFERENCE_REWARDS
    assert "max=2.0" in REFERENCE_REWARDS


def test_domain_randomization_matches_reference_sampling_granularity():
    events = CFG[CFG.index("class Go2WEventsCfg"):CFG.index("class Go2WTerminationsCfg")]
    material = events[events.index("randomize_material"):events.index("reset_base")]
    assert 'mode="reset"' in material
    assert "func=mdp.randomize_go2w_friction" in material
    assert "materials[env_ids_cpu, :, 0] = friction" in REFERENCE_REWARDS
    assert "materials[env_ids_cpu, :, 1] = friction" in REFERENCE_REWARDS
    assert "func=mdp.randomize_go2w_actuator_gains" in events
    assert "kp = torch.empty((count, 1)" in REFERENCE_REWARDS
    assert "kd = torch.empty((count, 1)" in REFERENCE_REWARDS
    assert "motor = torch.empty((count, 1)" in REFERENCE_REWARDS


def test_reference_terrain_recipe():
    expected = {
        '"smooth_slope_up"': "proportion=0.05",
        '"smooth_slope_down"': "proportion=0.05",
        '"rough_slope_down"': "proportion=0.10",
        '"stairs_up"': "proportion=0.35",
        '"stairs_down"': "proportion=0.20",
        '"discrete"': "proportion=0.25",
    }
    for name, proportion in expected.items():
        start = CFG.index(name)
        assert proportion in CFG[start:start + 500]
    assert "max_init_terrain_level=5" in CFG
    assert "step_height_range=(0.05, 0.23)" in CFG


def test_reference_rewards_and_no_mirror_term():
    block = CFG[CFG.index("class Go2WRewardsCfg"):CFG.index("class Go2WEventsCfg")]
    for name, weight in {
        "tracking_lin_vel": "1.5", "tracking_ang_vel": "0.75",
        "base_height": "-10.0", "hip_default": "-0.5",
        "stand_still": "-0.5", "collision": "-1.0",
        "feet_stumble": "-0.1", "action_rate": "-0.01",
        "torques": "-5.0e-4", "dof_vel": "-1.0e-7",
        "dof_acc": "-1.0e-7", "run_still": "-0.05",
    }.items():
        term = block[block.index(name):]
        assert f"weight={weight}" in term[:700]
    assert "mirror" not in block.lower()
    assert '"std": 0.5' in block


def test_reference_actor_has_no_small_output_initialization():
    assert "gain=0.01" not in CORE
    assert "self.std = nn.Parameter" in CORE
    assert "target_dim = sum(end - start" in CORE


def test_only_base_contact_terminates_and_no_patch_boundary():
    block = CFG[CFG.index("class Go2WTerminationsCfg"):CFG.index("class Go2WCurriculumCfg")]
    assert "base_contact" in block
    assert "out_of" not in block
    assert "hip_contact" not in block
