# Evaluation Artifacts

本目录只保存体积可控、可用于审计报告结论的结果，不保存原始实机点云或完整运行地图。

`localization/route_map_single_session/` 是单会话冻结地图的清单、参考轨迹和留出圈证据；
`localization/route_map_multisession_candidate/` 是多会话候选的同类证据。完整
`submaps/` 由 `deployment.mapping` 工具从原始采集重新生成。
