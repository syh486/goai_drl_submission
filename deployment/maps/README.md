# Runtime Maps

完整路线地图不随源码仓库分发。将构建完成的地图目录复制或链接为：

```text
deployment/maps/current/
  localization_map_manifest.json
  canonical_route_poses.npy
  canonical_route_trajectory.npz
  submaps/
```

也可以设置 `S10_ROUTE_MAP_DIR=/absolute/path/to/map`。比赛期间的地图清单和评测结果位于
`artifacts/localization/`，其中不包含运行所需的完整 `submaps/`。

