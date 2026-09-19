from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from distance_representation import DEFAULT_FLAT_LIDAR_REFERENCE
from models.model1 import Model1
from models.model1_trans_learnable import Model1TransLearnable, Model1TransLearnableConfig
from models.model1_trans_phy import Model1TransPhy, Model1TransPhyConfig
from models.model1_unet import Model1Unet
from models.model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig
from models.model1_opt_hybrid_additive import Model1OptHybridAdditive, Model1OptHybridAdditiveConfig
from models.model1_opt_hybrid_crossview import Model1OptHybridCrossView, Model1OptHybridCrossViewConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CAdditive, Model1OptHybridD2CAdditiveConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CAdditiveResGate, Model1OptHybridD2CAdditiveResGateConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CCrossViewCont, Model1OptHybridD2CCrossViewContConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CGate, Model1OptHybridD2CGateConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CGeoCont, Model1OptHybridD2CGeoContConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CGeoLate, Model1OptHybridD2CGeoLateConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CGeoLateWideCont, Model1OptHybridD2CGeoLateWideContConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CQuantGuide, Model1OptHybridD2CQuantGuideConfig
from models.model1_opt_hybrid_d2c_variants import Model1OptHybridD2CSpatialGate, Model1OptHybridD2CSpatialGateConfig
from models.model1_opt_hybrid_d2c_finalists import Model1OptHybridD2CGeoLateAdditiveResGate, Model1OptHybridD2CGeoLateAdditiveResGateConfig
from models.model1_opt_hybrid_d2c_finalists import Model1OptHybridD2CGeoLateCrossViewCont, Model1OptHybridD2CGeoLateCrossViewContConfig
from models.model1_opt_hybrid_d2c_finalists import Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate, Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig
from models.model1_opt_hybrid_dual import Model1OptHybridDual, Model1OptHybridDualConfig
from models.model1_opt_hybrid_dual_v2 import Model1OptHybridDualV2, Model1OptHybridDualV2Config
from models.model1_opt_hybrid_dual_v3 import Model1OptHybridDualV3, Model1OptHybridDualV3Config
from models.model1_opt_hybrid_dual_v3 import Model1OptHybridDualV3C2D, Model1OptHybridDualV3C2DConfig
from models.model1_opt_hybrid_dual_v3 import Model1OptHybridDualV3D2C, Model1OptHybridDualV3D2CConfig
from models.model1_opt_hybrid_dual_v3 import Model1OptHybridDualV3WideCont, Model1OptHybridDualV3WideContConfig
from models.model1_opt_hybrid_trans_learnable import Model1OptHybridTransLearnable, Model1OptHybridTransLearnableConfig
from models.model1_opt_hybrid_trans_phy import Model1OptHybridTransPhy, Model1OptHybridTransPhyConfig
from models.model1_opt_vqvae import Model1OptVqVae, Model1OptVqVaeConfig
from models.model1_opt_hybrid_geo import Model1OptHybridGeo, Model1OptHybridGeoConfig
from models.model1_rvq import Model1Rvq, Model1RvqConfig
from models.model1_vae import Model1Vae
from models.model1_vae_resizeconv import Model1VaeResizeConv
from models.model1_hybrid_var import Model1HybridVar, Model1HybridVarConfig
from models.model1_vae_var import Model1VaeVar, Model1VaeVarConfig
from models.model1_vq_hybrid import Model1VqHybrid, Model1VqHybridConfig
from models.model1_vqvae import Model1VqVae, Model1VqVaeConfig
from models.model2 import Model2
from models.model2_unet import Model2Unet
from models.model2_vae import Model2Vae
from models.model2_vqvae import Model2VqVae, Model2VqVaeConfig
from models.model2_v2 import Model2V2
from models.model2_v3 import Model2V3


@dataclass(frozen=True)
class InputMapping:
    front_target_idx: int
    rear_target_idx: int
    front_input_indices: tuple[int, ...]
    rear_input_indices: tuple[int, ...]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], torch.nn.Module]
    input_mapping: InputMapping
    distance_scale: float
    z_channel_definition: str
    z_metric_range: float
    distance_representation: str = "raw_distance"
    flat_reference_path: str | None = None
    target_metric_min: float = 0.0
    target_metric_max: float = 10.0


MODEL_SPECS: dict[str, ModelSpec] = {
    "model1": ModelSpec(
        name="model1",
        builder=Model1,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_vae": ModelSpec(
        name="model1_vae",
        builder=Model1Vae,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_vae_resizeconv": ModelSpec(
        name="model1_vae_resizeconv",
        builder=Model1VaeResizeConv,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_trans_learnable": ModelSpec(
        name="model1_trans_learnable",
        builder=Model1TransLearnable,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_trans_phy": ModelSpec(
        name="model1_trans_phy",
        builder=Model1TransPhy,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_vae_var": ModelSpec(
        name="model1_vae_var",
        builder=Model1VaeVar,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_hybrid_var": ModelSpec(
        name="model1_hybrid_var",
        builder=Model1HybridVar,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_unet": ModelSpec(
        name="model1_unet",
        builder=Model1Unet,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_vqvae": ModelSpec(
        name="model1_vqvae",
        builder=Model1VqVae,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_vqvae": ModelSpec(
        name="model1_opt_vqvae",
        builder=Model1OptVqVae,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_vq_hybrid": ModelSpec(
        name="model1_vq_hybrid",
        builder=Model1VqHybrid,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid": ModelSpec(
        name="model1_opt_hybrid",
        builder=Model1OptHybrid,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_geo": ModelSpec(
        name="model1_opt_hybrid_geo",
        builder=Model1OptHybridGeo,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_additive": ModelSpec(
        name="model1_opt_hybrid_additive",
        builder=Model1OptHybridAdditive,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_crossview": ModelSpec(
        name="model1_opt_hybrid_crossview",
        builder=Model1OptHybridCrossView,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_cont": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_cont",
        builder=Model1OptHybridD2CGeoCont,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_gate": ModelSpec(
        name="model1_opt_hybrid_d2c_gate",
        builder=Model1OptHybridD2CGate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_late": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_late",
        builder=Model1OptHybridD2CGeoLate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_late_widecont": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_late_widecont",
        builder=Model1OptHybridD2CGeoLateWideCont,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_spatial_gate": ModelSpec(
        name="model1_opt_hybrid_d2c_spatial_gate",
        builder=Model1OptHybridD2CSpatialGate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_qguide": ModelSpec(
        name="model1_opt_hybrid_d2c_qguide",
        builder=Model1OptHybridD2CQuantGuide,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_additive_resgate": ModelSpec(
        name="model1_opt_hybrid_d2c_additive_resgate",
        builder=Model1OptHybridD2CAdditiveResGate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_crossview_cont": ModelSpec(
        name="model1_opt_hybrid_d2c_crossview_cont",
        builder=Model1OptHybridD2CCrossViewCont,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_additive": ModelSpec(
        name="model1_opt_hybrid_d2c_additive",
        builder=Model1OptHybridD2CAdditive,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_late_additive_resgate": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_late_additive_resgate",
        builder=Model1OptHybridD2CGeoLateAdditiveResGate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_late_crossview_cont": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_late_crossview_cont",
        builder=Model1OptHybridD2CGeoLateCrossViewCont,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_d2c_geo_late_crossview_cont_additive_resgate": ModelSpec(
        name="model1_opt_hybrid_d2c_geo_late_crossview_cont_additive_resgate",
        builder=Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual": ModelSpec(
        name="model1_opt_hybrid_dual",
        builder=Model1OptHybridDual,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual_v2": ModelSpec(
        name="model1_opt_hybrid_dual_v2",
        builder=Model1OptHybridDualV2,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual_v3": ModelSpec(
        name="model1_opt_hybrid_dual_v3",
        builder=Model1OptHybridDualV3,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual_v3_c2d": ModelSpec(
        name="model1_opt_hybrid_dual_v3_c2d",
        builder=Model1OptHybridDualV3C2D,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual_v3_d2c": ModelSpec(
        name="model1_opt_hybrid_dual_v3_d2c",
        builder=Model1OptHybridDualV3D2C,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_dual_v3_widecont": ModelSpec(
        name="model1_opt_hybrid_dual_v3_widecont",
        builder=Model1OptHybridDualV3WideCont,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_trans_learnable": ModelSpec(
        name="model1_opt_hybrid_trans_learnable",
        builder=Model1OptHybridTransLearnable,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_opt_hybrid_trans_phy": ModelSpec(
        name="model1_opt_hybrid_trans_phy",
        builder=Model1OptHybridTransPhy,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model1_rvq": ModelSpec(
        name="model1_rvq",
        builder=Model1Rvq,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="raw_distance",
        flat_reference_path=None,
        target_metric_min=0.0,
        target_metric_max=10.0,
    ),
    "model2": ModelSpec(
        name="model2",
        builder=Model2,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-4.0,
        target_metric_max=10.0,
    ),
    "model2_v2": ModelSpec(
        name="model2_v2",
        builder=Model2V2,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-4.0,
        target_metric_max=10.0,
    ),
    "model2_v3": ModelSpec(
        name="model2_v3",
        builder=Model2V3,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-10.0,
        target_metric_max=10.0,
    ),
    "model2_vae": ModelSpec(
        name="model2_vae",
        builder=Model2Vae,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-10.0,
        target_metric_max=10.0,
    ),
    "model2_vqvae": ModelSpec(
        name="model2_vqvae",
        builder=Model2VqVae,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-10.0,
        target_metric_max=10.0,
    ),
    "model2_unet": ModelSpec(
        name="model2_unet",
        builder=Model2Unet,
        input_mapping=InputMapping(
            front_target_idx=0,
            rear_target_idx=1,
            front_input_indices=(2, 4),
            rear_input_indices=(3, 5),
        ),
        distance_scale=10.0,
        z_channel_definition="world_z",
        z_metric_range=3.0,
        distance_representation="flat_minus_distance",
        flat_reference_path=str(DEFAULT_FLAT_LIDAR_REFERENCE),
        target_metric_min=-10.0,
        target_metric_max=10.0,
    ),
}


DEFAULT_MODEL_NAME = "model1"


def resolve_model_config(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if model_name == "model1_vqvae":
        resolved = asdict(Model1VqVaeConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_vqvae":
        resolved = asdict(Model1OptVqVaeConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_vae_var":
        resolved = asdict(Model1VaeVarConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_trans_learnable":
        resolved = asdict(Model1TransLearnableConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_trans_phy":
        resolved = asdict(Model1TransPhyConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_hybrid_var":
        resolved = asdict(Model1HybridVarConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_vq_hybrid":
        resolved = asdict(Model1VqHybridConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid":
        resolved = asdict(Model1OptHybridConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_geo":
        resolved = asdict(Model1OptHybridGeoConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_additive":
        resolved = asdict(Model1OptHybridAdditiveConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_crossview":
        resolved = asdict(Model1OptHybridCrossViewConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_cont":
        resolved = asdict(Model1OptHybridD2CGeoContConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_gate":
        resolved = asdict(Model1OptHybridD2CGateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_late":
        resolved = asdict(Model1OptHybridD2CGeoLateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_late_widecont":
        resolved = asdict(Model1OptHybridD2CGeoLateWideContConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_spatial_gate":
        resolved = asdict(Model1OptHybridD2CSpatialGateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_qguide":
        resolved = asdict(Model1OptHybridD2CQuantGuideConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_additive_resgate":
        resolved = asdict(Model1OptHybridD2CAdditiveResGateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_crossview_cont":
        resolved = asdict(Model1OptHybridD2CCrossViewContConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_additive":
        resolved = asdict(Model1OptHybridD2CAdditiveConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_late_additive_resgate":
        resolved = asdict(Model1OptHybridD2CGeoLateAdditiveResGateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_late_crossview_cont":
        resolved = asdict(Model1OptHybridD2CGeoLateCrossViewContConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_d2c_geo_late_crossview_cont_additive_resgate":
        resolved = asdict(Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual":
        resolved = asdict(Model1OptHybridDualConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual_v2":
        resolved = asdict(Model1OptHybridDualV2Config())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual_v3":
        resolved = asdict(Model1OptHybridDualV3Config())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual_v3_c2d":
        resolved = asdict(Model1OptHybridDualV3C2DConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual_v3_d2c":
        resolved = asdict(Model1OptHybridDualV3D2CConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_dual_v3_widecont":
        resolved = asdict(Model1OptHybridDualV3WideContConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_trans_learnable":
        resolved = asdict(Model1OptHybridTransLearnableConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_opt_hybrid_trans_phy":
        resolved = asdict(Model1OptHybridTransPhyConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model1_rvq":
        resolved = asdict(Model1RvqConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_name == "model2_vqvae":
        resolved = asdict(Model2VqVaeConfig())
        if model_config is not None:
            resolved.update(model_config)
        return resolved
    if model_config:
        raise ValueError(f"Model '{model_name}' does not support explicit model_config overrides.")
    return None


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model_name '{model_name}'. Available models: {available}")
    return MODEL_SPECS[model_name]


def build_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    spec = get_model_spec(model_name)
    resolved_model_config = resolve_model_config(model_name, model_config)
    if model_name == "model1_vqvae":
        return Model1VqVae(config=Model1VqVaeConfig(**resolved_model_config))
    if model_name == "model1_opt_vqvae":
        return Model1OptVqVae(config=Model1OptVqVaeConfig(**resolved_model_config))
    if model_name == "model1_vae_var":
        return Model1VaeVar(config=Model1VaeVarConfig(**resolved_model_config))
    if model_name == "model1_trans_learnable":
        return Model1TransLearnable(config=Model1TransLearnableConfig(**resolved_model_config))
    if model_name == "model1_trans_phy":
        return Model1TransPhy(config=Model1TransPhyConfig(**resolved_model_config))
    if model_name == "model1_hybrid_var":
        return Model1HybridVar(config=Model1HybridVarConfig(**resolved_model_config))
    if model_name == "model1_vq_hybrid":
        return Model1VqHybrid(config=Model1VqHybridConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid":
        return Model1OptHybrid(config=Model1OptHybridConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_geo":
        return Model1OptHybridGeo(config=Model1OptHybridGeoConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_additive":
        return Model1OptHybridAdditive(config=Model1OptHybridAdditiveConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_crossview":
        return Model1OptHybridCrossView(config=Model1OptHybridCrossViewConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_geo_cont":
        return Model1OptHybridD2CGeoCont(config=Model1OptHybridD2CGeoContConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_gate":
        return Model1OptHybridD2CGate(config=Model1OptHybridD2CGateConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_geo_late":
        return Model1OptHybridD2CGeoLate(config=Model1OptHybridD2CGeoLateConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_geo_late_widecont":
        return Model1OptHybridD2CGeoLateWideCont(config=Model1OptHybridD2CGeoLateWideContConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_spatial_gate":
        return Model1OptHybridD2CSpatialGate(config=Model1OptHybridD2CSpatialGateConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_qguide":
        return Model1OptHybridD2CQuantGuide(config=Model1OptHybridD2CQuantGuideConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_additive_resgate":
        return Model1OptHybridD2CAdditiveResGate(config=Model1OptHybridD2CAdditiveResGateConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_crossview_cont":
        return Model1OptHybridD2CCrossViewCont(config=Model1OptHybridD2CCrossViewContConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_additive":
        return Model1OptHybridD2CAdditive(config=Model1OptHybridD2CAdditiveConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_d2c_geo_late_additive_resgate":
        return Model1OptHybridD2CGeoLateAdditiveResGate(
            config=Model1OptHybridD2CGeoLateAdditiveResGateConfig(**resolved_model_config)
        )
    if model_name == "model1_opt_hybrid_d2c_geo_late_crossview_cont":
        return Model1OptHybridD2CGeoLateCrossViewCont(
            config=Model1OptHybridD2CGeoLateCrossViewContConfig(**resolved_model_config)
        )
    if model_name == "model1_opt_hybrid_d2c_geo_late_crossview_cont_additive_resgate":
        return Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate(
            config=Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig(**resolved_model_config)
        )
    if model_name == "model1_opt_hybrid_dual":
        return Model1OptHybridDual(config=Model1OptHybridDualConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_dual_v2":
        return Model1OptHybridDualV2(config=Model1OptHybridDualV2Config(**resolved_model_config))
    if model_name == "model1_opt_hybrid_dual_v3":
        return Model1OptHybridDualV3(config=Model1OptHybridDualV3Config(**resolved_model_config))
    if model_name == "model1_opt_hybrid_dual_v3_c2d":
        return Model1OptHybridDualV3C2D(config=Model1OptHybridDualV3C2DConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_dual_v3_d2c":
        return Model1OptHybridDualV3D2C(config=Model1OptHybridDualV3D2CConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_dual_v3_widecont":
        return Model1OptHybridDualV3WideCont(config=Model1OptHybridDualV3WideContConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_trans_learnable":
        return Model1OptHybridTransLearnable(config=Model1OptHybridTransLearnableConfig(**resolved_model_config))
    if model_name == "model1_opt_hybrid_trans_phy":
        return Model1OptHybridTransPhy(config=Model1OptHybridTransPhyConfig(**resolved_model_config))
    if model_name == "model1_rvq":
        return Model1Rvq(config=Model1RvqConfig(**resolved_model_config))
    if model_name == "model2_vqvae":
        return Model2VqVae(config=Model2VqVaeConfig(**resolved_model_config))
    return spec.builder()
