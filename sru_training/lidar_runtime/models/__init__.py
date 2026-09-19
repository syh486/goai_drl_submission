from .model1 import (
    DEFAULT_SRU_DEPTH_WEIGHTS,
    Model1,
    Model1Config,
    Model1View,
)
from .model1_vae import (
    Model1Vae,
    Model1VaeConfig,
    Model1VaeView,
)
from .model1_vae_resizeconv import (
    Model1VaeResizeConv,
    Model1VaeResizeConvConfig,
    Model1VaeResizeConvView,
)
from .model1_vae_var import (
    Model1VaeVar,
    Model1VaeVarConfig,
    Model1VaeVarView,
)
from .model1_hybrid_var import (
    Model1HybridVar,
    Model1HybridVarConfig,
    Model1HybridVarView,
)
from .model1_trans_learnable import (
    Model1TransLearnable,
    Model1TransLearnableConfig,
    Model1TransLearnableView,
)
from .model1_trans_phy import (
    Model1TransPhy,
    Model1TransPhyConfig,
    Model1TransPhyView,
)
from .model1_unet import (
    Model1Unet,
    Model1UnetConfig,
    Model1UnetView,
)
from .model1_vqvae import (
    Model1VqVae,
    Model1VqVaeConfig,
    Model1VqVaeView,
)
from .model1_opt_vqvae import (
    Model1OptVqVae,
    Model1OptVqVaeConfig,
    Model1OptVqVaeView,
)
from .model1_opt_hybrid import (
    Model1OptHybrid,
    Model1OptHybridConfig,
    Model1OptHybridView,
)
from .model1_opt_hybrid_additive import (
    Model1OptHybridAdditive,
    Model1OptHybridAdditiveConfig,
    Model1OptHybridAdditiveView,
)
from .model1_opt_hybrid_crossview import (
    Model1OptHybridCrossView,
    Model1OptHybridCrossViewConfig,
    Model1OptHybridCrossViewView,
)
from .model1_opt_hybrid_d2c_variants import (
    Model1OptHybridD2CAdditive,
    Model1OptHybridD2CAdditiveConfig,
    Model1OptHybridD2CAdditiveView,
    Model1OptHybridD2CAdditiveResGate,
    Model1OptHybridD2CAdditiveResGateConfig,
    Model1OptHybridD2CAdditiveResGateView,
    Model1OptHybridD2CCrossViewCont,
    Model1OptHybridD2CCrossViewContConfig,
    Model1OptHybridD2CCrossViewContView,
    Model1OptHybridD2CGate,
    Model1OptHybridD2CGateConfig,
    Model1OptHybridD2CGateView,
    Model1OptHybridD2CGeoCont,
    Model1OptHybridD2CGeoContConfig,
    Model1OptHybridD2CGeoContView,
    Model1OptHybridD2CGeoLate,
    Model1OptHybridD2CGeoLateConfig,
    Model1OptHybridD2CGeoLateView,
    Model1OptHybridD2CGeoLateWideCont,
    Model1OptHybridD2CGeoLateWideContConfig,
    Model1OptHybridD2CGeoLateWideContView,
    Model1OptHybridD2CQuantGuide,
    Model1OptHybridD2CQuantGuideConfig,
    Model1OptHybridD2CQuantGuideView,
    Model1OptHybridD2CSpatialGate,
    Model1OptHybridD2CSpatialGateConfig,
    Model1OptHybridD2CSpatialGateView,
)
from .model1_opt_hybrid_d2c_finalists import (
    Model1OptHybridD2CGeoLateAdditiveResGate,
    Model1OptHybridD2CGeoLateAdditiveResGateConfig,
    Model1OptHybridD2CGeoLateAdditiveResGateView,
    Model1OptHybridD2CGeoLateCrossViewCont,
    Model1OptHybridD2CGeoLateCrossViewContConfig,
    Model1OptHybridD2CGeoLateCrossViewContView,
    Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate,
    Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig,
    Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateView,
)
from .model1_opt_hybrid_dual import (
    Model1OptHybridDual,
    Model1OptHybridDualConfig,
    Model1OptHybridDualView,
)
from .model1_opt_hybrid_geo import (
    Model1OptHybridGeo,
    Model1OptHybridGeoConfig,
    Model1OptHybridGeoView,
)
from .model1_opt_hybrid_dual_v2 import (
    Model1OptHybridDualV2,
    Model1OptHybridDualV2Config,
    Model1OptHybridDualV2View,
)
from .model1_opt_hybrid_dual_v3 import (
    Model1OptHybridDualV3,
    Model1OptHybridDualV3C2D,
    Model1OptHybridDualV3C2DConfig,
    Model1OptHybridDualV3C2DView,
    Model1OptHybridDualV3D2C,
    Model1OptHybridDualV3D2CConfig,
    Model1OptHybridDualV3D2CView,
    Model1OptHybridDualV3Config,
    Model1OptHybridDualV3View,
    Model1OptHybridDualV3WideCont,
    Model1OptHybridDualV3WideContConfig,
    Model1OptHybridDualV3WideContView,
)
from .model1_opt_hybrid_trans_learnable import (
    Model1OptHybridTransLearnable,
    Model1OptHybridTransLearnableConfig,
    Model1OptHybridTransLearnableView,
)
from .model1_opt_hybrid_trans_phy import (
    Model1OptHybridTransPhy,
    Model1OptHybridTransPhyConfig,
    Model1OptHybridTransPhyView,
)
from .model1_vq_hybrid import (
    Model1VqHybrid,
    Model1VqHybridConfig,
    Model1VqHybridView,
)
from .model1_rvq import (
    Model1Rvq,
    Model1RvqConfig,
    Model1RvqView,
)
from .model2 import (
    Model2,
    Model2Config,
    Model2View,
)
from .model2_vae import (
    Model2Vae,
    Model2VaeConfig,
    Model2VaeView,
)
from .model2_vqvae import (
    Model2VqVae,
    Model2VqVaeConfig,
    Model2VqVaeView,
)
from .model2_unet import (
    Model2Unet,
    Model2UnetConfig,
    Model2UnetView,
)
from .model2_v2 import (
    Model2V2,
    Model2V2Config,
    Model2V2View,
)
from .model2_v3 import (
    Model2V3,
    Model2V3Config,
    Model2V3View,
)

# Backward-compatible aliases for the original single-model code path.
SruDualLidarAutoencoder = Model1
SruLidarAutoencoderConfig = Model1Config
SruLidarViewAutoencoder = Model1View

__all__ = [
    "DEFAULT_SRU_DEPTH_WEIGHTS",
    "Model1",
    "Model1Config",
    "Model1View",
    "Model1Vae",
    "Model1VaeConfig",
    "Model1VaeView",
    "Model1VaeResizeConv",
    "Model1VaeResizeConvConfig",
    "Model1VaeResizeConvView",
    "Model1VaeVar",
    "Model1VaeVarConfig",
    "Model1VaeVarView",
    "Model1HybridVar",
    "Model1HybridVarConfig",
    "Model1HybridVarView",
    "Model1TransLearnable",
    "Model1TransLearnableConfig",
    "Model1TransLearnableView",
    "Model1TransPhy",
    "Model1TransPhyConfig",
    "Model1TransPhyView",
    "Model1Unet",
    "Model1UnetConfig",
    "Model1UnetView",
    "Model1VqVae",
    "Model1VqVaeConfig",
    "Model1VqVaeView",
    "Model1OptVqVae",
    "Model1OptVqVaeConfig",
    "Model1OptVqVaeView",
    "Model1OptHybrid",
    "Model1OptHybridConfig",
    "Model1OptHybridView",
    "Model1OptHybridAdditive",
    "Model1OptHybridAdditiveConfig",
    "Model1OptHybridAdditiveView",
    "Model1OptHybridCrossView",
    "Model1OptHybridCrossViewConfig",
    "Model1OptHybridCrossViewView",
    "Model1OptHybridD2CAdditive",
    "Model1OptHybridD2CAdditiveConfig",
    "Model1OptHybridD2CAdditiveView",
    "Model1OptHybridD2CAdditiveResGate",
    "Model1OptHybridD2CAdditiveResGateConfig",
    "Model1OptHybridD2CAdditiveResGateView",
    "Model1OptHybridD2CCrossViewCont",
    "Model1OptHybridD2CCrossViewContConfig",
    "Model1OptHybridD2CCrossViewContView",
    "Model1OptHybridD2CGate",
    "Model1OptHybridD2CGateConfig",
    "Model1OptHybridD2CGateView",
    "Model1OptHybridD2CGeoCont",
    "Model1OptHybridD2CGeoContConfig",
    "Model1OptHybridD2CGeoContView",
    "Model1OptHybridD2CGeoLate",
    "Model1OptHybridD2CGeoLateConfig",
    "Model1OptHybridD2CGeoLateView",
    "Model1OptHybridD2CGeoLateWideCont",
    "Model1OptHybridD2CGeoLateWideContConfig",
    "Model1OptHybridD2CGeoLateWideContView",
    "Model1OptHybridD2CQuantGuide",
    "Model1OptHybridD2CQuantGuideConfig",
    "Model1OptHybridD2CQuantGuideView",
    "Model1OptHybridD2CSpatialGate",
    "Model1OptHybridD2CSpatialGateConfig",
    "Model1OptHybridD2CSpatialGateView",
    "Model1OptHybridD2CGeoLateAdditiveResGate",
    "Model1OptHybridD2CGeoLateAdditiveResGateConfig",
    "Model1OptHybridD2CGeoLateAdditiveResGateView",
    "Model1OptHybridD2CGeoLateCrossViewCont",
    "Model1OptHybridD2CGeoLateCrossViewContConfig",
    "Model1OptHybridD2CGeoLateCrossViewContView",
    "Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate",
    "Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig",
    "Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateView",
    "Model1OptHybridDual",
    "Model1OptHybridDualConfig",
    "Model1OptHybridDualView",
    "Model1OptHybridGeo",
    "Model1OptHybridGeoConfig",
    "Model1OptHybridGeoView",
    "Model1OptHybridDualV2",
    "Model1OptHybridDualV2Config",
    "Model1OptHybridDualV2View",
    "Model1OptHybridDualV3",
    "Model1OptHybridDualV3C2D",
    "Model1OptHybridDualV3C2DConfig",
    "Model1OptHybridDualV3C2DView",
    "Model1OptHybridDualV3D2C",
    "Model1OptHybridDualV3D2CConfig",
    "Model1OptHybridDualV3D2CView",
    "Model1OptHybridDualV3Config",
    "Model1OptHybridDualV3View",
    "Model1OptHybridDualV3WideCont",
    "Model1OptHybridDualV3WideContConfig",
    "Model1OptHybridDualV3WideContView",
    "Model1OptHybridTransLearnable",
    "Model1OptHybridTransLearnableConfig",
    "Model1OptHybridTransLearnableView",
    "Model1OptHybridTransPhy",
    "Model1OptHybridTransPhyConfig",
    "Model1OptHybridTransPhyView",
    "Model1VqHybrid",
    "Model1VqHybridConfig",
    "Model1VqHybridView",
    "Model1Rvq",
    "Model1RvqConfig",
    "Model1RvqView",
    "Model2",
    "Model2Config",
    "Model2View",
    "Model2Vae",
    "Model2VaeConfig",
    "Model2VaeView",
    "Model2VqVae",
    "Model2VqVaeConfig",
    "Model2VqVaeView",
    "Model2Unet",
    "Model2UnetConfig",
    "Model2UnetView",
    "Model2V2",
    "Model2V2Config",
    "Model2V2View",
    "Model2V3",
    "Model2V3Config",
    "Model2V3View",
    "SruDualLidarAutoencoder",
    "SruLidarAutoencoderConfig",
    "SruLidarViewAutoencoder",
]
