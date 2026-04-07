from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy

from libero.lifelong.models.base_policy import get_policy_class, get_policy_list

# External policies (import to trigger PolicyMeta registration)
try:
    from diffusion_policy.diffusion_policy import DiffusionPolicy  # noqa: F401
except ImportError:
    pass

try:
    from flow_matching.flow_matching_policy import FlowMatchingPolicy  # noqa: F401
except ImportError:
    pass

try:
    from act.act_policy import ACTPolicy  # noqa: F401
except ImportError:
    pass
