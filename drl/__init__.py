# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.

from drl.algos.ddpg.ddpg import ddpg
from drl.algos.ppo.ppo import ppo
from drl.algos.sac.sac import sac
from drl.algos.td3.td3 import td3
from drl.algos.trpo.trpo import trpo
from drl.algos.vpg.vpg import vpg

# Loggers
from drl.utils.logx import Logger, EpochLogger

# Version
from drl.version import __version__