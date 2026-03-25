from . import Argphy_0 as ap0 #original
from . import Argphy_1 as ap1 #scaling
from . import Argphy_1_ as ap1_ #loss만 scaling
from . import Argphy_2 as ap2 #state 추가 + scaling
from . import Argphy_3 as ap3 #I only + scaling
from . import LatentODE as lo
from . import LatentODE as lo1
from . import NODE as no
from . import NODE_1 as no1
from . import PINN as pn
from . import model_tmp as tmp

__all__ = ["ap0",
           "ap1",
           "ap1_",
           "ap2",
           "ap3",
           "lo",
           "lo1",
           "no",
           "no1",
           "pn",
           "tmp"
           ]
