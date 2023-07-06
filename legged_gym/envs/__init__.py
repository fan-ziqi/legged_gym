# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO, A1FlatCfg, A1FlatCfgPPO
from .cyberdog.cyberdog_config import CyberdogRoughCfg, CyberdogRoughCfgPPO, CyberdogFlatCfg, CyberdogFlatCfgPPO, CyberdogMeihuaCfg, CyberdogMeihuaCfgPPO
from .keti1.keti1_config import Keti1RoughCfg, Keti1RoughCfgPPO, Keti1FlatCfg, Keti1FlatCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO, Go1FlatCfg, Go1MrssNovel, Go1FlatCfgPPO, Go1FlatNoVelCfg, Go1FlatNoVelCfgPPO, Go1MrssNovelRough


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1_rough", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_flat", LeggedRobot, A1FlatCfg(), A1FlatCfgPPO())
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "cyberdog_rough", LeggedRobot, CyberdogRoughCfg(), CyberdogRoughCfgPPO() )
task_registry.register( "cyberdog_flat", LeggedRobot, CyberdogFlatCfg(), CyberdogFlatCfgPPO() )
task_registry.register( "cyberdog_meihua", LeggedRobot, CyberdogMeihuaCfg(), CyberdogMeihuaCfgPPO() )
task_registry.register( "keti1_rough", LeggedRobot, Keti1RoughCfg(), Keti1RoughCfgPPO() )
task_registry.register( "keti1_flat", LeggedRobot, Keti1FlatCfg(), Keti1FlatCfgPPO() )
task_registry.register("go1_rough", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
task_registry.register("go1_flat", LeggedRobot, Go1FlatCfg(), Go1FlatCfgPPO())
task_registry.register("go1_flat_novel", LeggedRobot, Go1FlatNoVelCfg(), Go1FlatNoVelCfgPPO())
task_registry.register("go1_mrss_novel", LeggedRobot, Go1MrssNovel(), Go1FlatNoVelCfgPPO())
task_registry.register("go1_mrss_novel_rough", LeggedRobot, Go1MrssNovelRough(), Go1FlatNoVelCfgPPO())
