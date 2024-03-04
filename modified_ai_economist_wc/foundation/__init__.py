# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

#from modified_ai_economist_wc.foundation import utils
#from modified_ai_economist_wc.foundation.agents import agent_registry as agents
#from modified_ai_economist_wc.foundation.components import component_registry as components
#from modified_ai_economist_wc.foundation.entities import endogenous_registry as endogenous
#from modified_ai_economist_wc.foundation.entities import landmark_registry as landmarks
#from modified_ai_economist_wc.foundation.entities import resource_registry as resources
#from modified_ai_economist_wc.foundation.scenarios import scenario_registry as scenarios

from modified_ai_economist_wc.foundation import utils
from modified_ai_economist_wc.foundation.base.base_agent import agent_registry as agents
from modified_ai_economist_wc.foundation.base.base_component import component_registry as components
from modified_ai_economist_wc.foundation.entities.endogenous import endogenous_registry as endogenous
from modified_ai_economist_wc.foundation.entities.landmarks import landmark_registry as landmarks
from modified_ai_economist_wc.foundation.entities.resources import resource_registry as resources
from modified_ai_economist_wc.foundation.base.base_env import scenario_registry as scenarios

#def make_env_instance(scenario_name, **kwargs):
#    scenario_class = scenarios.get(scenario_name)
#    return scenario_class(**kwargs)