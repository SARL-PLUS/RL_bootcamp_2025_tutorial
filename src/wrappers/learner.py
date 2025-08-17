from typing import Any, Optional, List

        
def sb3_learner(agent, callback: Optional[List]= None, **kwargs) -> Any: 
    if callback:
        callback = list(callback)
        
    agent.learn(callback=callback, **kwargs)
    return agent.num_timesteps
