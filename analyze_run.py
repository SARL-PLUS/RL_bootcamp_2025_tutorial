from argparse import ArgumentParser
from pathlib import Path
import hydra
from omegaconf import OmegaConf

from stable_baselines3.common.evaluation import evaluate_policy
from src.utils.postprocessing import get_tensorboard_record, get_synced_traces, resolve_tags


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str)
    args = parser.parse_args()    
    
    run_path = Path().cwd().joinpath(args.run)
    cfg = OmegaConf.load(run_path.joinpath(".hydra", "config.yaml"))
    
    
    # analyze tensorboard record
    tb_record = get_tensorboard_record(run_path)
    rollout_tags = resolve_tags(obj=tb_record, prefix="rollout/")
    rollout = get_synced_traces(ea=tb_record, tags=rollout_tags)
    #
    # TODO: Plot rollout data
    
    train_tags = resolve_tags(obj=tb_record, prefix="rollout/")
    train = get_synced_traces(ea=tb_record, tags=train_tags)
    
    snapshot_path = run_path.joinpath("checkpoints", "best_model", "best_model.zip")
    env = hydra.utils.instantiate(cfg.env.eval_env)
    agent = hydra.utils.get_class(cfg.agent._target_).load(snapshot_path, device="cpu")

    mean_rewards, std_rewards, *_ = evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=cfg.callbacks.eval_callback.n_eval_episodes,
        deterministic=True,
    )
    print(mean_rewards, std_rewards)
    # TODO: Evaluate agent (e.g. render it)



    pass

            
        