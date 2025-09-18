from argparse import ArgumentParser
from pathlib import Path
import hydra
from omegaconf import OmegaConf

from stable_baselines3.common.evaluation import evaluate_policy

from src.utils.postprocessing import get_tensorboard_record, get_synced_traces, resolve_tags
from src.tools.render_policy import render_policy_to_mp4, render_policy_to_mp4_from_paths

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str)
    parser.add_argument("--render", action="store_true")
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
    env_id = cfg.env.eval_env.env_id['id']


    # exit(3)

    agent_class = hydra.utils.get_class(cfg.agent._target_)
    agent = agent_class.load(snapshot_path, device="cpu")

    n_eval_eps = cfg.callbacks.eval_callback.n_eval_episodes
    mean_rewards, std_rewards, *_ = evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=n_eval_eps,
        deterministic=True,
    )
    print(f"Mean reward over {n_eval_eps} episodes: {mean_rewards:.2f}±{std_rewards:.2f}")

    if args.render:
        from stable_baselines3 import PPO
        from src.tools.render_policy import render_policy_to_mp4

        name = snapshot_path.stem
        save_name = name + '.mp4'
        save_path = snapshot_path.parent / save_name

        # model_class = PPO
        # model_path = 'logs/runs/train_save_best/2025-08-22_18-55-24/checkpoints/best_model/best_model.zip'
        # env_id = 'CrippledAnt-v5'
        env_kwargs = {}
        out_dir = 'renders'
        video_length = 1000
        deterministic = True

        render_policy_to_mp4_from_paths(
            model_class=agent_class,
            model_path=snapshot_path,
            env_id=env_id,
            env_kwargs=env_kwargs,
            video_length=video_length,
            deterministic=deterministic
        )

    # pass
