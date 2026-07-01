from axlearn.experiments.text.gpt.c4_trainer import named_trainer_configs as c4_named_trainer_configs

def elastic_test_config():
    # Load base golden run test config (very small LLaMA model)
    cfg = c4_named_trainer_configs()["fuji-test-v1"]()
    
    # Configure 6D mesh layout (data=8, fsdp=4) for v5e-32
    cfg.mesh_shape = (1, 8, 1, 4, 1, 1)
    
    # Configure checkpointer to save frequently (every 5 steps)
    cfg.checkpointer.keep_last_n = 5
    cfg.checkpointer.save_policy.n = 5
    cfg.checkpointer.save_policy.max_step = 100
    
    # Configure Host RAM snapshotting interval
    cfg.snapshot_interval = 2
    
    return cfg

def named_trainer_configs():
    return {
        "elastic_test": elastic_test_config,
    }
