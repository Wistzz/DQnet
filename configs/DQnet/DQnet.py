_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
model_name = "DQNet"

train = dict(
    batch_size=20,
    num_workers=4,
    use_amp=True,
    num_epochs=40,
    epoch_based=True,
    lr=1e-4,#/4,
    optimizer=dict(
        mode="adamw",
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            # momentum=0.9,
            weight_decay=0.1, # 5e-4
            # nesterov=False,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode = "poly",
        cfg=dict(
            lr_decay=0.9
        )

    ),
)

test = dict(
    batch_size=40,
    num_workers=4,
    show_bar=False,
)

datasets = dict(
    train=dict(
        dataset_type="msi_cod_tr",
        shape=dict(h=384, w=384),
        extra_scales=[0.5, 2],
        path=["cod10k_camo_tr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="msi_cod_te",
        shape=dict(h=384, w=384),
        path=["camo_te", "chameleon", "cod10k_te", "nc4k"],
        interp_cfg=dict(),
    ),
)
