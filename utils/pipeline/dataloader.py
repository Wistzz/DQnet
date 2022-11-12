# -*- coding: utf-8 -*-

from functools import partial

from torch.utils import data

from utils import builder, misc


def get_tr_loader(cfg, shuffle=True, drop_last=True, pin_memory=True):
    dataset = builder.build_obj_from_registry(
        registry_name="DATASETS",
        obj_name=cfg.datasets.train.dataset_type,
        obj_cfg=dict(
            root=[(name, path) for name, path in cfg.datasets.train.path.items()],
            shape=cfg.datasets.train.shape,
            extra_scales=cfg.train.ms.extra_scales if cfg.train.ms.enable else None,
            interp_cfg=cfg.datasets.train.get("interp_cfg", None),
        ),
    )
    if True:#cfg.use_ddp:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
    else:
        train_sampler = None
        shuffle = shuffle

    if cfg.train.ms.enable:
        collate_fn = getattr(dataset, "collate_fn", None)
        assert collate_fn is not None
    else:
        collate_fn = None

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        # shuffle=shuffle, # ddp default shuffle=True
        num_workers=cfg.train.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=partial(misc.customized_worker_init_fn, base_seed=cfg.base_seed)
        if cfg.use_custom_worker_init
        else None,
    )
    print(f"Length of Trainset: {len(dataset)}")
    return loader


def get_te_loader(cfg, shuffle=False, drop_last=False, pin_memory=True) -> list:
    for i, (te_data_name, te_data_path) in enumerate(cfg.datasets.test.path.items()):
        dataset = builder.build_obj_from_registry(
            registry_name="DATASETS",
            obj_name=cfg.datasets.test.dataset_type,
            obj_cfg=dict(
                root=(te_data_name, te_data_path),
                shape=cfg.datasets.test.shape,
                interp_cfg=cfg.datasets.test.get("interp_cfg", None),
            ),
        )

        # if True:#cfg.use_ddp:
        #     num_tasks = misc.get_world_size()
        #     global_rank = misc.get_rank()
        #     test_sampler = data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        #     shuffle = False

        loader = data.DataLoader(
            dataset=dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=partial(misc.customized_worker_init_fn, base_seed=cfg.base_seed)
            if cfg.use_custom_worker_init
            else None,
        )
        print(f"Testing with testset: {te_data_name}: {len(dataset)}")
        yield te_data_name, te_data_path, loader
