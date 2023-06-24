import torch, json
import pytorch_lightning as pl
from bondnet.data.dataset import ReactionNetworkDatasetPrecomputed
from bondnet.data.dataloader import DataLoaderPrecomputedReactionGraphs
from bondnet.model.training_utils import (
    get_grapher,
    load_model_lightning,
)
from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)

# suppress warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.trainer.data_loading",
    lineno=102,
)

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs


def get_defaults():
    config = "./settings.json"
    config = json.load(open(config, "r"))
    return config


def test_model_construction():
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=False,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )
    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    model = load_model_lightning(config, device=device, load_dir="./test_checkpoints/")
    assert type(model) == GatedGCNReactionNetworkLightning


def test_model_save_load():
    precision = 16
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=False,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )

    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    model = load_model_lightning(config, device=device, load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, train_loader)
    trainer.save_checkpoint("./test_save_load/test.ckpt")
    config["restore_path"] = "./test_save_load/test.ckpt"
    config["restore"] = True
    model_restart = load_model_lightning(
        config, device=device, load_dir="./test_save_load/"
    )

    trainer_restart = pl.Trainer(resume_from_checkpoint="./test_save_load/test.ckpt")
    trainer_restart.fit_loop.max_epochs = 5
    trainer_restart.fit(model_restart, train_loader, train_loader)

    assert type(model_restart) == GatedGCNReactionNetworkLightning
    assert type(trainer_restart) == pl.Trainer


def test_transfer_learning():
    precision = 16
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=False,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    dataset_transfer = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG",
        classifier=False,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )
    train_loader_transfer = DataLoaderPrecomputedReactionGraphs(
        dataset_transfer, batch_size=128, shuffle=False
    )

    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    model = load_model_lightning(config, device=device, load_dir="./test_checkpoints/")

    trainer_transfer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer_transfer.fit(model, train_loader_transfer, train_loader_transfer)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, train_loader)

    print("training transfer works!")


def test_augmentation():
    precision = 16
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=False,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )

    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    config["augment"] = True
    model = load_model_lightning(config, device=device, load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, train_loader)


def test_classifier():
    precision = 16
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=True,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )

    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    config["classifier"] = True
    model = load_model_lightning(config, device=device, load_dir="./test_checkpoints/")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, train_loader)
