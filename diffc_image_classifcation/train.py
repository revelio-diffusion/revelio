import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from helpers.dataset import (
    HuggingFaceImageDataset,
    load_huggingface_dataset,
)
from helpers.helpers import clean_model_name
from helpers.collect_results import ResultsCollector
from helpers.helpers import get_sample_feature, set_random_seed
from ezcolorlog import root_logger as logger
import argparse
import wandb
from constants import (
    model_base_dict,
    diffusion_transformers_val,
    diffusion_transformers_train,
    clip_transforms,
)
from models import ImageClassifer
from feature_models.clip_infer import CLIPPromptSelector
from helpers.prompt_dict import prompt_dict


def training_step(
    model, train_loader, optimizer, criterion, args, prompt_selector, config
):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (diffusion_images, clip_images, labels, _) in enumerate(
        tqdm(train_loader)
    ):
        diffusion_images, clip_images, labels = (
            diffusion_images.to(device),
            clip_images.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        # Forward pass
        if args.prompt_type == "from_CLIP":
            best_prompts, _ = prompt_selector.select_best_prompt_for_images(clip_images)
            model.diffusion_pipe.text_encoder = model.diffusion_pipe.text_encoder.to(
                device
            )
            best_prompt_embeds = model.diffusion_pipe.encode_prompt(
                best_prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )[0]
            features = model(
                diffusion_images, best_prompt_embeds, config["diffusion_timestep"]
            )
        elif args.prompt_type == "generic":
            prompts = ["A photo of a pet"] * len(diffusion_images)
            model.diffusion_pipe.text_encoder = model.diffusion_pipe.text_encoder.to(
                device
            )
            best_prompt_embeds = model.diffusion_pipe.encode_prompt(
                prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )[0]
            features = model(
                diffusion_images, best_prompt_embeds, config["diffusion_timestep"]
            )
        elif args.prompt_type == "empty":
            features = model(diffusion_images, None, config["diffusion_timestep"])
        else:
            raise ValueError(f"Invalid prompt type: {args.prompt_type}")

        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()

        # Update metrics locally
        _, predicted = torch.max(features, 1)
        correct_predictions = (predicted == labels).sum().item()
        train_correct += correct_predictions
        train_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)

    # Calculate average loss and accuracy for training
    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total

    return train_loss, train_accuracy


def validation_step(model, test_loader, criterion, args, prompt_selector, config):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Perform validation
    with torch.no_grad():
        for batch_idx, (diffusion_images, clip_images, labels, _) in enumerate(
            tqdm(test_loader)
        ):
            # Move inputs to correct device
            diffusion_images, clip_images, labels = (
                diffusion_images.to(device),
                clip_images.to(device),
                labels.to(device),
            )

            # Forward pass without gradients
            if args.prompt_type == "from_CLIP":
                best_prompts, _ = prompt_selector.select_best_prompt_for_images(
                    clip_images
                )
                model.diffusion_pipe.text_encoder = (
                    model.diffusion_pipe.text_encoder.to(device)
                )
                best_prompt_embeds = model.diffusion_pipe.encode_prompt(
                    best_prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )[0]
                features = model(
                    diffusion_images, best_prompt_embeds, config["diffusion_timestep"]
                )
            elif args.prompt_type == "generic":
                prompts = ["A photo of a pet"] * len(diffusion_images)
                model.diffusion_pipe.text_encoder = (
                    model.diffusion_pipe.text_encoder.to(device)
                )
                best_prompt_embeds = model.diffusion_pipe.encode_prompt(
                    prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )[0]
                features = model(
                    diffusion_images, best_prompt_embeds, config["diffusion_timestep"]
                )
            elif args.prompt_type == "empty":
                features = model(diffusion_images, None, config["diffusion_timestep"])
            else:
                raise ValueError(f"Invalid prompt type: {args.prompt_type}")

            loss = criterion(features, labels)

            # Update metrics locally
            _, predicted = torch.max(features, 1)
            correct_predictions = (predicted == labels).sum().item()
            test_correct += correct_predictions
            test_loss += loss.item() * labels.size(0)
            test_total += labels.size(0)

    # Calculate final metrics
    test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total

    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument(
        "--dataset_flag",
        type=str,
        required=True,
        help="Dataset name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Feature extraction model name",
    )
    parser.add_argument(
        "--diffusion_timestep",
        type=int,
        required=True,
        help="Diffusion timestep for feature extraction",
    )

    parser.add_argument(
        "--diffusion_layer",
        type=str,
        choices=["bottleneck:0", "up_ft:0", "up_ft:1", "up_ft:2"],
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=90,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["empty", "from_CLIP", "generic"],
        required=True,
        help="Prompt type",
    )
    parser.add_argument(
        "--pooling_strategy",
        type=str,
        choices=["GAP", "GMP", "flatten"],
        required=True,
        help="Pooling strategy",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of classes in the dataset",
    )

    args = parser.parse_args()

    logger.info("Setting things up...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # Create output directory
    cleaned_model_name = clean_model_name(args.model_name)
    cleaned_dataset_flag = clean_model_name(args.dataset_flag)
    run_name = f"{cleaned_dataset_flag}/{cleaned_model_name}/diffusion_step_{args.diffusion_timestep}/layer_{args.diffusion_layer}/prompt_{args.prompt_type}/pool_{args.pooling_strategy}/dropout_{args.dropout_rate}"
    output_dir = os.path.join(args.output_dir, run_name)
    args.output_dir = output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    logger.info("Setting up device...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info("")

    logger.info("Setting up random seed...")
    set_random_seed(args.seed)
    logger.info("")

    logger.info("Setting up prompt selector...")
    from helpers.class_labels_dict import class_labels_dict

    class_labels = class_labels_dict[args.dataset_flag]
    prompt_dict = prompt_dict[args.dataset_flag]
    prompt_selector = CLIPPromptSelector(class_labels, prompt_dict, device=device)
    if args.prompt_type == "from_CLIP":
        logger.info("")
        logger.info(f"Prompts: {prompt_selector.prompts}")
        logger.info("")

    logger.info("Loading dataset...")
    hf_train_dataset = load_huggingface_dataset(args.dataset_flag, split="train")
    hf_test_dataset = load_huggingface_dataset(args.dataset_flag, split="test")

    # # Uncomment the following lines if you want to limit the dataset size for testing
    # hf_train_dataset = hf_train_dataset.select(range(1000))
    # hf_test_dataset = hf_test_dataset.select(range(1000))

    train_dataset = HuggingFaceImageDataset(
        hf_train_dataset, diffusion_transformers_train, clip_transforms
    )
    test_dataset = HuggingFaceImageDataset(
        hf_test_dataset, diffusion_transformers_val, clip_transforms
    )

    # Prepare data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        pin_memory=False,
    )

    config = vars(args)
    config["num_devices"] = torch.cuda.device_count()
    config["feature_model"] = model_base_dict[args.model_name]
    config["diffusion_step_type"] = "onestep"
    config["device"] = device

    sample_feature = get_sample_feature(train_dataset[0][0], config)
    input_channels = sample_feature.shape[1]
    config["input_channels"] = input_channels
    config["num_classes"] = args.num_classes
    config["dropout_rate"] = args.dropout_rate

    logger.info("Sample feature shape: ", sample_feature.shape)

    logger.info("Setting up model...")
    model = ImageClassifer(config)
    model.to(device)
    logger.info("")

    config["num_model_params"] = sum(p.numel() for p in model.parameters())
    config["num_trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    config["num_frozen_params"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    logger.info(
        f"Number of model parameters: {config['num_model_params']}, Trainable: {config['num_trainable_params']}, Frozen: {config['num_frozen_params']}"
    )
    logger.info("")

    # Dump config to file
    args_path = os.path.join(args.output_dir, "args.yaml")
    with open(args_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Final Config: {config}")
    logger.info("")

    # Initialize WandB logging
    wandb_config = vars(args)
    wandb_config["batch_size"] = args.batch_size  # Since only one process
    logger.info("Setting up WandB...")
    wandb.init(
        project="image-classification",
        config=wandb_config,
        name=run_name,
        group=f"{cleaned_dataset_flag}/{cleaned_model_name}",
        tags=[
            cleaned_dataset_flag,
            cleaned_model_name,
            args.diffusion_layer,
            str(args.diffusion_timestep),
            args.prompt_type,
            args.pooling_strategy,
        ],
    )
    logger.info("WandB initialized")

    logger.info("Setting up Training...")

    results_collector = ResultsCollector()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )

    logger.info("Starting Training...")
    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = training_step(
            model, train_loader, optimizer, criterion, args, prompt_selector, config
        )
        test_loss, test_accuracy = validation_step(
            model, test_loader, criterion, args, prompt_selector, config
        )

        # Log final results
        logger.info(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
        logger.info(
            f"Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        scheduler.step()

        # break at 30th epoch
        if epoch == 29:
            break

    # Save results at the end of training
    results_collector.save_results(args.output_dir)
    # Create a Done file to signal completion
    done_path = os.path.join(args.output_dir, "DONE")
    with open(done_path, "w") as f:
        f.write("DONE")

    logger.info("Training complete!")
    logger.info("Exiting...")

    wandb.finish()
