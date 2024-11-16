import torch
import clip


class CLIPPromptSelector:
    def __init__(self, class_labels, prompt_template, device="cuda"):
        """
        Initializes the CLIPPromptSelector with precomputed text embeddings for given class labels.
        Args:
            class_labels (list): List of class labels for the prompts.
            device (str): Device to run CLIP on (e.g., "cuda" or "cpu").
        """
        self.device = device
        self.class_labels = class_labels

        # Load CLIP model and preprocess function
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Generate text prompts based on class labels
        # self.prompts = [f"A photo of a {label}" for label in self.class_labels.values()]
        self.prompts = [
            prompt_template.format(label) for label in self.class_labels.values()
        ]

        # Compute text embeddings for all prompts (amortized in init)
        self.prompt_embeds = self._compute_prompt_embeddings()

    def _compute_prompt_embeddings(self):
        """
        Compute and return the text embeddings for all class label prompts.

        Returns:
            prompt_embeds (torch.Tensor): Precomputed CLIP embeddings of the prompts.
        """
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in self.prompts]).to(
            self.device
        )
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        # Normalize the embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def select_best_prompt_for_images(self, batch_images):
        """
        Select the best prompt for each image in the batch using precomputed prompt embeddings.

        Args:
            batch_images (list of numpy arrays): Batch of images (preprocessed).

        Returns:
            best_prompts (list): Best prompt for each image in the batch.
            best_prompt_embeds (torch.Tensor): CLIP embeddings of the best prompts.
        """
        # Preprocess and encode images
        batch_images = batch_images.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch_images)

        # Normalize image embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between image features and precomputed prompt embeddings
        similarity = image_features @ self.prompt_embeds.T

        # Select the best prompt for each image
        best_prompt_indices = similarity.argmax(dim=1)
        best_prompts = [self.prompts[idx] for idx in best_prompt_indices]

        # Return the best prompts and corresponding prompt embeddings
        best_prompt_embeds = self.prompt_embeds[best_prompt_indices]
        return best_prompts, best_prompt_embeds


# # Example usage
# class_labels = ["dog", "cat", "car"]  # Example class labels
# batch_images = [...]  # Batch of preprocessed images (as a list of numpy arrays)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Initialize the CLIPPromptSelector
# clip_selector = CLIPPromptSelector(class_labels, device)

# # Select the best prompts for the batch of images
# best_prompts, best_prompt_embeds = clip_selector.select_best_prompt_for_images(batch_images)

# print(best_prompts)  # Output the best prompt for each image
