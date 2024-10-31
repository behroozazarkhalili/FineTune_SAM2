from typing import Dict, List, Tuple, Optional
import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMTrainer:
    """
    A class for training and fine-tuning Segment Anything Model (SAM) on custom datasets.
    
    This class provides functionality for:
    - Loading and preprocessing image segmentation datasets
    - Training SAM with customizable parameters
    - Performing inference with trained models
    - Visualizing results
    
    Attributes:
        data_dir (str): Base directory containing the dataset
        images_dir (str): Directory containing input images
        masks_dir (str): Directory containing segmentation masks
        device (str): Device to use for training ('cuda' or 'cpu')
        model_cfg (str): Path to SAM model configuration file
        checkpoint_path (str): Path to pretrained SAM checkpoint
        
    Example:
        >>> trainer = SAMTrainer(
        ...     data_dir="path/to/dataset",
        ...     model_cfg="sam2_config.yaml",
        ...     checkpoint_path="sam2_checkpoint.pt"
        ... )
        >>> trainer.prepare_data()
        >>> trainer.train(steps=1000)
    """
    
    def __init__(
        self,
        data_dir: str,
        model_cfg: str,
        checkpoint_path: str,
        device: str = "cuda"
    ) -> None:
        """
        Initialize the SAM trainer with necessary paths and configurations.
        
        Args:
            data_dir: Root directory containing the dataset
            model_cfg: Path to the SAM model configuration file
            checkpoint_path: Path to the pretrained model checkpoint
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        
        # Initialize model-related attributes
        self.sam_model = None
        self.predictor = None
        self.train_data = []
        self.test_data = []
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Prepare training and testing datasets by loading and splitting the data.
        
        Args:
            test_size: Proportion of dataset to include in the test split
            random_state: Random seed for reproducibility
        """
        # Load the training CSV file
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        
        # Split data into training and testing sets
        train_df, test_df = train_test_split(
            train_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Prepare training data
        self.train_data = [
            {
                "image": os.path.join(self.images_dir, row['ImageId']),
                "annotation": os.path.join(self.masks_dir, row['MaskId'])
            }
            for _, row in train_df.iterrows()
        ]
        
        # Prepare testing data
        self.test_data = [
            {
                "image": os.path.join(self.images_dir, row['ImageId']),
                "annotation": os.path.join(self.masks_dir, row['MaskId'])
            }
            for _, row in test_df.iterrows()
        ]
    
    def read_batch(
        self, 
        data: List[Dict[str, str]], 
        visualize: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Read and preprocess a random batch from the dataset.
        
        Args:
            data: List of dictionaries containing image and annotation paths
            visualize: If True, visualize the batch data
            
        Returns:
            Tuple containing:
                - Processed image (numpy array)
                - Binary mask (numpy array)
                - Points coordinates (numpy array)
                - Number of masks
        """
        # Select a random entry
        entry = data[np.random.randint(len(data))]
        
        # Read image and annotation
        image = cv2.imread(entry["image"])[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(entry["annotation"], cv2.IMREAD_GRAYSCALE)
        
        if image is None or ann_map is None:
            print(f"Error reading files: {entry['image']} or {entry['annotation']}")
            return None, None, None, 0
        
        # Calculate resize factor
        resize_factor = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
        
        # Resize image and annotation
        image = cv2.resize(
            image, 
            (int(image.shape[1] * resize_factor), 
             int(image.shape[0] * resize_factor))
        )
        ann_map = cv2.resize(
            ann_map,
            (int(ann_map.shape[1] * resize_factor),
             int(ann_map.shape[0] * resize_factor)),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Process masks and points
        binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
        points = []
        
        # Get unique indices (excluding background)
        indices = np.unique(ann_map)[1:]
        
        # Create binary mask
        for idx in indices:
            mask = (ann_map == idx).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, mask)
            
        # Erode mask to avoid boundary points
        eroded_mask = cv2.erode(
            binary_mask, 
            np.ones((5, 5), np.uint8), 
            iterations=1
        )
        
        # Get coordinates and sample points
        coords = np.argwhere(eroded_mask > 0)
        if len(coords) > 0:
            for _ in indices:
                yx = coords[np.random.randint(len(coords))]
                points.append([yx[1], yx[0]])
                
        points = np.array(points)
        
        # Visualize if requested
        if visualize:
            self._visualize_batch(image, binary_mask, points)
            
        # Prepare outputs
        binary_mask = np.expand_dims(binary_mask, axis=-1)
        binary_mask = binary_mask.transpose((2, 0, 1))
        points = np.expand_dims(points, axis=1)
        
        return image, binary_mask, points, len(indices)

    def _visualize_batch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray
    ) -> None:
        """
        Visualize a batch of data including original image, mask, and points.
        
        Args:
            image: Input image
            mask: Binary mask
            points: Array of point coordinates
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        
        # Plot binary mask
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        # Plot mask with points
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(mask, cmap='gray')
        
        # Add points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(
                point[0], 
                point[1], 
                c=colors[i % len(colors)], 
                s=100, 
                label=f'Point {i+1}'
            )
            
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def initialize_model(self) -> None:
        """
        Initialize the SAM model and predictor with the specified configuration.
        """
        self.sam_model = build_sam2(
            self.model_cfg,
            self.checkpoint_path,
            device=self.device
        )
        self.predictor = SAM2ImagePredictor(self.sam_model)
        
        # Set components to training mode
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)

    def train(
        self,
        steps: int = 2000,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        accumulation_steps: int = 4,
        scheduler_step_size: int = 500,
        scheduler_gamma: float = 0.2,
        checkpoint_interval: int = 500,
        model_name_prefix: str = "./models/fine_tuned_sam2"
    ) -> None:
        """
        Train the SAM model using the prepared dataset.
        
        Args:
            steps: Number of training steps
            learning_rate: Initial learning rate for optimizer
            weight_decay: Weight decay factor for optimizer
            accumulation_steps: Number of steps to accumulate gradients
            scheduler_step_size: Number of steps before learning rate adjustment
            scheduler_gamma: Factor to reduce learning rate by
            checkpoint_interval: Number of steps between saving checkpoints
            model_name_prefix: Prefix for saved model checkpoints
        """
        if not self.predictor:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        # Configure optimizer and scheduler
        optimizer = AdamW(
            params=self.predictor.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )
        scaler = GradScaler()

        mean_iou = 0

        for step in range(1, steps + 1):
            with autocast('cuda'):
                # Get batch data
                image, mask, input_point, num_masks = self.read_batch(
                    self.train_data,
                    visualize=False
                )
                
                # Skip invalid batches
                if (image is None or mask is None or num_masks == 0 or
                    not isinstance(input_point, np.ndarray)):
                    continue

                # Prepare input labels
                input_label = np.ones((num_masks, 1))
                if input_point.size == 0 or input_label.size == 0:
                    continue

                # Process image and prepare prompts
                self.predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                    input_point,
                    input_label,
                    box=None,
                    mask_logits=None,
                    normalize_coords=True
                )

                if (unnorm_coords is None or labels is None or
                    unnorm_coords.shape[0] == 0 or labels.shape[0] == 0):
                    continue

                # Generate embeddings
                sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels),
                    boxes=None,
                    masks=None
                )

                # Process features and generate masks
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in self.predictor._features["high_res_feats"]
                ]
                
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features
                )

                # Post-process masks and calculate losses
                prd_masks = self.predictor._transforms.postprocess_masks(
                    low_res_masks,
                    self.predictor._orig_hw[-1]
                )

                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                
                # Calculate segmentation loss
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) -
                          (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()

                # Calculate IoU and score loss
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                union = (gt_mask.sum(1).sum(1) +
                        (prd_mask > 0.5).sum(1).sum(1) - inter)
                iou = inter / union
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                
                # Combined loss with weighting
                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps

            # Gradient scaling and accumulation
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.predictor.model.parameters(),
                max_norm=1.0
            )

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                self.predictor.model.zero_grad()

            scheduler.step()

            # Save checkpoint at intervals
            # check if the directory exists
            if not os.path.exists(os.path.dirname(model_name_prefix)):
                os.makedirs(os.path.dirname(model_name_prefix))
                
            if step % checkpoint_interval == 0:
                checkpoint_name = f"{model_name_prefix}_{step}.torch"
                torch.save(
                    self.predictor.model.state_dict(),
                    checkpoint_name
                )

            # Update and log mean IoU
            if step == 1:
                mean_iou = float(iou.mean().cpu().detach())
            else:
                mean_iou = mean_iou * 0.99 + 0.01 * float(iou.mean().cpu().detach())

            if step % 100 == 0:
                print(f"Step {step}: Accuracy (IoU) = {mean_iou:.4f}")

    def inference(
        self,
        image_path: str,
        mask_path: str,
        num_points: int = 30,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform inference on a single image using the trained model.
        
        Args:
            image_path: Path to input image
            mask_path: Path to ground truth mask
            num_points: Number of points to sample per segment
            checkpoint_path: Path to fine-tuned model checkpoint
            
        Returns:
            Tuple containing:
                - Original image
                - Ground truth mask
                - Predicted segmentation map
        """
        # Load and preprocess image and mask
        image, mask = self._read_inference_image(image_path, mask_path)
        input_points = self._sample_points(mask, num_points)

        # Load fine-tuned weights if provided
        if checkpoint_path:
            self.predictor.model.load_state_dict(
                torch.load(checkpoint_path)
            )

        # Perform inference
        with torch.no_grad():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )

        # Process predictions
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0]
        sorted_masks = np_masks[np.argsort(np_scores)][::-1]

        # Create final segmentation map
        seg_map = self._create_segmentation_map(sorted_masks)

        return image, mask, seg_map

    def _read_inference_image(
        self,
        image_path: str,
        mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read and preprocess image and mask for inference.
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask
            
        Returns:
            Tuple of processed image and mask
        """
        img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask = cv2.imread(mask_path, 0)
        
        # Calculate resize factor
        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        
        # Resize image and mask
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        mask = cv2.resize(
            mask,
            (int(mask.shape[1] * r), int(mask.shape[0] * r)),
            interpolation=cv2.INTER_NEAREST
        )
        return img, mask

    def _sample_points(
        self,
        mask: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """
        Sample points inside the input mask.
        
        Args:
            mask: Input binary mask
            num_points: Number of points to sample
            
        Returns:
            Array of sampled point coordinates
        """
        points = []
        coords = np.argwhere(mask > 0)
        
        for _ in range(num_points):
            yx = coords[np.random.randint(len(coords))]
            points.append([[yx[1], yx[0]]])
            
        return np.array(points)

    def _create_segmentation_map(
        self,
        sorted_masks: np.ndarray
    ) -> np.ndarray:
        """
        Create a segmentation map from sorted predicted masks.
        
        Args:
            sorted_masks: Array of sorted binary masks
            
        Returns:
            Final segmentation map
        """
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

        for i, mask in enumerate(sorted_masks):
            if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                continue

            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = i + 1
            occupancy_mask[mask_bool] = True

        return seg_map