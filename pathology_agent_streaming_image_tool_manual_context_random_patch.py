import base64, io, json, math, os, random, logging
from configparser import ConfigParser
from util import encode_image, crop_to_image_with_guides
from openslide import OpenSlide
from openai import OpenAI, AzureOpenAI
import shutil
import time
import h5py
import numpy as np

SYSTEM_PROMPT_MULTI_STEP = """You are a meticulous pathology agent with expertise in histopathological analysis. Your goal is to achieve the most comprehensive understanding possible of the tissue slide.

COORDINATE SYSTEM: The full slide resolution is {w0} × {h0} pixels. All coordinates must reference this full resolution. The position guides on each image show full-resolution pixel coordinates.

ANALYSIS STRATEGY: You have {steps_remaining} total steps available. Use up to {analysis_steps} steps for intensive slide examination using view_random_region. Reserve your final step for reporting.
{context_limitation}
EXAMINATION PHASE (Steps 1-{analysis_steps}):
- CRITICAL: Use as many steps as possible to thoroughly analyze the slide
- Call view_random_region repeatedly to examine multiple tissue regions
- Each call will show you a random region of the slide (you cannot choose the location or size)
- Look for tissue architecture patterns, cellular morphology details, and any pathological changes
- Continue examining until you have achieved comprehensive understanding OR reached step {analysis_steps}
- Do NOT rush - use all available viewing opportunities

IMPORTANT: The view_random_region tool shows you RANDOM regions - you cannot control where you look. Use it extensively to build a comprehensive understanding across multiple random samples of the tissue.

REPORTING PHASE (Final Step):
Structure your final response with these exact XML tags:

<report>
Provide a detailed pathological report including:
- Overall tissue architecture and structural organization
- Cellular morphology and characteristics observed
- Any abnormalities, lesions, or pathological changes identified
- Description of specific regions examined and key findings
- Diagnostic reasoning based on your observations
</report>

<response>
Provide your specific answer to the user's question based on your comprehensive analysis.
</response>

REMEMBER: Quality over speed - use all available analysis steps to view as many random regions as possible before providing your final report and answer.
"""

SYSTEM_PROMPT_SINGLE_STEP = """You are a meticulous pathology agent with expertise in histopathological analysis. Your goal is to achieve the most comprehensive understanding possible of the tissue slide.

Structure your final response with these exact XML tags:

<report>
Provide a detailed pathological report including:
- Overall tissue architecture and structural organization
- Cellular morphology and characteristics observed
- Any abnormalities, lesions, or pathological changes identified
- Description of specific regions examined and key findings
- Your diagnostic reasoning based on your observations
</report>

<response>
Provide your specific answer to the user's question based on your comprehensive analysis.
</response>
"""

class PathologyAgentStreamingRandomPatch:
    def __init__(self, wsi_file, segmentation_file, session_id, max_iterations=15, model="gpt-5", 
                 target_size_long_size=1000, random_seed=None):
        """
        Initialize the random patch pathology agent for ablation studies.
        
        Args:
            wsi_file: Path to whole slide image
            segmentation_file: Path to CLAM h5 segmentation file to determine viable tissue area
            session_id: Unique identifier for this session
            max_iterations: Number of random boxes to show (default: 15)
            model: Model identifier (default: "gpt-5")
            target_size_long_size: Maximum resolution for image long side
            random_seed: Random seed for reproducibility (None = use system random)
        """
        logging.info(f"[RandomPatchAgent] Initializing for session {session_id}")
        init_start = time.time()
        
        # Config
        logging.debug(f"[RandomPatchAgent] Loading config...")
        config = ConfigParser()
        config.read("config.ini")
        api_key = config.get("main", "AZURE_OPENAI_API_KEY")
        endpoint = config.get("main", "AZURE_OPENAI_ENDPOINT")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2025-03-01-preview",
            azure_endpoint=endpoint
        )

        self.MODEL_ID = model
        self.MAX_ITERS = max_iterations
        self.target_size_long_size = target_size_long_size
        
        # Session and file setup
        self.session_id = session_id
        self.wsi_file = wsi_file
        self.segmentation_file = segmentation_file
        
        logging.info(f"[RandomPatchAgent] Opening WSI: {wsi_file}")
        self.slide = OpenSlide(wsi_file)
        w, h = self.slide.dimensions
        logging.info(f"[RandomPatchAgent] WSI dimensions: {w} x {h}")
        
        self.debug_dir = f"debug/{session_id}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load tissue bounding box from segmentation file
        logging.info(f"[RandomPatchAgent] Computing tissue bounding box from {segmentation_file}...")
        bbox_start = time.time()
        self.tissue_bbox = self._compute_tissue_bbox()
        bbox_time = time.time() - bbox_start
        logging.info(f"[RandomPatchAgent] Tissue bbox computed in {bbox_time:.2f}s: "
                    f"x=[{self.tissue_bbox['x_min']},{self.tissue_bbox['x_max']}], "
                    f"y=[{self.tissue_bbox['y_min']},{self.tissue_bbox['y_max']}], "
                    f"size={self.tissue_bbox['width']}x{self.tissue_bbox['height']}")
        
        # Fixed box size constraints
        # Min: 1000 pixels, Max: size of the tissue bounding box (larger dimension)
        self.min_box_size = 1000
        self.max_box_size = max(self.tissue_bbox['width'], self.tissue_bbox['height'])
        logging.info(f"[RandomPatchAgent] Box size range: [{self.min_box_size}, {self.max_box_size}]")
        
        # Generate random boxes within tissue area
        if random_seed is not None:
            random.seed(random_seed)
        logging.info(f"[RandomPatchAgent] Generating {max_iterations} random boxes...")
        boxes_start = time.time()
        self.selected_boxes = self._generate_random_boxes()
        boxes_time = time.time() - boxes_start
        logging.info(f"[RandomPatchAgent] Generated {len(self.selected_boxes)} boxes in {boxes_time:.2f}s")
        self.current_box_index = 0  # Track which box to show next
        
        # Conversation state
        self.conversation = []
        self.iteration_count = 0
        self.raw_responses = []
        
        # Tool schema - single tool that returns next random region
        self.tools = [{
            "type": "function",
            "name": "view_random_region",
            "description": "View a random region of the tissue slide. You cannot control the location or size - each call shows a different random region from viable tissue areas. Use this tool repeatedly to examine multiple regions and build comprehensive understanding.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }]

        # Initialize conversation
        logging.debug(f"[RandomPatchAgent] Initializing conversation...")
        if not self.conversation:
            self._initialize_conversation()
        
        init_time = time.time() - init_start
        logging.info(f"[RandomPatchAgent] Initialization complete in {init_time:.2f}s")
    
    def _compute_tissue_bbox(self):
        """Compute overall bounding box of viable tissue from CLAM coordinates and create tissue mask."""
        try:
            with h5py.File(self.segmentation_file, 'r') as f:
                # CLAM stores coordinates in 'coords' dataset
                coords = f['coords'][:]
                
                # Handle different coordinate formats
                # Coords can be (N, 2) or (1, N, 2) - squeeze to get (N, 2)
                if coords.ndim == 3:
                    coords = coords.squeeze(0)
                
                # Store coordinates for tissue coverage checking
                self.tissue_patch_coords = coords
                
                # CLAM patch size at level 0 is 224x224
                self.patch_size = 224
                
                # Compute bounding box
                x_min = int(np.min(coords[:, 0]))
                y_min = int(np.min(coords[:, 1]))
                x_max = int(np.max(coords[:, 0])) + self.patch_size
                y_max = int(np.max(coords[:, 1])) + self.patch_size
                
                # Get slide dimensions to ensure we don't exceed bounds
                w0, h0 = self.slide.dimensions
                
                bbox = {
                    'x_min': max(0, x_min),
                    'y_min': max(0, y_min),
                    'x_max': min(w0, x_max),
                    'y_max': min(h0, y_max),
                    'width': x_max - x_min,
                    'height': y_max - y_min
                }
                
                # Create PATCH-LEVEL binary tissue mask for integral image (much more memory efficient!)
                
                # Compute patch grid dimensions
                patch_grid_w = (bbox['width'] + self.patch_size - 1) // self.patch_size
                patch_grid_h = (bbox['height'] + self.patch_size - 1) // self.patch_size
                
                # Create patch-level mask (1 = tissue patch, 0 = no tissue)
                patch_mask = np.zeros((patch_grid_h, patch_grid_w), dtype=np.float32)
                
                for coord in coords:
                    px, py = int(coord[0]), int(coord[1])
                    # Convert to patch coordinates (relative to bounding box)
                    mx = (px - bbox['x_min']) // self.patch_size
                    my = (py - bbox['y_min']) // self.patch_size
                    # Set this patch cell to 1.0 (tissue present)
                    if 0 <= my < patch_grid_h and 0 <= mx < patch_grid_w:
                        patch_mask[my, mx] = 1.0
                
                self.patch_mask = patch_mask
                self.patch_grid_w = patch_grid_w
                self.patch_grid_h = patch_grid_h
                
                # Precompute integral image for fast queries
                self.tissue_integral = self._integral_image(patch_mask)
                
                return bbox
        except Exception as e:
            raise ValueError(f"Failed to load segmentation file {self.segmentation_file}: {e}")
    
    def _integral_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute integral image (summed-area table) from binary mask.
        
        Args:
            mask: HxW binary/float array where 1.0 = tissue, 0.0 = background
            
        Returns:
            (H+1)x(W+1) integral image with zero-padding at [0,:] and [:,0]
        """
        I = mask.cumsum(axis=0).cumsum(axis=1)
        # Pad a top row and left col of zeros so rectangle sums are simple
        return np.pad(I, ((1, 0), (1, 0)), mode='constant')
    
    def _rect_sum(self, I: np.ndarray, x: int, y: int, w: int, h: int = None) -> float:
        """
        Compute sum in a rectangular region using integral image.
        
        Args:
            I: Integral image
            x, y: Top-left corner (in mask coordinates)
            w: Width
            h: Height (if None, uses w for a square)
            
        Returns:
            Sum of values in the region
        """
        if h is None:
            h = w
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        # Integral image indices are +1 shifted due to padding
        return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    
    def _all_square_sums(self, I: np.ndarray, s: int) -> np.ndarray:
        """
        Vectorized computation of sums for all s×s squares in the image.
        
        Args:
            I: Integral image (H+1 × W+1)
            s: Square side length
            
        Returns:
            (H - s + 1) × (W - s + 1) array of sums for each possible position
        """
        H = I.shape[0] - 1
        W = I.shape[1] - 1
        
        # Use integral image trick vectorized over the whole grid
        A = I[s:, s:] - I[:-s, s:] - I[s:, :-s] + I[:-s, :-s]
        return A
    
    def _conservative_size_upper_bound(self, min_frac: float = 0.25) -> int:
        """
        Compute a guaranteed upper bound for any square box that can have ≥min_frac tissue.
        Based on global tissue area and image size.
        
        Args:
            min_frac: Minimum fraction of tissue required
            
        Returns:
            Maximum feasible square size (in pixels)
        """
        # Total tissue pixels available
        tissue_patches = float(self.patch_mask.sum())
        T = tissue_patches * (self.patch_size * self.patch_size)
        
        # Image dimensions in pixels
        H = self.tissue_bbox['height']
        W = self.tissue_bbox['width']
        
        # If an s×s square contains at most min(T, s²) tissue, needing ≥min_frac*s²
        # implies s² ≤ T / min_frac. Also cannot exceed image dimensions.
        return int(min(min(H, W), np.floor(np.sqrt(T / min_frac))))
    
    def _feasible_positions_for_size(self, s: int, min_frac: float = 0.25) -> np.ndarray:
        """
        Compute boolean grid of valid top-left positions (at patch boundaries) for a given square size.
        
        Args:
            s: Square side length (in pixels)
            min_frac: Minimum fraction of tissue required
            
        Returns:
            Boolean array indicating valid patch-aligned positions
        """
        # Convert box size from pixels to patches (round up to be conservative)
        s_patches = (s + self.patch_size - 1) // self.patch_size
        
        # Get all patch-level sums for this size
        if s_patches > self.patch_grid_h or s_patches > self.patch_grid_w:
            # Box too large, return empty array
            return np.zeros((1, 1), dtype=bool)
        
        sums = self._all_square_sums(self.tissue_integral, s_patches)
        
        # Each patch in the sum represents patch_size × patch_size tissue pixels
        tissue_pixels_per_patch = self.patch_size * self.patch_size
        tissue_pixels = sums * tissue_pixels_per_patch
        
        # Box area in pixels
        box_area = s * s
        
        # Check which positions meet the minimum fraction requirement
        accept = tissue_pixels >= (min_frac * box_area)
        
        return accept
    
    def _calculate_tissue_coverage(self, x0, y0, box_size):
        """
        Calculate the percentage of a box covered by tissue using patch-level integral image (O(1) query).
        
        Args:
            x0, y0: Top-left corner of the box (in full slide coordinates, pixels)
            box_size: Size of the square box (pixels)
            
        Returns:
            float: Percentage of box area covered by tissue (0.0 to 1.0)
        """
        # Convert to patch coordinates (relative to bounding box)
        # We need to find which patches overlap with this box
        mx_pixel = x0 - self.tissue_bbox['x_min']
        my_pixel = y0 - self.tissue_bbox['y_min']
        
        # Convert to patch grid coordinates
        patch_x_start = mx_pixel // self.patch_size
        patch_y_start = my_pixel // self.patch_size
        patch_x_end = (mx_pixel + box_size + self.patch_size - 1) // self.patch_size
        patch_y_end = (my_pixel + box_size + self.patch_size - 1) // self.patch_size
        
        # Clamp to grid bounds
        patch_x_start = max(0, patch_x_start)
        patch_y_start = max(0, patch_y_start)
        patch_x_end = min(self.patch_grid_w, patch_x_end)
        patch_y_end = min(self.patch_grid_h, patch_y_end)
        
        # Query integral image for number of tissue patches in this region
        patch_w = patch_x_end - patch_x_start
        patch_h = patch_y_end - patch_y_start
        
        if patch_w <= 0 or patch_h <= 0:
            return 0.0
        
        tissue_patches = self._rect_sum(self.tissue_integral, patch_x_start, patch_y_start, patch_w, patch_h)
        
        # Convert patch count to approximate pixel count
        # Each patch represents patch_size × patch_size pixels of tissue
        tissue_pixels = tissue_patches * (self.patch_size * self.patch_size)
        
        # Calculate coverage
        box_area = box_size * box_size
        coverage = tissue_pixels / box_area if box_area > 0 else 0
        
        return coverage
    
    def _generate_random_boxes(self):
        """
        Generate random-sized square boxes with GUARANTEED 25% tissue coverage using integral image.
        
        This uses an efficient O(1) integral image approach to:
        1. Pick a random square size
        2. Find all positions with ≥25% tissue coverage
        3. Sample k boxes from valid positions
        
        This guarantees 25% coverage and is much faster than trial-and-error sampling.
        """
        boxes = []
        min_coverage = 0.25
        rng = np.random.default_rng()
        
        # Compute upper bound for feasible box sizes (in pixels)
        logging.debug(f"[RandomPatchAgent] Computing size upper bound for {min_coverage*100}% coverage...")
        s_ub = self._conservative_size_upper_bound(min_coverage)
        logging.debug(f"[RandomPatchAgent] Size upper bound: {s_ub}")
        
        if s_ub < self.min_box_size:
            raise ValueError(f"No box size can meet {min_coverage*100}% tissue requirement. "
                           f"Upper bound {s_ub} < minimum size {self.min_box_size}")
        
        # For each box, choose a size and sample from valid positions
        max_size_attempts = 100
        
        for i in range(self.MAX_ITERS):
            # Try to find a size with valid positions
            size_found = False
            chosen_s = None
            accept = None
            
            logging.debug(f"[RandomPatchAgent] Generating box {i+1}/{self.MAX_ITERS}...")
            
            for attempt in range(max_size_attempts):
                # Randomly sample size in feasible range
                candidate_s = int(rng.integers(self.min_box_size, s_ub + 1))
                
                # Ensure box fits within tissue bounding box
                if candidate_s > self.tissue_bbox['height'] or candidate_s > self.tissue_bbox['width']:
                    continue
                
                # Get all valid positions for this size (returns patch-grid coordinates)
                accept = self._feasible_positions_for_size(candidate_s, min_coverage)
                
                if accept.any():
                    chosen_s = candidate_s
                    size_found = True
                    logging.debug(f"[RandomPatchAgent]   Box {i+1}: size={chosen_s} (attempt {attempt+1})")
                    break
            
            if not size_found:
                # Fallback: try a few specific sizes
                fallback_sizes = [
                    self.min_box_size,
                    max(self.min_box_size, s_ub // 2),
                    max(self.min_box_size, s_ub // 4),
                    s_ub
                ]
                
                for candidate_s in fallback_sizes:
                    if candidate_s > self.tissue_bbox['height'] or candidate_s > self.tissue_bbox['width']:
                        continue
                    
                    accept = self._feasible_positions_for_size(candidate_s, min_coverage)
                    
                    if accept.any():
                        chosen_s = candidate_s
                        size_found = True
                        break
            
            if not size_found:
                raise ValueError(f"Could not find any feasible box size with ≥{min_coverage*100}% tissue coverage")
            
            # Sample one random position from valid positions (in patch-grid coordinates)
            ys, xs = np.nonzero(accept)
            n_valid = len(xs)
            
            if n_valid == 0:
                raise RuntimeError("Logic error: empty feasible set after acceptance check")
            
            # Pick random valid position (in patch-grid coordinates)
            idx = rng.integers(0, n_valid)
            patch_x, patch_y = int(xs[idx]), int(ys[idx])
            
            # Convert from patch-grid coordinates to pixel coordinates (relative to bbox)
            mx_pixel = patch_x * self.patch_size
            my_pixel = patch_y * self.patch_size
            
            # Convert to full slide coordinates
            x0 = mx_pixel + self.tissue_bbox['x_min']
            y0 = my_pixel + self.tissue_bbox['y_min']
            
            # Ensure box doesn't exceed tissue bounding box boundaries
            # This can happen due to patch-grid to pixel-level conversion
            x0 = min(x0, self.tissue_bbox['x_max'] - chosen_s)
            y0 = min(y0, self.tissue_bbox['y_max'] - chosen_s)
            x0 = max(x0, self.tissue_bbox['x_min'])
            y0 = max(y0, self.tissue_bbox['y_min'])
            
            # Verify coverage (should be guaranteed by construction)
            coverage = self._calculate_tissue_coverage(x0, y0, chosen_s)
            
            boxes.append({
                'x0': x0,
                'y0': y0,
                'size': chosen_s,
                'tissue_coverage': coverage,
                'valid_positions': n_valid
            })
        
        # Verify all boxes meet the requirement
        for idx, box in enumerate(boxes):
            assert box['tissue_coverage'] >= min_coverage, \
                f"Box {idx} has only {box['tissue_coverage']*100:.1f}% coverage (< {min_coverage*100}%)"
            assert box['x0'] >= self.tissue_bbox['x_min'], f"Box {idx} x0 out of bounds"
            assert box['y0'] >= self.tissue_bbox['y_min'], f"Box {idx} y0 out of bounds"
            assert box['x0'] + box['size'] <= self.tissue_bbox['x_max'], f"Box {idx} right edge out of bounds"
            assert box['y0'] + box['size'] <= self.tissue_bbox['y_max'], f"Box {idx} bottom edge out of bounds"
        
        return boxes

    def get_slide_dimensions(self):
        """Get slide dimensions"""
        w0, h0 = self.slide.dimensions
        return {"width": w0, "height": h0}

    def _initialize_conversation(self):
        """Initialize conversation with system message"""
        w0, h0 = self.slide.dimensions
        
        # Choose appropriate prompt based on max_iterations
        if self.MAX_ITERS == 1:
            # Single step - no tool use, just analyze overview
            prompt_content = SYSTEM_PROMPT_SINGLE_STEP.format(w0=w0, h0=h0)
        else:
            # Multi-step - use tool for detailed analysis
            context_limitation = ""
            
            prompt_content = SYSTEM_PROMPT_MULTI_STEP.format(
                w0=w0, 
                h0=h0, 
                steps_remaining=self.MAX_ITERS,
                analysis_steps=self.MAX_ITERS - 1,
                context_limitation=context_limitation
            )
        
        self.conversation.append({
            "role": "system",
            "type": "message",
            "content": prompt_content
        })

    def _print_token_usage(self, response):
        """Print token usage from OpenAI response object"""
        try:
            usage = response.usage
            
            if usage:
                # Extract token counts
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                reasoning_tokens = usage.output_tokens_details.reasoning_tokens

                print(f"\n{'='*60}")
                print(f"TOKEN USAGE")
                print(f"{'='*60}")
                print(f"Input tokens:     {input_tokens:,}")
                print(f"Output tokens: {output_tokens:,}")
                print(f"Reasoning tokens (part of output):      {reasoning_tokens:,}")
                print(f"{'='*60}\n")
            else:
                print(f"No usage data found in response")
                
        except Exception as e:
            print(f"Error extracting token usage: {e}")

    # Wrapper of crop_image_to_guides that also saves a local version for the web app
    def crop_to_data_url(self, x0, y0, w, h, step, save=False):
        region_with_guides = crop_to_image_with_guides(self.slide, x0, y0, w, h, target_max=self.target_size_long_size)

        buf = io.BytesIO()
        region_with_guides.save(buf, format="JPEG")
        buf.seek(0)

        static_fn = None
        if save:
            fn = f"{self.debug_dir}/step_{step}.jpg"
            region_with_guides.save(fn, "JPEG")

            # Copy to static for web access
            static_dir = f"static/{self.session_id}"
            os.makedirs(static_dir, exist_ok=True)
            static_fn = f"{static_dir}/step_{step}.jpg"
            shutil.copy2(fn, static_fn)

        return f"data:image/jpeg;base64,{base64.b64encode(buf.read()).decode()}", static_fn

    def analyze_streaming(self, user_message, store_intermediate_images=False):
        """
        Stream analysis updates in real-time with iterative random region viewing.
        Model calls view_random_region tool to see random regions (no control over location/size).
        """
        try:
            logging.info(f"[RandomPatchAgent] Starting analysis streaming with max_iters={self.MAX_ITERS}")
            
            # Add overview image
            w0, h0 = self.slide.dimensions
            logging.debug(f"[RandomPatchAgent] Creating overview image...")
            overview_url, overview_static_path = self.crop_to_data_url(0, 0, w0, h0, step=0, save=store_intermediate_images)

            # Add user message with overview to conversation
            self.conversation.append({
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": user_message},
                    {"type": "input_image", "image_url": overview_url},
                ]
            })

            # Yield overview
            logging.info(f"[RandomPatchAgent] Yielding overview, starting iteration loop")
            yield {
                "type": "iteration",
                "step": 0,
                "iteration_type": "overview",
                "image_url": f"/{overview_static_path}",
                "coordinates": {"x0": 0, "y0": 0, "w": w0, "h": h0},
                "message": f"Starting analysis of slide ({w0} × {h0} pixels)",
                "tokens": {"prompt": 0, "completion": 0},
                "cost": 0.0
            }

            # Analysis loop
            for step in range(1, self.MAX_ITERS + 1):
                logging.info(f"[RandomPatchAgent] === Iteration {step}/{self.MAX_ITERS} ===")
                iter_start = time.time()
                self.iteration_count += 1
                
                # Determine if tools should be available (only if max_iterations > 1)
                tools_to_use = self.tools if self.MAX_ITERS > 1 else None
                
                # Send request to OpenAI using responses API with full conversation history
                logging.info(f"[RandomPatchAgent] Sending API request (model={self.MODEL_ID})...")
                api_start = time.time()
                response = self.client.responses.create(
                    input=self.conversation,
                    model=self.MODEL_ID,
                    tools=tools_to_use,
                    reasoning={"effort": "high", "summary": "detailed"}
                )
                api_time = time.time() - api_start
                logging.info(f"[RandomPatchAgent] API response received in {api_time:.2f}s")

                # Store raw response dictionary for this iteration
                response_dict = response.to_dict()
                self.raw_responses.append({
                    "iteration": step,
                    "response": response_dict
                })

                # Append all response output items to conversation (includes reasoning, messages, function calls)
                for output_item in response.output:
                    self.conversation.append(output_item.to_dict())

                function_calls = [rx.to_dict() for rx in response.output if rx.type == 'function_call']
                messages = [rx.to_dict() for rx in response.output if rx.type == 'message']

                logging.info(f"[RandomPatchAgent] Response contains {len(function_calls)} function calls, {len(messages)} messages")

                # Extract reasoning text if available
                reasoning_text = ""
                try:
                    # Look for reasoning in the output array
                    if hasattr(response, 'output') and response.output:
                        for output_item in response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'reasoning':
                                if hasattr(output_item, 'summary') and output_item.summary:
                                    reasoning_parts = []
                                    for summary_item in output_item.summary:
                                        if hasattr(summary_item, 'text'):
                                            reasoning_parts.append(summary_item.text)
                                    reasoning_text = ' '.join(reasoning_parts)
                                    logging.debug(f"[RandomPatchAgent] Extracted reasoning: {reasoning_text[:100]}...")
                                    break
                except Exception as e:
                    logging.warning(f"[RandomPatchAgent] Error extracting reasoning: {e}")
                    reasoning_text = ""
                
                # Handle function calls
                if len(function_calls) > 0:
                    logging.info(f"[RandomPatchAgent] Processing {len(function_calls)} function call(s)...")
                    # Process function calls and create outputs
                    for response_item in response.output:
                        if response_item.type == 'function_call':
                            try:
                                # Handle view_random_region tool - show next random box
                                if response_item.name == 'view_random_region':
                                    logging.info(f"[RandomPatchAgent] Tool call: view_random_region (box {self.current_box_index+1}/{len(self.selected_boxes)})")
                                    # Get next random box
                                    if self.current_box_index >= len(self.selected_boxes):
                                        # No more boxes available
                                        logging.warning(f"[RandomPatchAgent] No more random regions available!")
                                        function_output = {
                                            "type": "function_call_output",
                                            "call_id": response_item.call_id,
                                            "output": "No more regions available. Please provide your final analysis."
                                        }
                                        self.conversation.append(function_output)
                                        continue
                                    
                                    box = self.selected_boxes[self.current_box_index]
                                    self.current_box_index += 1
                                    
                                    x0 = box['x0']
                                    y0 = box['y0']
                                    size = box['size']
                                    
                                    logging.debug(f"[RandomPatchAgent] Cropping region at ({x0},{y0}) size={size}...")
                                    # Crop the region
                                    data_url, static_path = self.crop_to_data_url(x0, y0, size, size, step=step, save=store_intermediate_images)
                                    logging.debug(f"[RandomPatchAgent] Region cropped successfully")

                                    # Yield iteration update
                                    yield {
                                        "type": "iteration",
                                        "step": step,
                                        "iteration_type": "random_region",
                                        "image_url": f"/{static_path}",
                                        "coordinates": {"x0": x0, "y0": y0, "w": size, "h": size},
                                        "message": f"Random region at ({x0}, {y0}) - {size} × {size} pixels",
                                        "reasoning": reasoning_text,
                                        "tokens": {"prompt": 0, "completion": 0},
                                        "cost": 0.0
                                    }

                                    # Calculate remaining regions
                                    regions_remaining = len(self.selected_boxes) - self.current_box_index
                                    steps_remaining = self.MAX_ITERS - step - 1
                                    
                                    # Build reminder text
                                    reminder_text = (
                                        f"Random region at ({x0}, {y0}) with size {size}×{size} pixels. "
                                        f"Full slide resolution is {w0} × {h0} pixels. "
                                        f"You have {steps_remaining} remaining function calls and {regions_remaining} more random regions available to view."
                                    )
                                    
                                    # Create function output and append to conversation
                                    function_output = {
                                        "type": "function_call_output",
                                        "call_id": response_item.call_id,
                                        "output": [
                                            {
                                                "type": "input_text",
                                                "text": reminder_text
                                            },
                                            {
                                                "type": "input_image",
                                                "detail": "high",
                                                "image_url": data_url
                                            }
                                        ]
                                    }
                                    self.conversation.append(function_output)
                                
                            except Exception as e:
                                function_output = {
                                    "type": "function_call_output",
                                    "call_id": response_item.call_id,
                                    "output": f"Error: {str(e)}"
                                }
                                self.conversation.append(function_output)

                    iter_time = time.time() - iter_start
                    logging.info(f"[RandomPatchAgent] Iteration {step} completed in {iter_time:.2f}s")
                    continue

                # Handle final messages (no more function calls)
                if len(messages) > 0:
                    # Analysis complete
                    logging.info(f"[RandomPatchAgent] Received final message, analysis complete!")
                    final_content = response.output_text if hasattr(response, 'output_text') and response.output_text else 'Analysis completed.'
                    
                    yield {
                        "type": "final",
                        "analysis": final_content,
                        "reasoning": reasoning_text,
                        "metadata": {
                            "total_iterations": self.iteration_count,
                            "regions_viewed": self.current_box_index,
                            "total_regions_available": len(self.selected_boxes),
                            "total_tokens": 0,
                            "total_cost": 0.0,
                            "slide_file": self.wsi_file,
                            "segmentation_file": self.segmentation_file,
                            "tissue_bbox": self.tissue_bbox,
                            "selected_boxes": self.selected_boxes[:self.current_box_index]
                        },
                        "raw_responses": self.raw_responses,
                        "max_iterations_reached": False
                    }
                    return

                # If no function calls and no messages, something went wrong
                if len(function_calls) == 0 and len(messages) == 0:
                    logging.warning(f"[RandomPatchAgent] No function calls or messages in response!")
                    yield {
                        "type": "final",
                        "analysis": "Analysis completed. No further actions requested.",
                        "metadata": {
                            "total_iterations": self.iteration_count,
                            "regions_viewed": self.current_box_index,
                            "total_regions_available": len(self.selected_boxes),
                            "total_tokens": 0,
                            "total_cost": 0.0,
                            "slide_file": self.wsi_file
                        },
                        "raw_responses": self.raw_responses,
                        "max_iterations_reached": False
                    }
                    return
                
                iter_time = time.time() - iter_start
                logging.info(f"[RandomPatchAgent] Iteration {step} completed in {iter_time:.2f}s")

            # If we hit max iterations
            logging.info(f"[RandomPatchAgent] Reached maximum iterations ({self.MAX_ITERS})")
            yield {
                "type": "final",
                "analysis": "Analysis completed after maximum iterations.",
                "metadata": {
                    "total_iterations": self.iteration_count,
                    "regions_viewed": self.current_box_index,
                    "total_regions_available": len(self.selected_boxes),
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "slide_file": self.wsi_file,
                    "note": "Reached maximum iteration limit"
                },
                "raw_responses": self.raw_responses,
                "max_iterations_reached": True
            }

        except Exception as e:
            logging.error(f"[RandomPatchAgent] Analysis error: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Analysis error: {str(e)}"
            } 