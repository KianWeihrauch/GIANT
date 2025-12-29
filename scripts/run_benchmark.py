import argparse
import ast
import json
import logging
import sys
import time
import uuid
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from openslide import OpenSlide
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Set, List
import pandas as pd
from tqdm import tqdm
import h5py
import base64
from io import BytesIO
from configparser import ConfigParser
from openai import AzureOpenAI, OpenAI
from anthropic import Anthropic
import numpy as np

# Add both current directory and parent directory to path for imports
# This allows the script to work when run from either root directory or scripts directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))  # Add parent directory (root)
sys.path.insert(0, str(current_dir))  # Add current directory

from pathology_agent_streaming_image_tool_manual_context import PathologyAgentStreaming
from pathology_agent_streaming_image_tool_manual_context_claude import PathologyAgentStreamingClaude
from pathology_agent_streaming_image_tool_manual_context_random_patch import PathologyAgentStreamingRandomPatch
from util import crop_to_image


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def log_error_and_exit(error_msg: str, file_name: str = None, log_dir: Path = None):
    """Log error with timestamp and exit the script."""
    timestamp = datetime.now(timezone.utc).strftime(ISO)
    log_entry = f"[{timestamp}] ERROR"
    if file_name:
        log_entry += f" (file: {file_name})"
    log_entry += f": {error_msg}\n"
    
    # Write to error log file
    if log_dir is None:
        log_dir = Path(".")
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = log_dir / "benchmark_errors.log"
    
    with error_log_path.open("a", encoding="utf-8") as f:
        f.write(log_entry)
    
    # Also print to console
    print(f"\n{'='*60}")
    print(f"FATAL ERROR - Script stopped")
    print(f"{'='*60}")
    print(log_entry)
    print(f"Error logged to: {error_log_path.absolute()}")
    print(f"{'='*60}\n")
    
    sys.exit(1)

GTEX_PROMPT = """What organ type is shown in this histopathology image? 

Select from the following options: 
{options}

Please choose the number of the correct option and respond in the following JSON format:
```json
{{"answer": YOUR_ANSWER}}
```
"""


TCGA_PROMPT = """
What is the primary diagnosis for this histopathology image?

Select from the following options:
{options}

Please choose the number of the correct option and respond in the following JSON format:
```json
{{"answer": YOUR_ANSWER}}
```
"""

PANDA_PROMPT = """You are being provided a sample of prostate tissue. Please analyze the tissue thoroughly and provide a Gleason score and ISUP grade.

At the end of your report, provide your Gleason score and ISUP grade using the following JSON format:
```json
{{
  "primary_pattern": PRIMARY_NUMBER,
  "secondary_pattern": SECONDARY_NUMBER,
  "total_score": TOTAL,
  "isup_grade": ISUP_GRADE_NUMBER
}}
```

Note: ISUP grades are defined as follows:
- Grade 1: Gleason ≤6
- Grade 2: Gleason 3+4=7
- Grade 3: Gleason 4+3=7
- Grade 4: Gleason 8
- Grade 5: Gleason 9-10
"""


TCGA_SLIDEBENCH_PROMPT = """{question}

Select from the following options:
{options}

Please respond with the OPTION NUMBER (1, 2, 3, or 4) of the correct answer in the following JSON format:
```json
{{"answer": OPTION_NUMBER}}
```

For example, if option 2 is correct, respond with {{"answer": 2}}
"""

TCGA_EXPERT_VQA_PROMPT = """{question}

Select from the following options:
{options}

Please respond with the OPTION NUMBER (1, 2, 3, or 4) of the correct answer in the following JSON format:
```json
{{"answer": OPTION_NUMBER}}
```

For example, if option 2 is correct, respond with {{"answer": 2}}
"""






# Benchmark configurations
BENCHMARK_CONFIGS = {
    "gtex": {
        "prompt": GTEX_PROMPT,
        "column_mapping": {
            "Tissue Sample ID": "file_id",
            "image_path": "file_name",  # Extract filename from path
            "options": "options",  # Column containing the options
        },
        "file_name_from_path": True,  # Extract filename from full path
    },
    "tcga": {
        "prompt": TCGA_PROMPT,
        "column_mapping": {
            "image": "file_id", 
            "image_path": "file_name",
            "options": "options"
        },
        "file_name_from_path": False,
    },
    "tcga_slidebench": {
        "prompt": TCGA_SLIDEBENCH_PROMPT,
        "column_mapping": {
            "ID": "file_id", 
            "image_path": "file_name",
            "options": "options",
            "Question": "question"
        },
        "file_name_from_path": False,
    },
    "panda": {
        "prompt": PANDA_PROMPT,
        "column_mapping": {
            "id": "file_id",
            "image_path": "file_name"
        },
        "file_name_from_path": False,
    },
    "tcga_expert_vqa": {
        "prompt": TCGA_EXPERT_VQA_PROMPT,
        "column_mapping": {
            "id": "file_id", 
            "image_path": "file_name",
            "options": "options",
            "question": "question"
        },
        "file_name_from_path": False,
    },

}


@dataclass
class AgentResult:
    # Inputs/keys
    file_id: Optional[str]
    file_name: str
    conv_id: str
    prompt: str
    benchmark: str

    # Outputs
    status: str                           # "ok" | "missing" | "error" | "timeout" | "max_retries_exceeded"
    prediction: Optional[str] = None      # LLM output/prediction
    error: Optional[str] = None
    raw_responses: Optional[list] = None  # Raw response dictionaries from each iteration

    # Metadata
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    start_time: str = ""
    end_time: str = ""
    duration_s: float = 0.0
    script_version: str = "benchmark_v2"
    retry_count: int = 0                  # Number of retries attempted
    mode: str = "agent"                   # "agent" | "patch" | "thumbnail" | "random_patch_agent"
    patch_responses: Optional[list] = None  # For patch mode: list of responses per patch
    
    # Patch-specific fields (populated in patch mode)
    patch_idx: Optional[int] = None
    patch_x: Optional[int] = None
    patch_y: Optional[int] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def image_to_data_url(image) -> str:
    """Convert PIL Image to data URL for Azure OpenAI."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"


def call_llm_with_image(client: AzureOpenAI, image, prompt: str, model: str, timeout_s: Optional[float]) -> Dict[str, Any]:
    """Call Azure OpenAI with a single image and prompt."""
    try:
        data_url = image_to_data_url(image)
        
        chat_input = [{
            "role": "user",
            "type": "message",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
        
        start = time.time()
        response = client.responses.create(
            input=chat_input,
            model=model,
            reasoning={"effort": "high", "summary": "detailed"}
        )
        
        elapsed = time.time() - start
        if timeout_s and elapsed > timeout_s:
            return {"status": "timeout", "error": f"Timed out after {timeout_s}s"}
        
        # Extract text from response
        text_content = ""
        for item in response.output:
            if item.type == 'message':
                # Get the text content from the message
                if hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            text_content += content_item.text
        
        return {
            "status": "ok",
            "prediction": text_content,
            "response": response.to_dict()
        }
    except Exception as e:
        return {"status": "error", "error": f"LLM call error: {e!r}"}


def call_openai_with_image(client: OpenAI, image, prompt: str, model: str, timeout_s: Optional[float]) -> Dict[str, Any]:
    """Call regular OpenAI API with a single image and prompt."""
    try:
        data_url = image_to_data_url(image)
        
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            max_tokens=4096
        )
        
        elapsed = time.time() - start
        if timeout_s and elapsed > timeout_s:
            return {"status": "timeout", "error": f"Timed out after {timeout_s}s"}
        
        # Extract text from response
        text_content = response.choices[0].message.content
        
        return {
            "status": "ok",
            "prediction": text_content,
            "response": response.model_dump()
        }
    except Exception as e:
        return {"status": "error", "error": f"LLM call error: {e!r}"}


def call_claude_with_image(client: Anthropic, image, prompt: str, model: str, timeout_s: Optional[float]) -> Dict[str, Any]:
    """Call Claude API with a single image and prompt using streaming."""
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        start = time.time()
        
        # Use streaming like the Claude agent does
        with client.messages.stream(
            model=model,
            max_tokens=30000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_data
                        }
                    }
                ]
            }],
            thinking={
                "type": "enabled",
                "budget_tokens": 1024
            }
        ) as stream:
            # Consume the stream to get the final response
            response = stream.get_final_message()
        
        elapsed = time.time() - start
        if timeout_s and elapsed > timeout_s:
            return {"status": "timeout", "error": f"Timed out after {timeout_s}s"}
        
        # Extract text from response
        text_content = ""
        for content_block in response.content:
            if content_block.type == "text":
                text_content += content_block.text
        
        return {
            "status": "ok",
            "prediction": text_content,
            "response": response.model_dump()
        }
    except Exception as e:
        return {"status": "error", "error": f"LLM call error: {e!r}"}


def stream_final(agent, prompt: str, timeout_s: Optional[float]) -> Dict[str, Any]:
    """
    Consume streaming updates until 'final' or 'error'.
    Works with both PathologyAgentStreaming, PathologyAgentStreamingClaude, and PathologyAgentStreamingRandomPatch.
    Returns {"status": "ok", "analysis": "...", "raw_responses": [...]} or {"status": "error"/"timeout", "error": "..."}.
    """
    start = time.time()
    last_heartbeat = start
    heartbeat_interval = 30.0  # Log heartbeat every 30 seconds
    iteration_count = 0
    
    try:
        it = agent.analyze_streaming(prompt)
        while True:
            current_time = time.time()
            
            # Heartbeat logging for long-running operations
            if current_time - last_heartbeat > heartbeat_interval:
                elapsed = current_time - start
                logging.info(f"    [HEARTBEAT] Still processing after {elapsed:.1f}s, iteration {iteration_count}")
                last_heartbeat = current_time
            
            if timeout_s and (current_time - start) > timeout_s:
                elapsed = current_time - start
                logging.warning(f"    [TIMEOUT] After {elapsed:.1f}s at iteration {iteration_count}")
                return {"status": "timeout", "error": f"Timed out after {timeout_s}s"}
            
            try:
                update = next(it)
            except StopIteration:
                # No final message produced
                logging.error(f"    [ERROR] Stream ended without final result after {iteration_count} iterations")
                return {"status": "error", "error": "Stream ended without final result"}
            except Exception as e:
                logging.error(f"    [ERROR] Stream error: {e!r}")
                return {"status": "error", "error": f"Stream error: {e!r}"}

            utype = update.get("type")
            
            # Track and log iteration updates
            if utype == "iteration":
                iteration_count = update.get("step", iteration_count + 1)
                iter_type = update.get("iteration_type", "unknown")
                message = update.get("message", "")
                logging.debug(f"    [ITER {iteration_count}] {iter_type}: {message}")
                last_heartbeat = current_time  # Reset heartbeat on progress
                
            elif utype == "note":
                observation = update.get("observation", "")[:100]
                logging.debug(f"    [NOTE] {observation}...")
                
            elif utype == "final":
                elapsed = current_time - start
                max_iter_flag = update.get("max_iterations_reached", False)
                status_msg = "MAX_ITERATIONS" if max_iter_flag else "COMPLETE"
                logging.info(f"    [FINAL] {status_msg} after {iteration_count} iterations in {elapsed:.1f}s")
                return {
                    "status": "ok", 
                    "prediction": update.get("analysis"),
                    "raw_responses": update.get("raw_responses", []),
                    "max_iterations_reached": max_iter_flag
                }
                
            elif utype == "error":
                error_msg = update.get("message") or "Unknown agent error"
                logging.error(f"    [ERROR] Agent error: {error_msg}")
                return {"status": "error", "error": error_msg}
                
    except Exception as e:
        logging.error(f"    [EXCEPTION] Agent exception: {e!r}", exc_info=True)
        return {"status": "error", "error": f"Agent exception: {e!r}"}


def read_processed_keys(output_path: Path, mode: str = "agent") -> Set:
    """For resume: read existing JSONL and collect keys already processed.
    
    In agent/thumbnail/random_patch_agent mode: returns Set[str] of file_ids (or file_names if file_id not available)
    In patch mode: returns Set[tuple] of (file_name, patch_idx) tuples
    """
    processed = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    
                    if mode == "patch":
                        # In patch mode, track (file_name, patch_idx) tuples
                        file_name = rec.get("file_name")
                        patch_idx = rec.get("patch_idx")
                        if file_name and patch_idx is not None:
                            processed.add((file_name, patch_idx))
                    else:
                        # In agent/thumbnail/random_patch_agent mode, use file_id as unique identifier
                        # This handles VQA benchmarks where multiple questions exist for the same image
                        file_id = rec.get("file_id")
                        if file_id:
                            processed.add(file_id)
                        else:
                            # Fallback to file_name if file_id not available (shouldn't happen in practice)
                            file_name = rec.get("file_name")
                            if file_name:
                                processed.add(file_name)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    return processed


def iter_rows(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    for _, row in df.iterrows():
        yield row.to_dict()


def expand_to_patches(df: pd.DataFrame, filedir: Path, patch_dir: Path, num_patches: Optional[int], processed_patches: Optional[Set[tuple]] = None) -> pd.DataFrame:
    """Expand dataset so each row represents a single patch instead of a slide.
    
    Args:
        df: DataFrame with WSI information
        filedir: Directory containing WSI files
        patch_dir: Directory containing patch h5 files
        num_patches: Number of patches to sample per WSI (None for all)
        processed_patches: Set of (file_name, patch_idx) tuples already processed (for resume)
    """
    expanded_rows = []
    processed_patches = processed_patches or set()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding to patches"):
        file_name = row.get("file_name")
        wsi_path = filedir / file_name
        
        if not wsi_path.exists():
            logging.warning(f"Slide not found: {wsi_path}")
            continue
        
        # Early check: if num_patches is specified and we already have enough processed patches, skip entirely
        if num_patches is not None:
            num_already_processed = sum(1 for fn, _ in processed_patches if fn == file_name)
            if num_already_processed >= num_patches:
                logging.info(f"WSI {file_name} already has {num_already_processed}/{num_patches} patches processed, skipping")
                continue
        
        # Get the slide name without extension
        slide_name = Path(file_name).stem
        patch_h5_path = patch_dir / f"{slide_name}.h5"
        
        if not patch_h5_path.exists():
            logging.warning(f"Patch file not found: {patch_h5_path}")
            continue
        
        try:
            with h5py.File(patch_h5_path, "r") as f:
                coords = f["coords"][:]  # (N, 2) -> usually (x, y) at level=0
            
            # Filter out already-processed patches
            total_patches = len(coords)
            available_indices = [i for i in range(total_patches) if (file_name, i) not in processed_patches]
            num_already_processed = total_patches - len(available_indices)
            
            if not available_indices:
                logging.info(f"All patches already processed for {file_name}, skipping")
                continue
            
            if num_already_processed > 0:
                logging.info(f"Skipping {num_already_processed} already-processed patches for {file_name}, {len(available_indices)} remaining")
            
            # Calculate how many patches we still need for this WSI
            if num_patches is not None:
                patches_still_needed = num_patches - num_already_processed
                patches_to_sample = min(patches_still_needed, len(available_indices))
            else:
                patches_to_sample = len(available_indices)
            
            # Sample random patches without replacement from available patches
            if patches_to_sample < len(available_indices):
                sampled_indices = np.random.choice(available_indices, size=patches_to_sample, replace=False)
                coords_to_process = [(i, coords[i]) for i in sampled_indices]
            else:
                coords_to_process = [(i, coords[i]) for i in available_indices]
            
            # Create a row for each patch
            for patch_idx, (x, y) in coords_to_process:
                patch_row = row.to_dict().copy()
                patch_row["patch_idx"] = int(patch_idx)  # Convert to Python int for consistency
                patch_row["patch_x"] = int(x)
                patch_row["patch_y"] = int(y)
                expanded_rows.append(patch_row)
                
        except Exception as e:
            logging.warning(f"Error processing patches for {file_name}: {e}")
            continue
    
    return pd.DataFrame(expanded_rows)


def run_one(row: Dict[str, Any], filedir: Path, prompt_template: str, benchmark: str, timeout_s: Optional[float], max_iterations: int = 15, max_retries: int = 3, model: str = "o3", max_images_in_context: Optional[int] = None, enabled_note_tool: bool = False, mode: str = "agent", log_dir: Path = None, image_resolution: Optional[int] = None, patch_dir: Optional[Path] = None) -> AgentResult:
    file_id = row.get("file_id")
    file_name = row.get("file_name")
    
    # Handle options if they exist (some benchmarks don't have options)
    prompt_kwargs = {}
    if "options" in row and row.get("options") is not None:
        options = ast.literal_eval(row.get("options"))
        # Structure the options as a numbered list
        option_text = ""
        for i, option in enumerate(options):
            option_text += f"{i+1}. {option}\n"
        prompt_kwargs["options"] = option_text
    
    # Handle question if it exists (for benchmarks with per-row questions)
    if "question" in row and row.get("question") is not None:
        prompt_kwargs["question"] = row.get("question")

    # Format the prompt using the template and row data
    prompt = prompt_template.format(**prompt_kwargs)

    start_ts = now_iso()
    t0 = time.time()
    wsi_path = filedir / file_name
    
    if not wsi_path.exists():
        res = AgentResult(
            file_id=file_id, file_name=file_name,
            conv_id="", prompt=prompt, benchmark=benchmark, status="missing",
            start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
            mode=mode
        )
        return res

    # Check that slide can be opened
    try:
        slide = OpenSlide(str(wsi_path))
    except Exception as e:
        print(e)
        res = AgentResult(
            file_id=file_id, file_name=file_name,
            conv_id="", prompt=prompt, benchmark=benchmark, status="invalid_image",
            start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
            mode=mode
        )
        return res

    conv_id = str(uuid.uuid4())
    
    # Mode-specific processing
    if mode == "agent":
        slide.close()
        # Retry loop for agent mode
        retry_count = 0
        all_raw_responses = []
        
        while retry_count <= max_retries:
            conv_id = str(uuid.uuid4())  # New conversation ID for each attempt
            
            # Map model name if needed (gpt-4o-non-azure -> gpt-4o for agent)
            # The agent will handle API selection internally, but we need to pass the right model name
            actual_model = "gpt-4o" if model == "gpt-4o-non-azure" else model
            
            # Determine which agent class to use based on model
            if model.startswith("claude"):
                # Use Claude agent
                # Default to 500 for Claude if resolution not specified
                target_resolution = image_resolution if image_resolution is not None else 500
                agent = PathologyAgentStreamingClaude(str(wsi_path), conv_id, max_iterations=max_iterations, model=actual_model, max_images_in_context=max_images_in_context, enable_note_tool=enabled_note_tool, target_size_long_size=target_resolution)
            else:
                # Use regular PathologyAgentStreaming (for GPT models, o-series, etc.)
                # Default to 1000 for OpenAI models if resolution not specified
                target_resolution = image_resolution if image_resolution is not None else 1000
                agent = PathologyAgentStreaming(str(wsi_path), conv_id, max_iterations=max_iterations, model=actual_model, max_images_in_context=max_images_in_context, enable_note_tool=enabled_note_tool, target_size_long_size=target_resolution)

            # Optional: try to record model metadata if available on your class
            model_name = getattr(agent, "MODEL_ID", model)  # Use MODEL_ID from agent, fallback to passed model
            model_version = getattr(agent, "model_version", None)

            out = stream_final(agent, prompt, timeout_s=timeout_s)

            status = out["status"]
            prediction = out.get("prediction")
            error = out.get("error")
            raw_responses = out.get("raw_responses", [])
            
            # Collect all raw responses across retries
            all_raw_responses.extend([{
                "retry": retry_count,
                "attempt_conv_id": conv_id,
                **resp
            } for resp in raw_responses])

            # If successful or non-retryable error, return result
            if status != "ok" or not out.get("max_iterations_reached", False):
                res = AgentResult(
                    file_id=file_id, file_name=file_name,
                    conv_id=conv_id, prompt=prompt, benchmark=benchmark, status=status, 
                    prediction=prediction, error=error, raw_responses=all_raw_responses,
                    model_name=model_name, model_version=model_version,
                    start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                    retry_count=retry_count, mode=mode
                )
                return res
            
            # If max iterations reached and we haven't exceeded max retries, retry
            retry_count += 1
            if retry_count <= max_retries:
                print(f"Max iterations reached for {file_name}, retrying ({retry_count}/{max_retries})")
        
        # If we've exhausted all retries
        res = AgentResult(
            file_id=file_id, file_name=file_name,
            conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="max_retries_exceeded", 
            prediction=prediction, error="Maximum retries exceeded due to max iterations being reached",
            raw_responses=all_raw_responses,
            model_name=model_name, model_version=model_version,
            start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
            retry_count=retry_count, mode=mode
        )
        return res
    
    elif mode == "random_patch_agent":
        slide.close()
        logging.info(f"Starting random_patch_agent mode for {file_name}")
        
        # Retry loop for random patch agent mode
        retry_count = 0
        all_raw_responses = []
        
        # Get the h5 segmentation file for this WSI
        if not patch_dir:
            logging.error(f"patch_dir not provided for random_patch_agent mode")
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                error="patch_dir is required for random_patch_agent mode",
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode
            )
            return res
        
        slide_name = Path(file_name).stem
        segmentation_file = patch_dir / f"{slide_name}.h5"
        
        logging.info(f"Looking for segmentation file: {segmentation_file}")
        if not segmentation_file.exists():
            logging.error(f"Segmentation file not found: {segmentation_file}")
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                error=f"Segmentation file not found: {segmentation_file}",
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode
            )
            return res
        
        while retry_count <= max_retries:
            conv_id = str(uuid.uuid4())  # New conversation ID for each attempt
            logging.info(f"Random patch agent attempt {retry_count+1}/{max_retries+1} (conv_id: {conv_id})")
            
            # Map model name if needed
            actual_model = "gpt-4o" if model == "gpt-4o-non-azure" else model
            
            # Default to 1000 for random patch agent if resolution not specified
            target_resolution = image_resolution if image_resolution is not None else 1000
            
            # Initialize random patch agent (no random_seed = random patches each time)
            logging.info(f"Initializing PathologyAgentStreamingRandomPatch (model={actual_model}, max_iter={max_iterations}, resolution={target_resolution})")
            init_start = time.time()
            try:
                agent = PathologyAgentStreamingRandomPatch(
                    str(wsi_path), 
                    str(segmentation_file), 
                    conv_id, 
                    max_iterations=max_iterations, 
                    model=actual_model,
                    target_size_long_size=target_resolution,
                    random_seed=None
                )
                init_time = time.time() - init_start
                logging.info(f"Agent initialized successfully in {init_time:.2f}s")
            except Exception as e:
                init_time = time.time() - init_start
                logging.error(f"Failed to initialize agent after {init_time:.2f}s: {e!r}", exc_info=True)
                res = AgentResult(
                    file_id=file_id, file_name=file_name,
                    conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                    error=f"Agent initialization failed: {e!r}",
                    start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                    mode=mode
                )
                return res

            # Optional: try to record model metadata if available on your class
            model_name = getattr(agent, "MODEL_ID", model)
            model_version = getattr(agent, "model_version", None)

            logging.info(f"Starting analysis streaming for {file_name}")
            stream_start = time.time()
            out = stream_final(agent, prompt, timeout_s=timeout_s)
            stream_time = time.time() - stream_start
            
            status = out["status"]
            prediction = out.get("prediction")
            error = out.get("error")
            raw_responses = out.get("raw_responses", [])
            
            logging.info(f"Analysis completed with status={status} in {stream_time:.2f}s")
            
            # Collect all raw responses across retries
            all_raw_responses.extend([{
                "retry": retry_count,
                "attempt_conv_id": conv_id,
                **resp
            } for resp in raw_responses])

            # If successful or non-retryable error, return result
            if status != "ok" or not out.get("max_iterations_reached", False):
                logging.info(f"Returning result: status={status}, retry_count={retry_count}")
                res = AgentResult(
                    file_id=file_id, file_name=file_name,
                    conv_id=conv_id, prompt=prompt, benchmark=benchmark, status=status, 
                    prediction=prediction, error=error, raw_responses=all_raw_responses,
                    model_name=model_name, model_version=model_version,
                    start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                    retry_count=retry_count, mode=mode
                )
                return res
            
            # If max iterations reached and we haven't exceeded max retries, retry
            retry_count += 1
            if retry_count <= max_retries:
                logging.warning(f"Max iterations reached for {file_name}, retrying ({retry_count}/{max_retries})")
                print(f"Max iterations reached for {file_name}, retrying ({retry_count}/{max_retries})")
        
        # If we've exhausted all retries
        res = AgentResult(
            file_id=file_id, file_name=file_name,
            conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="max_retries_exceeded", 
            prediction=prediction, error="Maximum retries exceeded due to max iterations being reached",
            raw_responses=all_raw_responses,
            model_name=model_name, model_version=model_version,
            start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
            retry_count=retry_count, mode=mode
        )
        return res
    
    elif mode == "patch":
        # Patch mode: process a single patch (coordinates are in the row)
        patch_idx = row.get("patch_idx")
        patch_x = row.get("patch_x")
        patch_y = row.get("patch_y")
        
        if patch_idx is None or patch_x is None or patch_y is None:
            slide.close()
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                error="Missing patch coordinates in row",
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode,
                patch_idx=patch_idx,
                patch_x=patch_x,
                patch_y=patch_y
            )
            return res
        
        config = ConfigParser()
        config.read("config.ini")
        
        # Determine which client to use based on model
        if model == "gpt-4o-non-azure":
            api_key = config.get("main", "OPENAI_KEY_TB")
            client = OpenAI(api_key=api_key)
            client_type = "openai"
            actual_model = "gpt-4o"  # Map to actual OpenAI model name
        elif model.startswith("claude"):
            api_key = config.get("main", "ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key)
            client_type = "claude"
            actual_model = model
        else:
            api_key = config.get("main", "AZURE_OPENAI_API_KEY")
            endpoint = config.get("main", "AZURE_OPENAI_ENDPOINT")
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2025-03-01-preview",
                azure_endpoint=endpoint
            )
            client_type = "azure"
            actual_model = model
        
        try:
            # Fixed parameters
            patch_size = 224
            read_level = 0
            
            # At level 0, no downsampling - read patch_size directly
            region = slide.read_region((patch_x, patch_y), read_level, (patch_size, patch_size)).convert("RGB")
            slide.close()
            
            if client_type == "azure":
                result = call_llm_with_image(client, region, prompt, actual_model, timeout_s)
            elif client_type == "openai":
                result = call_openai_with_image(client, region, prompt, actual_model, timeout_s)
            else:  # claude
                result = call_claude_with_image(client, region, prompt, actual_model, timeout_s)
            
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark,
                status=result["status"],
                prediction=result.get("prediction"),
                error=result.get("error"),
                model_name=model,
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode,
                patch_idx=patch_idx,
                patch_x=patch_x,
                patch_y=patch_y,
                patch_responses=[{
                    "patch_idx": patch_idx,
                    "coordinates": (patch_x, patch_y),
                    "status": result["status"],
                    "prediction": result.get("prediction"),
                    "error": result.get("error")
                }]
            )
            return res
            
        except Exception as e:
            slide.close()
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                error=f"Patch processing error: {e!r}",
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode,
                patch_idx=patch_idx if patch_idx is not None else None,
                patch_x=patch_x if patch_x is not None else None,
                patch_y=patch_y if patch_y is not None else None
            )
            return res
    
    elif mode == "thumbnail":
        # Thumbnail mode: get low-res thumbnail and query LLM
        config = ConfigParser()
        config.read("config.ini")
        
        # Determine which client to use based on model
        if model == "gpt-4o-non-azure":
            api_key = config.get("main", "OPENAI_KEY_TB")
            client = OpenAI(api_key=api_key)
            client_type = "openai"
            actual_model = "gpt-4o"  # Map to actual OpenAI model name
        elif model.startswith("claude"):
            api_key = config.get("main", "ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key)
            client_type = "claude"
            actual_model = model
        else:
            api_key = config.get("main", "AZURE_OPENAI_API_KEY")
            endpoint = config.get("main", "AZURE_OPENAI_ENDPOINT")
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2025-03-01-preview",
                azure_endpoint=endpoint
            )
            client_type = "azure"
            actual_model = model
        
        try:
            # Get thumbnail with configurable resolution (default 1024)
            thumbnail_size = image_resolution if image_resolution is not None else 1024
            
            # Use high-resolution crop function for resolutions above 1024
            if thumbnail_size > 1024:
                # Get whole slide dimensions
                slide_width, slide_height = slide.level_dimensions[0]
                # Use crop_to_image to get high-quality downsampled version
                thumbnail = crop_to_image(slide, 0, 0, slide_width, slide_height, target_max=thumbnail_size)
            else:
                # Use standard get_thumbnail for lower resolutions
                thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
            
            # # Save thumbnail for debugging
            # debug_path = f"{Path(file_name).stem}_{thumbnail_size}.jpg"
            # thumbnail.save(debug_path)
            # logging.info(f"Saved thumbnail to {debug_path}")
            
            slide.close()
            
            if client_type == "azure":
                result = call_llm_with_image(client, thumbnail, prompt, actual_model, timeout_s)
            elif client_type == "openai":
                result = call_openai_with_image(client, thumbnail, prompt, actual_model, timeout_s)
            else:  # claude
                result = call_claude_with_image(client, thumbnail, prompt, actual_model, timeout_s)
            
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark,
                status=result["status"],
                prediction=result.get("prediction"),
                error=result.get("error"),
                model_name=model,
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode
            )
            return res
            
        except Exception as e:
            slide.close()
            res = AgentResult(
                file_id=file_id, file_name=file_name,
                conv_id=conv_id, prompt=prompt, benchmark=benchmark, status="error",
                error=f"Thumbnail processing error: {e!r}",
                start_time=start_ts, end_time=now_iso(), duration_s=time.time() - t0,
                mode=mode
            )
            return res


@dataclass
class ExperimentRun:
    """Represents a single experiment run configuration."""
    benchmark: str
    dataset: str
    filedir: str
    model: str
    mode: str
    output_path: str
    
    # Agent-specific params
    max_iterations: int = 1
    max_images_in_context: Optional[int] = None
    enable_note_tool: bool = False
    image_resolution: Optional[int] = None
    
    # Patch-specific params
    patch_dir: Optional[str] = None
    num_patches: Optional[int] = None
    
    # General params
    timeout: float = 300.0
    max_retries: int = 3
    max_workers: int = 1
    limit: Optional[int] = None
    resume: bool = True
    log_dir: Optional[str] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_runs(config: Dict[str, Any]) -> List[ExperimentRun]:
    """Generate experiment runs from unified configuration.
    
    Iterates over all combinations of:
    - models
    - benchmarks
    - enabled modes (agent/patch/thumbnail)
    - iterations (for agent mode)
    - max_images_in_context (for agent mode)
    """
    runs = []
    models = config['models']
    modes_config = config['modes']
    
    output_dir = Path(config['output_dir'])
    log_dir = config.get('log_dir')
    
    for benchmark_cfg in config['benchmarks']:
        benchmark = benchmark_cfg['name']
        dataset = benchmark_cfg['dataset']
        filedir = benchmark_cfg['filedir']
        patch_dir = benchmark_cfg.get('patch_dir')
        
        for model in models:
            # Agent mode - iterate over all combinations of iterations, max_images_in_context, and image_resolutions
            if modes_config.get('agent', {}).get('enabled', False):
                agent_cfg = modes_config['agent']
                iterations_list = agent_cfg.get('iterations', [15])
                max_images_list = agent_cfg.get('max_images_in_context', [None])
                enable_note_tool = agent_cfg.get('enable_note_tool', False)
                resolutions_list = agent_cfg.get('image_resolutions', [None])
                
                for iterations in iterations_list:
                    for max_images in max_images_list:
                        for resolution in resolutions_list:
                            # Create descriptive output filename
                            max_img_str = "unlimited" if max_images is None else f"img{max_images}"
                            note_str = "_note" if enable_note_tool else ""
                            res_str = "" if resolution is None else f"_res{resolution}"
                            
                            # If only single values, use simpler naming
                            if len(iterations_list) == 1 and len(max_images_list) == 1 and not enable_note_tool and len(resolutions_list) == 1:
                                output_filename = f"{benchmark}_{model}_agent.jsonl"
                            else:
                                output_filename = f"{benchmark}_{model}_agent_iter{iterations}_{max_img_str}{note_str}{res_str}.jsonl"
                            
                            output_path = output_dir / output_filename
                            
                            run = ExperimentRun(
                                benchmark=benchmark,
                                dataset=dataset,
                                filedir=filedir,
                                model=model,
                                mode="agent",
                                output_path=str(output_path),
                                max_iterations=iterations,
                                max_images_in_context=max_images,
                                enable_note_tool=enable_note_tool,
                                image_resolution=resolution,
                                timeout=config.get('timeout', 300.0),
                                max_retries=config.get('max_retries', 3),
                                max_workers=config.get('max_workers', 1),
                                limit=config.get('limit'),
                                resume=config.get('resume', True),
                                log_dir=log_dir
                            )
                            runs.append(run)
            
            # Random patch agent mode - similar to agent but with random patch viewing
            if modes_config.get('random_patch_agent', {}).get('enabled', False):
                random_patch_cfg = modes_config['random_patch_agent']
                iterations_list = random_patch_cfg.get('iterations', [15])
                resolutions_list = random_patch_cfg.get('image_resolutions', [None])
                
                for iterations in iterations_list:
                    for resolution in resolutions_list:
                        # Create descriptive output filename
                        res_str = "" if resolution is None else f"_res{resolution}"
                        
                        # If only single values, use simpler naming
                        if len(iterations_list) == 1 and len(resolutions_list) == 1:
                            output_filename = f"{benchmark}_{model}_random_patch_agent.jsonl"
                        else:
                            output_filename = f"{benchmark}_{model}_random_patch_agent_iter{iterations}{res_str}.jsonl"
                        
                        output_path = output_dir / output_filename
                        
                        run = ExperimentRun(
                            benchmark=benchmark,
                            dataset=dataset,
                            filedir=filedir,
                            model=model,
                            mode="random_patch_agent",
                            output_path=str(output_path),
                            max_iterations=iterations,
                            image_resolution=resolution,
                            patch_dir=patch_dir,  # Required for random patch agent (to get h5 files)
                            timeout=config.get('timeout', 300.0),
                            max_retries=config.get('max_retries', 3),
                            max_workers=config.get('max_workers', 1),
                            limit=config.get('limit'),
                            resume=config.get('resume', True),
                            log_dir=log_dir
                        )
                        runs.append(run)
            
            # Patch mode
            if modes_config.get('patch', {}).get('enabled', False):
                patch_cfg = modes_config['patch']
                output_filename = f"{benchmark}_{model}_patch.jsonl"
                output_path = output_dir / output_filename
                
                run = ExperimentRun(
                    benchmark=benchmark,
                    dataset=dataset,
                    filedir=filedir,
                    model=model,
                    mode="patch",
                    output_path=str(output_path),
                    patch_dir=patch_dir,
                    num_patches=patch_cfg.get('num_patches'),
                    timeout=config.get('timeout', 300.0),
                    max_retries=config.get('max_retries', 3),
                    max_workers=config.get('max_workers', 1),
                    limit=config.get('limit'),
                    resume=config.get('resume', True),
                    log_dir=log_dir
                )
                runs.append(run)
            
            # Thumbnail mode
            if modes_config.get('thumbnail', {}).get('enabled', False):
                thumbnail_cfg = modes_config['thumbnail']
                resolutions_list = thumbnail_cfg.get('image_resolutions', [None])
                
                for resolution in resolutions_list:
                    # Create descriptive output filename
                    res_str = "" if resolution is None else f"_res{resolution}"
                    
                    # If only single value, use simpler naming
                    if len(resolutions_list) == 1:
                        output_filename = f"{benchmark}_{model}_thumbnail.jsonl"
                    else:
                        output_filename = f"{benchmark}_{model}_thumbnail{res_str}.jsonl"
                    
                    output_path = output_dir / output_filename
                    
                    run = ExperimentRun(
                        benchmark=benchmark,
                        dataset=dataset,
                        filedir=filedir,
                        model=model,
                        mode="thumbnail",
                        output_path=str(output_path),
                        image_resolution=resolution,
                        timeout=config.get('timeout', 300.0),
                        max_retries=config.get('max_retries', 3),
                        max_workers=config.get('max_workers', 1),
                        limit=config.get('limit'),
                        resume=config.get('resume', True),
                        log_dir=log_dir
                    )
                    runs.append(run)
    
    return runs


def execute_run(run: ExperimentRun):
    """Execute a single experiment run."""
    print(f"\n{'='*80}")
    print(f"STARTING RUN: {Path(run.output_path).name}")
    print(f"  Benchmark: {run.benchmark}")
    print(f"  Model: {run.model}")
    print(f"  Mode: {run.mode}")
    if run.mode == "agent":
        print(f"  Iterations: {run.max_iterations}")
        print(f"  Max images in context: {run.max_images_in_context or 'unlimited'}")
        print(f"  Note tool: {'enabled' if run.enable_note_tool else 'disabled'}")
        print(f"  Image resolution: {run.image_resolution or 'default'}")
    elif run.mode == "random_patch_agent":
        print(f"  Iterations: {run.max_iterations}")
        print(f"  Image resolution: {run.image_resolution or 'default'}")
    elif run.mode == "patch":
        print(f"  Num patches: {run.num_patches or 'all'}")
    elif run.mode == "thumbnail":
        print(f"  Image resolution: {run.image_resolution or 'default (1024)'}")
    print(f"  Output: {run.output_path}")
    print(f"{'='*80}\n")
    
    # Validate benchmark
    if run.benchmark not in BENCHMARK_CONFIGS:
        raise SystemExit(f"Unknown benchmark: {run.benchmark}")
    
    # Validate patch mode requirements
    if run.mode == "patch" and not run.patch_dir:
        raise SystemExit(f"patch_dir is required for patch mode in benchmark {run.benchmark}")
    
    # Validate random patch agent mode requirements
    if run.mode == "random_patch_agent" and not run.patch_dir:
        raise SystemExit(f"patch_dir is required for random_patch_agent mode in benchmark {run.benchmark}")
    
    # Get benchmark configuration
    benchmark_config = BENCHMARK_CONFIGS[run.benchmark]
    
    patch_dir = Path(run.patch_dir) if run.patch_dir else None
    
    # Load dataset
    df = pd.read_csv(run.dataset)
    
    # Use benchmark-specific column mapping
    rename_map = benchmark_config["column_mapping"]
    have = [c for c in rename_map if c in df.columns]
    df = df[have].rename(columns={c: rename_map[c] for c in have}).reset_index(drop=True)
    
    # Handle file_name extraction from path if needed
    if benchmark_config.get("file_name_from_path", False) and "file_name" in df.columns:
        df["file_name"] = df["file_name"].apply(lambda x: Path(x).name if pd.notna(x) else x)
    
    required = ["file_name"]
    for r in required:
        if r not in df.columns:
            raise SystemExit(f"Missing required column '{r}'. Found columns: {list(df.columns)}")
    
    if run.limit:
        df = df.head(run.limit).reset_index(drop=True)
    
    out_path = Path(run.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    filedir = Path(run.filedir)
    log_dir = Path(run.log_dir) if run.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Read processed items for resume
    processed = set()
    if run.resume and out_path.exists():
        processed = read_processed_keys(out_path, mode=run.mode)
        if run.mode == "patch":
            logging.info(f"Resume enabled: found {len(processed)} already-processed patches")
        else:
            logging.info(f"Resume enabled: found {len(processed)} already-processed items")
    
    # Expand dataset to patch level if in patch mode
    if run.mode == "patch":
        logging.info(f"Expanding dataset to patch level...")
        df = expand_to_patches(df, filedir, patch_dir, run.num_patches, processed_patches=processed if run.resume else None)
        logging.info(f"Total patches to process: {len(df)}")
        print(f"\n{'='*60}")
        print(f"PATCH MODE: Expanded to {len(df)} total patches")
        print(f"{'='*60}\n")
    elif run.resume:
        # For agent/thumbnail/random_patch_agent mode, filter based on file_id (unique identifier for VQA)
        before = len(df)
        df = df[~df["file_id"].isin(processed)].reset_index(drop=True)
        logging.info(f"Resume enabled: skipped {before - len(df)} already-processed rows.")
    
    # Process rows
    rows = list(iter_rows(df))
    
    def _write_jsonl(rec: AgentResult):
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    
    prompt_template = benchmark_config["prompt"]
    
    logging.info(f"Running in {run.mode} mode with {run.max_iterations} max iterations per case")
    
    if run.max_workers == 1:
        for row in tqdm(rows, desc="Processing", unit="item"):
            res = run_one(
                row, filedir, prompt_template, run.benchmark, run.timeout, 
                run.max_iterations, run.max_retries, run.model, 
                run.max_images_in_context, run.enable_note_tool, run.mode, log_dir, 
                run.image_resolution, patch_dir
            )
            _write_jsonl(res)
            retry_info = f" (retries: {res.retry_count})" if res.retry_count > 0 else ""
            
            # Display info based on mode
            if run.mode == "patch":
                patch_info = f" [patch {row.get('patch_idx', '?')} at ({row.get('patch_x', '?')},{row.get('patch_y', '?')})]"
                tqdm.write(f"{res.file_name}{patch_info} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                logging.info(f"{res.file_name}{patch_info} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
            else:
                tqdm.write(f"{res.file_name} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                logging.info(f"{res.file_name} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=run.max_workers) as ex:
            fut_to_row = {
                ex.submit(
                    run_one, row, filedir, prompt_template, run.benchmark, run.timeout,
                    run.max_iterations, run.max_retries, run.model,
                    run.max_images_in_context, run.enable_note_tool, run.mode, log_dir,
                    run.image_resolution, patch_dir
                ): row for row in rows
            }
            
            with tqdm(total=len(rows), desc="Processing", unit="item") as pbar:
                for fut in as_completed(fut_to_row):
                    res = fut.result()
                    row = fut_to_row[fut]
                    _write_jsonl(res)
                    retry_info = f" (retries: {res.retry_count})" if res.retry_count > 0 else ""
                    
                    # Display info based on mode
                    if run.mode == "patch":
                        patch_info = f" [patch {row.get('patch_idx', '?')} at ({row.get('patch_x', '?')},{row.get('patch_y', '?')})]"
                        tqdm.write(f"{res.file_name}{patch_info} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                        logging.info(f"{res.file_name}{patch_info} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                    else:
                        tqdm.write(f"{res.file_name} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                        logging.info(f"{res.file_name} -> {res.status} ({res.duration_s:.2f}s){retry_info}")
                    pbar.update(1)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED RUN: {Path(run.output_path).name}")
    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser(description="Benchmark PathologyAgentStreaming using YAML configuration.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                   help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s %(levelname)s %(message)s")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    
    logging.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Get num_repeat parameter (default to 1 if not specified or None)
    num_repeat = config.get('num_repeat')
    if num_repeat is None:
        num_repeat = 1
    
    # Store base output_dir
    base_output_dir = config.get('output_dir', 'results')
    
    # Track total runs across all repeats
    total_completed_runs = 0
    
    # Loop over repeats
    for repeat_num in range(1, num_repeat + 1):
        # Modify output_dir for this repeat
        if num_repeat > 1:
            config['output_dir'] = f"{base_output_dir}_{repeat_num}"
            repeat_info = f" (Repeat {repeat_num}/{num_repeat})"
        else:
            config['output_dir'] = base_output_dir
            repeat_info = ""
        
        # Generate experiment runs for this repeat
        runs = generate_runs(config)
        
        logging.info(f"Generated {len(runs)} experiment runs{repeat_info}")
        
        # Display summary of runs
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY{repeat_info}")
        print(f"{'='*80}")
        if num_repeat > 1:
            print(f"Repeat: {repeat_num}/{num_repeat}")
        print(f"Total runs: {len(runs)}")
        print(f"Models: {config.get('models', [])}")
        print(f"Benchmarks: {[b['name'] for b in config.get('benchmarks', [])]}")
        enabled_modes = [mode for mode, cfg in config.get('modes', {}).items() if cfg.get('enabled', False)]
        print(f"Enabled modes: {enabled_modes}")
        print(f"Output directory: {config['output_dir']}")
        print(f"{'='*80}\n")
        
        # Execute all runs for this repeat
        for i, run in enumerate(runs, 1):
            logging.info(f"Starting run {i}/{len(runs)}{repeat_info}: {Path(run.output_path).name}")
            try:
                execute_run(run)
                total_completed_runs += 1
            except Exception as e:
                logging.error(f"Run failed: {e}")
                raise
        
        print(f"\n{'='*80}")
        print(f"COMPLETED REPEAT {repeat_num}/{num_repeat}")
        print(f"{'='*80}")
        print(f"Completed {len(runs)} runs for this repeat")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    if num_repeat > 1:
        print(f"Completed {num_repeat} repeats")
    print(f"Total runs completed: {total_completed_runs}")
    print(f"{'='*80}\n")
    
    logging.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
